# pylint: disable=too-many-lines  # Reason: file service consolidates upload, list, delete, presign, and storage backend logic
"""
User-level file management services.

Provides business logic for managing files across all workflow runs for a user.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from tortoise.functions import Count, Max, Min, Sum

from seer.api.files import models as api_models
from seer.config import config
from seer.database import Organization, OrganizationMembership, User, WorkflowFile
from seer.database.organization_models import OrganizationRole
from seer.database.workflow_models import make_run_public_id
from seer.logger import get_logger

if TYPE_CHECKING:
    pass


async def _can_view_file(
    user: User,
    file: WorkflowFile,
    membership: Optional[OrganizationMembership] = None,
) -> bool:
    """Check if user can view file based on ownership or org membership."""
    if file.user_id == user.id:
        return True
    if file.organization_id:
        if membership is None:
            membership = await OrganizationMembership.get_or_none(
                organization_id=file.organization_id, user=user
            )
        return membership is not None
    return False


async def _can_manage_file(
    user: User,
    file: WorkflowFile,
    membership: Optional[OrganizationMembership] = None,
) -> bool:
    """Check if user can delete file."""
    if file.user_id == user.id:
        return True
    if file.organization_id:
        if membership is None:
            membership = await OrganizationMembership.get_or_none(
                organization_id=file.organization_id, user=user
            )
        if membership and membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
            return True
    return False


async def _get_file_org_scoped(
    user: User,
    file_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
    require_manage: bool = False,
) -> WorkflowFile:
    """Get file with org-scoped access control."""
    if organization:
        file = await WorkflowFile.filter(file_id=file_id, organization=organization).first()
    else:
        file = await WorkflowFile.filter(file_id=file_id, user=user).first()

    if not file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{file_id}' not found")

    if require_manage:
        has_access = await _can_manage_file(user, file, membership)
    else:
        has_access = await _can_view_file(user, file, membership)

    if not has_access:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{file_id}' not found")

    return file

logger = get_logger("seer.api.files.services")


@dataclass
class FileListParams:  # pylint: disable=too-many-instance-attributes  # DTO needs all filter/pagination fields
    """Parameters for listing user files with filtering and pagination."""

    limit: int = 50
    cursor: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None
    source_tool: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_size_bytes: Optional[int] = None
    max_size_bytes: Optional[int] = None
    sort_by: str = "created_at"
    sort_order: str = "desc"


def _format_size_human(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _apply_content_filters(query, params: FileListParams):
    """Apply MIME type, filename, and source tool filters to query."""
    if params.mime_type:
        if params.mime_type.endswith("/*"):
            prefix = params.mime_type[:-2]
            query = query.filter(mime_type__startswith=prefix)
        else:
            query = query.filter(mime_type=params.mime_type)

    if params.filename:
        query = query.filter(filename__icontains=params.filename)

    if params.source_tool:
        query = query.filter(source_tool=params.source_tool)

    return query


def _apply_date_size_filters(query, params: FileListParams):
    """Apply date range and size filters to query."""
    if params.created_after:
        query = query.filter(created_at__gte=params.created_after)

    if params.created_before:
        query = query.filter(created_at__lte=params.created_before)

    if params.min_size_bytes is not None:
        query = query.filter(size_bytes__gte=params.min_size_bytes)

    if params.max_size_bytes is not None:
        query = query.filter(size_bytes__lte=params.max_size_bytes)

    return query


def _apply_sorting(query, params: FileListParams):
    """Apply sorting to query based on params."""
    sort_field_map = {"created_at": "created_at", "size_bytes": "size_bytes", "filename": "filename"}
    sort_field = sort_field_map.get(params.sort_by, "created_at")
    order_prefix = "-" if params.sort_order == "desc" else ""
    return query.order_by(f"{order_prefix}{sort_field}")


def _file_to_list_item(f: WorkflowFile) -> api_models.UserFileListItem:
    """Convert a WorkflowFile to a UserFileListItem."""
    run = f.workflow_run
    workflow = getattr(run, "workflow", None) if run else None
    return api_models.UserFileListItem(
        file_id=f.file_id,
        filename=f.filename,
        mime_type=f.mime_type,
        size_bytes=f.size_bytes,
        size_human=f.size_human,
        run_id=make_run_public_id(run.id) if run else None,
        workflow_id=workflow.workflow_id if workflow else None,
        workflow_name=workflow.name if workflow else None,
        source_node_id=f.source_node_id,
        source_tool=f.source_tool,
        created_at=f.created_at,
    )


# pylint: disable=too-many-arguments,too-many-locals  # API service receives query params from router
async def list_user_files(
    user: User,
    *,
    organization: Optional[Organization] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    source_tool: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    min_size_bytes: Optional[int] = None,
    max_size_bytes: Optional[int] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> api_models.UserFileListResponse:
    """
    List all files for a user with filtering, sorting, and pagination.

    Args:
        user: Authenticated user.
        limit: Max items to return (1-100).
        cursor: Pagination cursor (file_id).
        mime_type: Filter by MIME type (supports wildcards like "image/*").
        filename: Partial filename match (case-insensitive).
        source_tool: Filter by tool that created the file.
        created_after: Filter files created after this date.
        created_before: Filter files created before this date.
        min_size_bytes: Minimum file size filter.
        max_size_bytes: Maximum file size filter.
        sort_by: Sort field (created_at, size_bytes, filename).
        sort_order: Sort direction (asc, desc).

    Returns:
        Paginated list of file metadata.
    """
    params = FileListParams(
        limit=max(1, min(limit, 100)),
        cursor=cursor,
        mime_type=mime_type,
        filename=filename,
        source_tool=source_tool,
        created_after=created_after,
        created_before=created_before,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    if organization:
        query = WorkflowFile.filter(organization=organization)
    else:
        query = WorkflowFile.filter(user=user)
    query = _apply_content_filters(query, params)
    query = _apply_date_size_filters(query, params)

    # Handle cursor pagination
    if params.cursor:
        if organization:
            cursor_file = await WorkflowFile.filter(file_id=params.cursor, organization=organization).first()
        else:
            cursor_file = await WorkflowFile.filter(file_id=params.cursor, user=user).first()
        if cursor_file:
            if params.sort_order == "desc":
                query = query.filter(created_at__lt=cursor_file.created_at)
            else:
                query = query.filter(created_at__gt=cursor_file.created_at)

    query = _apply_sorting(query, params)

    # Fetch files with related data
    files = await query.limit(params.limit + 1).prefetch_related("workflow_run", "workflow_run__workflow")

    # Get total count and size (without pagination filters)
    if organization:
        base_query = WorkflowFile.filter(organization=organization)
    else:
        base_query = WorkflowFile.filter(user=user)
    total_count = await base_query.count()
    total_size_result = await base_query.annotate(total=Sum("size_bytes")).values("total")
    total_size_bytes = total_size_result[0]["total"] or 0 if total_size_result else 0

    items = [_file_to_list_item(f) for f in files[:params.limit]]
    next_cursor = items[-1].file_id if len(files) > params.limit else None

    return api_models.UserFileListResponse(
        files=items,
        total_count=total_count,
        total_size_bytes=total_size_bytes,
        next_cursor=next_cursor,
    )


async def get_user_storage_stats(
    user: User,
    organization: Optional[Organization] = None,
) -> api_models.UserStorageStatsResponse:
    """
    Get storage statistics for a user or organization.

    Args:
        user: Authenticated user.
        organization: Optional organization context for team stats.

    Returns:
        Storage statistics including totals and breakdowns.
    """
    if organization:
        base_query = WorkflowFile.filter(organization=organization)
    else:
        base_query = WorkflowFile.filter(user=user)

    # Get aggregate stats
    agg_result = await base_query.annotate(
        total_files=Count("id"),
        total_size=Sum("size_bytes"),
        oldest=Min("created_at"),
        newest=Max("created_at"),
    ).values("total_files", "total_size", "oldest", "newest")

    if not agg_result:
        return api_models.UserStorageStatsResponse(
            total_files=0,
            total_size_bytes=0,
            total_size_human="0 B",
            files_by_mime_type=[],
            files_by_tool=[],
        )

    stats = agg_result[0]
    total_files = stats["total_files"] or 0
    total_size_bytes = stats["total_size"] or 0

    # Get breakdown by MIME type
    mime_stats = await base_query.annotate(
        file_count=Count("id"),
        size_total=Sum("size_bytes"),
    ).group_by("mime_type").values("mime_type", "file_count", "size_total")

    files_by_mime_type = [
        api_models.MimeTypeStats(
            mime_type=m["mime_type"],
            file_count=m["file_count"],
            total_size_bytes=m["size_total"] or 0,
            total_size_human=_format_size_human(m["size_total"] or 0),
        )
        for m in mime_stats
    ]

    # Get breakdown by source tool (excluding nulls)
    tool_stats = await base_query.exclude(source_tool=None).annotate(
        file_count=Count("id"),
        size_total=Sum("size_bytes"),
    ).group_by("source_tool").values("source_tool", "file_count", "size_total")

    files_by_tool = [
        api_models.ToolStats(
            source_tool=t["source_tool"],
            file_count=t["file_count"],
            total_size_bytes=t["size_total"] or 0,
        )
        for t in tool_stats
    ]

    return api_models.UserStorageStatsResponse(
        total_files=total_files,
        total_size_bytes=total_size_bytes,
        total_size_human=_format_size_human(total_size_bytes),
        files_by_mime_type=files_by_mime_type,
        files_by_tool=files_by_tool,
        oldest_file_date=stats["oldest"],
        newest_file_date=stats["newest"],
    )


async def get_user_file(
    user: User,
    file_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.UserFileResponse:
    """
    Get metadata for a specific file.

    Args:
        user: Authenticated user.
        file_id: File UUID.
        organization: Optional organization context for team access.
        membership: Optional organization membership.

    Returns:
        File metadata.

    Raises:
        HTTPException: If file not found or access denied.
    """
    file = await _get_file_org_scoped(user, file_id, organization, membership)
    await file.fetch_related("workflow_run", "workflow_run__workflow")

    run = file.workflow_run
    workflow = getattr(run, "workflow", None) if run else None

    return api_models.UserFileResponse(
        file=api_models.UserFileListItem(
            file_id=file.file_id,
            filename=file.filename,
            mime_type=file.mime_type,
            size_bytes=file.size_bytes,
            size_human=file.size_human,
            run_id=make_run_public_id(run.id) if run else None,
            workflow_id=workflow.workflow_id if workflow else None,
            workflow_name=workflow.name if workflow else None,
            source_node_id=file.source_node_id,
            source_tool=file.source_tool,
            created_at=file.created_at,
        )
    )


async def get_user_file_download_url(
    user: User,
    file_id: str,
    inline: bool = False,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.UserFileDownloadResponse:
    """
    Get a presigned URL to download or preview a file.

    Args:
        user: Authenticated user.
        file_id: File UUID.
        inline: If True, returns URL for inline preview instead of download.
        organization: Optional organization context for team access.
        membership: Optional organization membership.

    Returns:
        Presigned download URL.

    Raises:
        HTTPException: If file not found, access denied, or storage not configured.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    from seer.core.files.service import WorkflowFileSystem, file_to_ref

    file = await _get_file_org_scoped(user, file_id, organization, membership)

    if not config.is_workflow_file_system_configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Workflow file storage is not configured")

    fs = WorkflowFileSystem.instance()
    expires_seconds = config.workflow_file_presigned_url_expiry_seconds
    download_url = await fs.get_presigned_url(file_to_ref(file), expires_seconds, inline=inline)

    return api_models.UserFileDownloadResponse(
        file_id=file_id, filename=file.filename, download_url=download_url, expires_in_seconds=expires_seconds
    )


async def get_user_file_content(
    user: User,
    file_id: str,
    max_size_bytes: int,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> StreamingResponse:
    """
    Stream file content directly for preview.

    Args:
        user: Authenticated user.
        file_id: File UUID.
        max_size_bytes: Maximum file size allowed for preview.
        organization: Optional organization context for team access.
        membership: Optional organization membership.

    Returns:
        StreamingResponse with file content.

    Raises:
        HTTPException: If file not found, too large, or storage not configured.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    from seer.core.files.service import WorkflowFileSystem, file_to_ref

    file = await _get_file_org_scoped(user, file_id, organization, membership)

    if file.size_bytes > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large for preview (max {max_size_bytes // (1024 * 1024)}MB)"
        )

    if not config.is_workflow_file_system_configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Workflow file storage is not configured")

    fs = WorkflowFileSystem.instance()
    file_ref = file_to_ref(file)

    # Retrieve file content (size already checked above)
    content = await fs.get_file_content(file_ref)

    async def content_generator():
        """Yield content as a single chunk."""
        yield content

    return StreamingResponse(
        content_generator(),
        media_type=file.mime_type,
        headers={
            "Content-Disposition": f'inline; filename="{file.filename}"',
            "Content-Length": str(file.size_bytes),
            "Cache-Control": "private, max-age=3600",
        }
    )


async def delete_user_file(
    user: User,
    file_id: str,
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.UserFileDeleteResponse:
    """
    Delete a file.

    Args:
        user: Authenticated user.
        file_id: File UUID.
        organization: Optional organization context for team access.
        membership: Optional organization membership.

    Returns:
        Deletion confirmation.

    Raises:
        HTTPException: If file not found or access denied.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    from seer.core.files.service import WorkflowFileSystem, file_to_ref

    file = await _get_file_org_scoped(user, file_id, organization, membership, require_manage=True)

    # Delete from storage if configured
    deleted_from_storage = False
    if config.is_workflow_file_system_configured:
        try:
            fs = WorkflowFileSystem.instance()
            deleted_from_storage = await fs.delete_file(file_to_ref(file))
        except OSError as e:
            logger.warning("Failed to delete file from storage: %s", e)

    # Delete metadata from database
    await file.delete()
    logger.info("Deleted file %s (storage: %s)", file_id, deleted_from_storage)

    return api_models.UserFileDeleteResponse(file_id=file_id, deleted=True)


async def bulk_delete_user_files(
    user: User,
    file_ids: list[str],
    organization: Optional[Organization] = None,
    membership: Optional[OrganizationMembership] = None,
) -> api_models.BulkDeleteFilesResponse:
    """
    Delete multiple files at once.

    Args:
        user: Authenticated user.
        file_ids: List of file UUIDs to delete.

    Returns:
        Results for each file.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    from seer.core.files.service import WorkflowFileSystem, file_to_ref

    results = []
    deleted_count = 0
    failed_count = 0
    total_size_freed = 0

    fs = WorkflowFileSystem.instance() if config.is_workflow_file_system_configured else None

    for file_id in file_ids:
        try:
            file = await _get_file_org_scoped(user, file_id, organization, membership, require_manage=True)
        except HTTPException:
            results.append(api_models.BulkDeleteResult(file_id=file_id, deleted=False, error="File not found"))
            failed_count += 1
            continue

        size = file.size_bytes

        # Delete from storage
        if fs:
            try:
                await fs.delete_file(file_to_ref(file))
            except OSError as e:
                logger.warning("Failed to delete file %s from storage: %s", file_id, e)

        # Delete from database
        try:
            await file.delete()
            results.append(api_models.BulkDeleteResult(
                file_id=file_id,
                deleted=True,
            ))
            deleted_count += 1
            total_size_freed += size
        except Exception as e:  # pylint: disable=broad-exception-caught  # Catch all to report per-file errors
            logger.error("Failed to delete file %s: %s", file_id, e)
            results.append(api_models.BulkDeleteResult(
                file_id=file_id,
                deleted=False,
                error=str(e),
            ))
            failed_count += 1

    return api_models.BulkDeleteFilesResponse(
        results=results,
        deleted_count=deleted_count,
        failed_count=failed_count,
        total_size_freed_bytes=total_size_freed,
    )


async def search_user_files(
    user: User,
    query: str,
    limit: int = 20,
    organization: Optional[Organization] = None,
) -> api_models.FileSearchResponse:
    """
    Search files by filename.

    Args:
        user: Authenticated user.
        query: Search query (matches filename).
        limit: Max results to return.
        organization: Optional organization context for team access.

    Returns:
        Matching files.
    """
    limit = max(1, min(limit, 50))

    # Search by filename (case-insensitive partial match)
    if organization:
        files = await WorkflowFile.filter(
            organization=organization,
            filename__icontains=query,
        ).prefetch_related(
            "workflow_run", "workflow_run__workflow"
        ).order_by("-created_at").limit(limit)
        total_matches = await WorkflowFile.filter(
            organization=organization,
            filename__icontains=query,
        ).count()
    else:
        files = await WorkflowFile.filter(
            user=user,
            filename__icontains=query,
        ).prefetch_related(
            "workflow_run", "workflow_run__workflow"
        ).order_by("-created_at").limit(limit)
        total_matches = await WorkflowFile.filter(
            user=user,
            filename__icontains=query,
        ).count()

    items = []
    for f in files:
        run = f.workflow_run
        workflow = getattr(run, "workflow", None) if run else None
        items.append(
            api_models.UserFileListItem(
                file_id=f.file_id,
                filename=f.filename,
                mime_type=f.mime_type,
                size_bytes=f.size_bytes,
                size_human=f.size_human,
                run_id=make_run_public_id(run.id) if run else None,
                workflow_id=workflow.workflow_id if workflow else None,
                workflow_name=workflow.name if workflow else None,
                source_node_id=f.source_node_id,
                source_tool=f.source_tool,
                created_at=f.created_at,
            )
        )

    return api_models.FileSearchResponse(
        query=query,
        results=items,
        total_matches=total_matches,
    )
