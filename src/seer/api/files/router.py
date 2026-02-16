"""
User-level file management API router.

Provides endpoints for listing, downloading, uploading, and deleting files
across all workflow runs for a user.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from seer.api.files import models as api_models
from seer.api.files import services
from seer.database import User
from seer.logger import get_logger

logger = get_logger("seer.api.files.router")

router = APIRouter(prefix="/v1/files", tags=["files"])


def _require_user(request: Request) -> User:
    """Extract authenticated user from request or raise 401."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


# pylint: disable=invalid-name  # Enum values use lowercase to match API query parameter values
class SortByField(str, Enum):
    """Sort field options for file listing."""

    created_at = "created_at"
    size_bytes = "size_bytes"
    filename = "filename"


class SortOrder(str, Enum):
    """Sort order options."""

    asc = "asc"
    desc = "desc"
# pylint: enable=invalid-name


# pylint: disable=too-many-arguments,too-many-positional-arguments  # FastAPI injects query parameters as function arguments
@router.get("", response_model=api_models.UserFileListResponse)
async def list_files(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Maximum number of files to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (file_id)"),
    mime_type: Optional[str] = Query(None, description="Filter by MIME type (supports wildcards like 'image/*')"),
    filename: Optional[str] = Query(None, description="Filter by filename (partial match, case-insensitive)"),
    source_tool: Optional[str] = Query(None, description="Filter by tool that created the file"),
    created_after: Optional[datetime] = Query(None, description="Filter files created after this date"),
    created_before: Optional[datetime] = Query(None, description="Filter files created before this date"),
    min_size_bytes: Optional[int] = Query(None, ge=0, description="Minimum file size in bytes"),
    max_size_bytes: Optional[int] = Query(None, ge=0, description="Maximum file size in bytes"),
    sort_by: SortByField = Query(SortByField.created_at, description="Field to sort by"),
    sort_order: SortOrder = Query(SortOrder.desc, description="Sort direction"),
) -> api_models.UserFileListResponse:
    """
    List all files for the authenticated user.

    Supports filtering by MIME type, filename, source tool, date range, and size.
    Results are paginated using cursor-based pagination.
    """
    user = _require_user(request)
    return await services.list_user_files(
        user,
        limit=limit,
        cursor=cursor,
        mime_type=mime_type,
        filename=filename,
        source_tool=source_tool,
        created_after=created_after,
        created_before=created_before,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
        sort_by=sort_by.value,
        sort_order=sort_order.value,
    )
# pylint: enable=too-many-arguments,too-many-positional-arguments


@router.get("/stats", response_model=api_models.UserStorageStatsResponse)
async def get_storage_stats(request: Request) -> api_models.UserStorageStatsResponse:
    """
    Get storage statistics for the authenticated user.

    Returns total file count, total size, and breakdowns by MIME type and source tool.
    """
    user = _require_user(request)
    return await services.get_user_storage_stats(user)


@router.get("/search", response_model=api_models.FileSearchResponse)
async def search_files(
    request: Request,
    q: str = Query(..., min_length=1, max_length=255, description="Search query (matches filename)"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
) -> api_models.FileSearchResponse:
    """
    Search files by filename.

    Performs case-insensitive partial matching on filenames.
    """
    user = _require_user(request)
    return await services.search_user_files(user, q, limit)


@router.get("/{file_id}", response_model=api_models.UserFileResponse)
async def get_file(
    request: Request,
    file_id: str,
) -> api_models.UserFileResponse:
    """
    Get metadata for a specific file.

    Returns 404 if the file doesn't exist or doesn't belong to the user.
    """
    user = _require_user(request)
    return await services.get_user_file(user, file_id)


@router.get("/{file_id}/download", response_model=api_models.UserFileDownloadResponse)
async def get_file_download_url(
    request: Request,
    file_id: str,
    inline: bool = Query(False, description="If true, returns URL for inline preview instead of download"),
) -> api_models.UserFileDownloadResponse:
    """
    Get a presigned URL to download a file.

    The URL expires after a configurable time (default: 1 hour).
    Set inline=true to get a URL suitable for in-browser preview.
    Returns 503 if file storage is not configured.
    """
    user = _require_user(request)
    return await services.get_user_file_download_url(user, file_id, inline=inline)


# Maximum file size for content preview (5MB)
MAX_PREVIEW_SIZE_BYTES = 5 * 1024 * 1024


@router.get("/{file_id}/content")
async def get_file_content(
    request: Request,
    file_id: str,
) -> StreamingResponse:
    """
    Stream file content directly for preview.

    This endpoint proxies file content to avoid CORS issues when previewing files.
    Returns the raw file content with appropriate Content-Type header.
    Limited to files under 5MB for preview purposes.
    Returns 503 if file storage is not configured.
    """
    user = _require_user(request)
    return await services.get_user_file_content(user, file_id, MAX_PREVIEW_SIZE_BYTES)


@router.delete("/{file_id}", response_model=api_models.UserFileDeleteResponse)
async def delete_file(
    request: Request,
    file_id: str,
) -> api_models.UserFileDeleteResponse:
    """
    Delete a file.

    Removes both the file from storage and the database record.
    Returns 404 if the file doesn't exist or doesn't belong to the user.
    """
    user = _require_user(request)
    return await services.delete_user_file(user, file_id)


@router.post("/bulk-delete", response_model=api_models.BulkDeleteFilesResponse)
async def bulk_delete_files(
    request: Request,
    body: api_models.BulkDeleteFilesRequest,
) -> api_models.BulkDeleteFilesResponse:
    """
    Delete multiple files at once.

    Maximum 100 files per request. Returns results for each file,
    including any errors encountered.
    """
    user = _require_user(request)
    return await services.bulk_delete_user_files(user, body.file_ids)


# pylint: disable=too-many-locals,protected-access  # Upload requires multiple local vars; accessing S3 backend internals for direct upload
@router.post("/upload", response_model=api_models.UserFileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    request: Request,
    file: UploadFile,
    filename: Optional[str] = Query(None, description="Override filename (defaults to uploaded filename)"),
) -> api_models.UserFileUploadResponse:
    """
    Upload a file directly.

    Files uploaded via this endpoint are not associated with any workflow run.
    Maximum file size is determined by server configuration (default: 100MB).
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    import uuid
    from datetime import timezone

    from seer.config import config
    from seer.core.files.service import WorkflowFileSystem
    from seer.database import WorkflowFile

    user = _require_user(request)

    if not config.is_workflow_file_system_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow file storage is not configured"
        )

    # Read file content
    content = await file.read()
    size_bytes = len(content)

    # Check file size limit
    max_size = config.workflow_file_max_size_mb * 1024 * 1024
    if size_bytes > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed ({config.workflow_file_max_size_mb} MB)"
        )

    # Determine filename and MIME type
    actual_filename = filename or file.filename or "unnamed"
    mime_type = file.content_type or "application/octet-stream"

    # Generate file ID
    file_id = str(uuid.uuid4())

    # Store in S3
    fs = WorkflowFileSystem.instance()

    # Build storage path for user upload (no run_id)
    # Format: workflow-files/{user_id}/uploads/{file_id}/{filename}
    # Use user.user_id (string like "user_36pO...") for consistency with workflow tools
    user_id = user.user_id

    # Store via backend directly with custom path
    import hashlib
    md5_hash = hashlib.md5(content).hexdigest()

    # Sanitize filename
    safe_filename = actual_filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
    safe_user_id = user_id.replace("/", "_").replace("\\", "_")

    backend = fs.backend
    key = f"{backend.prefix}/{safe_user_id}/uploads/{file_id}/{safe_filename}"

    try:
        await backend._run_sync(
            backend._client.put_object,
            Bucket=backend.bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
        )
    except Exception as e:
        logger.error("Failed to upload file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        ) from e

    storage_path = f"s3://{backend.bucket}/{key}"
    created_at = datetime.now(timezone.utc)

    # Create database record
    db_file = await WorkflowFile.create(
        file_id=file_id,
        user=user,
        workflow_run=None,  # User upload, no run
        storage_path=storage_path,
        filename=actual_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        md5_hash=md5_hash,
        source_node_id=None,
        source_tool="user_upload",
    )

    logger.info("User %s uploaded file %s (%s, %d bytes)", user.id, file_id, actual_filename, size_bytes)

    return api_models.UserFileUploadResponse(
        file_id=file_id,
        filename=actual_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        size_human=db_file.size_human,
        created_at=created_at,
    )
