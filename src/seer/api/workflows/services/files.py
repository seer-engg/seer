"""
Workflow file management services.

Provides API layer for managing files created during workflow execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from seer.api.workflows import models as api_models
from seer.database import User, WorkflowFile, WorkflowRun, parse_run_public_id
from seer.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("seer.api.workflows.services.files")


async def list_run_files(user: User, run_id: str) -> api_models.WorkflowFileListResponse:
    """
    List all files created during a workflow run.

    Args:
        user: Authenticated user.
        run_id: Workflow run ID (e.g., "run_123").

    Returns:
        List of file metadata.

    Raises:
        HTTPException: If run not found or access denied.
    """
    run_pk = parse_run_public_id(run_id)

    # Verify user owns the run
    run = await WorkflowRun.filter(id=run_pk, user=user).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )

    # Get all files for the run
    files = await WorkflowFile.filter(workflow_run_id=run_pk).order_by("-created_at")

    file_items = [
        api_models.WorkflowFileItem(
            file_id=f.file_id,
            filename=f.filename,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
            size_human=f.size_human,
            source_node_id=f.source_node_id,
            source_tool=f.source_tool,
            created_at=f.created_at,
        )
        for f in files
    ]

    total_size = sum(f.size_bytes for f in files)

    return api_models.WorkflowFileListResponse(
        run_id=run_id,
        files=file_items,
        total_count=len(file_items),
        total_size_bytes=total_size,
    )


async def get_run_file(user: User, run_id: str, file_id: str) -> api_models.WorkflowFileResponse:
    """
    Get metadata for a specific file.

    Args:
        user: Authenticated user.
        run_id: Workflow run ID.
        file_id: File UUID.

    Returns:
        File metadata.

    Raises:
        HTTPException: If file not found or access denied.
    """
    run_pk = parse_run_public_id(run_id)

    # Verify user owns the run
    run = await WorkflowRun.filter(id=run_pk, user=user).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )

    # Get the file
    file = await WorkflowFile.filter(file_id=file_id, workflow_run_id=run_pk).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found"
        )

    return api_models.WorkflowFileResponse(
        file=api_models.WorkflowFileItem(
            file_id=file.file_id,
            filename=file.filename,
            mime_type=file.mime_type,
            size_bytes=file.size_bytes,
            size_human=file.size_human,
            source_node_id=file.source_node_id,
            source_tool=file.source_tool,
            created_at=file.created_at,
        )
    )


async def get_run_file_download_url(
    user: User, run_id: str, file_id: str
) -> api_models.WorkflowFileDownloadResponse:
    """
    Get a presigned URL to download a file.

    Args:
        user: Authenticated user.
        run_id: Workflow run ID.
        file_id: File UUID.

    Returns:
        Presigned download URL.

    Raises:
        HTTPException: If file not found, access denied, or file system not configured.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports with config/files modules
    from seer.config import config
    from seer.core.files.models import WorkflowFileRef
    from seer.core.files.service import WorkflowFileSystem

    run_pk = parse_run_public_id(run_id)

    # Verify user owns the run
    run = await WorkflowRun.filter(id=run_pk, user=user).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )

    # Get the file
    file = await WorkflowFile.filter(file_id=file_id, workflow_run_id=run_pk).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found"
        )

    # Check if file system is configured
    if not config.is_workflow_file_system_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow file storage is not configured"
        )

    # Create a file reference for the file system
    file_ref = WorkflowFileRef(
        file_id=file.file_id,
        storage_path=file.storage_path,
        filename=file.filename,
        mime_type=file.mime_type,
        size_bytes=file.size_bytes,
        workflow_run_id=run_id,
        created_at=file.created_at,
        md5_hash=file.md5_hash,
    )

    # Get presigned URL
    fs = WorkflowFileSystem.instance()
    expires_seconds = config.workflow_file_presigned_url_expiry_seconds
    download_url = await fs.get_presigned_url(file_ref, expires_seconds)

    return api_models.WorkflowFileDownloadResponse(
        file_id=file_id,
        filename=file.filename,
        download_url=download_url,
        expires_in_seconds=expires_seconds,
    )


async def delete_run_file(
    user: User, run_id: str, file_id: str
) -> api_models.WorkflowFileDeleteResponse:
    """
    Delete a file from a workflow run.

    Args:
        user: Authenticated user.
        run_id: Workflow run ID.
        file_id: File UUID.

    Returns:
        Deletion confirmation.

    Raises:
        HTTPException: If file not found, access denied, or deletion fails.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports with config/files modules
    from seer.config import config
    from seer.core.files.models import WorkflowFileRef
    from seer.core.files.service import WorkflowFileSystem

    run_pk = parse_run_public_id(run_id)

    # Verify user owns the run
    run = await WorkflowRun.filter(id=run_pk, user=user).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found"
        )

    # Get the file
    file = await WorkflowFile.filter(file_id=file_id, workflow_run_id=run_pk).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found"
        )

    # Delete from storage if configured
    deleted_from_storage = False
    if config.is_workflow_file_system_configured:
        try:
            file_ref = WorkflowFileRef(
                file_id=file.file_id,
                storage_path=file.storage_path,
                filename=file.filename,
                mime_type=file.mime_type,
                size_bytes=file.size_bytes,
                workflow_run_id=run_id,
                created_at=file.created_at,
                md5_hash=file.md5_hash,
            )
            fs = WorkflowFileSystem.instance()
            deleted_from_storage = await fs.delete_file(file_ref)
        except OSError as e:
            logger.warning("Failed to delete file from storage: %s", e)
            # Continue to delete metadata even if storage deletion fails

    # Delete metadata from database
    await file.delete()
    logger.info("Deleted file %s from run %s (storage: %s)", file_id, run_id, deleted_from_storage)

    return api_models.WorkflowFileDeleteResponse(
        file_id=file_id,
        deleted=True,
    )
