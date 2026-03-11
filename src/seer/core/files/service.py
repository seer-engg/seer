"""
Workflow File System service.

This module provides the WorkflowFileSystem singleton that tools use to store
and retrieve files. It abstracts the storage backend and provides convenience
methods for common operations.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Optional

from seer.core.files.models import WORKFLOW_FILE_REF_TYPE, WorkflowFileRef, is_file_ref, parse_file_ref
from seer.core.files.storage import FileStorageBackend
from seer.logger import get_logger

if TYPE_CHECKING:
    from seer.database import User, WorkflowFile, WorkflowRun

logger = get_logger("seer.core.files.service")


class WorkflowFileSystem:
    """
    Singleton service for workflow file operations.

    Provides a consistent interface for tools to store/retrieve files
    without knowing the underlying storage backend. The backend is
    configured via environment variables.

    Usage:
        fs = WorkflowFileSystem.instance()
        file_ref = await fs.store_file(run_id, "doc.pdf", data, "application/pdf")
        content = await fs.get_file_content(file_ref)
    """

    _instance: Optional[WorkflowFileSystem] = None

    def __init__(self, backend: FileStorageBackend):
        """
        Initialize with a storage backend.

        Use WorkflowFileSystem.instance() to get the configured singleton.

        Args:
            backend: Storage backend implementation.
        """
        self._backend = backend

    @classmethod
    def instance(cls) -> WorkflowFileSystem:
        """
        Get the singleton instance with configured backend.

        Returns:
            Configured WorkflowFileSystem instance.

        Raises:
            ValueError: If required configuration is missing.
        """
        if cls._instance is None:
            backend = _create_backend_from_config()
            cls._instance = cls(backend)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    @classmethod
    def set_instance(cls, instance: WorkflowFileSystem) -> None:
        """Set a custom instance (useful for testing)."""
        cls._instance = instance

    @property
    def backend(self) -> FileStorageBackend:
        """Get the underlying storage backend."""
        return self._backend

    async def store_file(
        self,
        user_id: str,
        run_id: str,
        filename: str,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> WorkflowFileRef:
        """
        Store a file and return a reference.

        Args:
            user_id: User ID for scoping files (data isolation).
            run_id: Workflow run ID for organizing files.
            filename: Original filename.
            data: Raw file bytes.
            mime_type: MIME type (default: application/octet-stream).
            metadata: Optional key-value metadata.

        Returns:
            WorkflowFileRef that can be stored in workflow state.
        """
        logger.debug("Storing file: user=%s run=%s name=%s size=%d", user_id, run_id, filename, len(data))
        return await self._backend.store(user_id, run_id, filename, data, mime_type=mime_type, metadata=metadata)

    async def store_from_base64(
        self,
        user_id: str,
        run_id: str,
        filename: str,
        base64_data: str,
        *,
        mime_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> WorkflowFileRef:
        """
        Store a file from base64-encoded data.

        Convenience method for tools that receive base64 data.

        Args:
            user_id: User ID for scoping files.
            run_id: Workflow run ID.
            filename: Original filename.
            base64_data: Base64-encoded file content.
            mime_type: MIME type.
            metadata: Optional metadata.

        Returns:
            WorkflowFileRef.
        """
        data = base64.b64decode(base64_data)
        return await self.store_file(user_id, run_id, filename, data, mime_type=mime_type, metadata=metadata)

    # pylint: disable=too-many-arguments  # All parameters are necessary for file storage with DB record tracking
    async def store_file_with_record(
        self,
        user: "User",
        run_id: str,
        filename: str,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
        source_node_id: Optional[str] = None,
        source_tool: Optional[str] = None,
        workflow_run: Optional["WorkflowRun"] = None,
        organization_id: int | None = None,
    ) -> WorkflowFileRef:
        """
        Store a file and create a database record for tracking.

        This is the preferred method for tools that want files to appear
        in the user's file management console (/api/v1/files).

        Args:
            user: User who owns the file.
            run_id: Workflow run ID (used for S3 path organization).
            filename: Original filename.
            data: Raw file bytes.
            mime_type: MIME type (default: application/octet-stream).
            metadata: Optional S3 metadata.
            source_node_id: Node that created the file (optional).
            source_tool: Tool that created the file (e.g., "google_drive_download_file").
            workflow_run: WorkflowRun model instance (optional, for FK relationship).

        Returns:
            WorkflowFileRef that can be stored in workflow state.
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports with database models
        from seer.database import WorkflowFile

        # 1. Store in S3 (existing behavior)
        file_ref = await self.store_file(
            user_id=user.user_id,
            run_id=run_id,
            filename=filename,
            data=data,
            mime_type=mime_type,
            metadata=metadata,
        )

        # 2. Create database record for file management API
        try:
            await WorkflowFile.create(
                file_id=file_ref.file_id,
                user=user,
                workflow_run=workflow_run,
                storage_path=file_ref.storage_path,
                filename=file_ref.filename,
                mime_type=file_ref.mime_type,
                size_bytes=file_ref.size_bytes,
                md5_hash=file_ref.md5_hash,
                source_node_id=source_node_id,
                source_tool=source_tool,
                organization_id=organization_id,
            )
            logger.debug(
                "Created WorkflowFile record: file_id=%s user=%s tool=%s",
                file_ref.file_id, user.user_id, source_tool
            )
        except Exception as e:  # pylint: disable=broad-exception-caught  # Catch all: S3 succeeded, DB record is optional
            # Log but don't fail - S3 storage succeeded, DB record is for management only
            logger.warning("Failed to create WorkflowFile record: %s", e)

        return file_ref

    async def get_file_content(self, file_ref: WorkflowFileRef) -> bytes:
        """
        Retrieve file content from a reference.

        Args:
            file_ref: File reference from a previous store operation.

        Returns:
            Raw file bytes.
        """
        logger.debug("Retrieving file: id=%s", file_ref.file_id)
        return await self._backend.retrieve(file_ref)

    async def get_file_as_base64(self, file_ref: WorkflowFileRef) -> str:
        """
        Retrieve file content as a base64 string.

        Convenience method for tools that need base64 output.

        Args:
            file_ref: File reference.

        Returns:
            Base64-encoded file content.
        """
        data = await self.get_file_content(file_ref)
        return base64.b64encode(data).decode("utf-8")

    async def delete_file(self, file_ref: WorkflowFileRef) -> bool:
        """
        Delete a file.

        Args:
            file_ref: File reference to delete.

        Returns:
            True if deleted, False if not found.
        """
        logger.debug("Deleting file: id=%s", file_ref.file_id)
        return await self._backend.delete(file_ref)

    async def delete_run_files(self, user_id: str, run_id: str) -> int:
        """
        Delete all files for a workflow run.

        Args:
            user_id: User ID for scoping.
            run_id: Workflow run ID.

        Returns:
            Number of files deleted.
        """
        logger.info("Deleting all files for user %s, run %s", user_id, run_id)
        return await self._backend.delete_by_run(user_id, run_id)

    async def delete_user_files(self, user_id: str) -> int:
        """
        Delete all files for a user.

        Args:
            user_id: User ID.

        Returns:
            Number of files deleted.
        """
        logger.info("Deleting all files for user: %s", user_id)
        return await self._backend.delete_by_user(user_id)

    async def get_presigned_url(
        self,
        file_ref: WorkflowFileRef,
        expires_seconds: int = 3600,
        inline: bool = False,
    ) -> str:
        """
        Get a presigned URL for direct download or inline preview.

        Args:
            file_ref: File reference.
            expires_seconds: URL expiration time.
            inline: If True, sets Content-Disposition to inline for browser preview.

        Returns:
            Presigned URL string.
        """
        return await self._backend.get_presigned_url(file_ref, expires_seconds, inline=inline)

    async def file_exists(self, file_ref: WorkflowFileRef) -> bool:
        """
        Check if a file exists.

        Args:
            file_ref: File reference to check.

        Returns:
            True if exists, False otherwise.
        """
        return await self._backend.exists(file_ref)

    async def get_file_by_id(self, file_id: str, user: "User") -> tuple[bytes, WorkflowFileRef]:
        """
        Retrieve file content and metadata by file_id from user's storage.

        This is used for resolving static_file_ref inputs, where the workflow
        references a file that was previously uploaded to the user's storage.

        Args:
            file_id: Unique file identifier.
            user: User who owns the file (for access control).

        Returns:
            Tuple of (file_bytes, WorkflowFileRef).

        Raises:
            FileNotFoundError: If the file doesn't exist or doesn't belong to the user.
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports with database models
        from seer.database import WorkflowFile

        file_record = await WorkflowFile.filter(file_id=file_id, user=user).first()
        if not file_record:
            raise FileNotFoundError(
                f"File '{file_id}' not found in user's storage. "
                "Ensure the file exists and belongs to the current user."
            )

        file_ref = file_to_ref(file_record)
        content = await self.get_file_content(file_ref)
        logger.debug("Retrieved file by ID: file_id=%s user=%s size=%d", file_id, user.user_id, len(content))

        return content, file_ref

    async def get_file_metadata_by_id(self, file_id: str, user: "User") -> WorkflowFileRef:
        """
        Get file metadata by file_id without downloading content.

        This is used when only metadata is needed (e.g., for validation or display).

        Args:
            file_id: Unique file identifier.
            user: User who owns the file (for access control).

        Returns:
            WorkflowFileRef with file metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist or doesn't belong to the user.
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports with database models
        from seer.database import WorkflowFile

        file_record = await WorkflowFile.filter(file_id=file_id, user=user).first()
        if not file_record:
            raise FileNotFoundError(
                f"File '{file_id}' not found in user's storage. "
                "Ensure the file exists and belongs to the current user."
            )

        return file_to_ref(file_record)

    def is_file_ref(self, value: Any) -> bool:
        """
        Check if a value is a file reference.

        Args:
            value: Any value to check.

        Returns:
            True if it's a file reference dict.
        """
        return is_file_ref(value)

    def parse_file_ref(self, value: dict[str, Any]) -> WorkflowFileRef:
        """
        Parse a dictionary into a WorkflowFileRef.

        Args:
            value: Dictionary containing file reference fields.

        Returns:
            WorkflowFileRef instance.
        """
        return parse_file_ref(value)

    async def resolve_to_bytes(self, value: Any) -> bytes:
        """
        Resolve a value to bytes, handling both file refs and base64.

        This is useful for tools that need to accept both file references
        and legacy base64 content.

        Args:
            value: Either a file reference dict or a base64 string.

        Returns:
            Raw file bytes.

        Raises:
            ValueError: If the value is neither a file ref nor valid base64.
        """
        if self.is_file_ref(value):
            file_ref = self.parse_file_ref(value)
            return await self.get_file_content(file_ref)
        if isinstance(value, str):
            # Assume it's base64
            try:
                return base64.b64decode(value)
            except Exception as e:
                raise ValueError(f"Invalid base64 data: {e}") from e
        else:
            raise ValueError(f"Expected file reference or base64 string, got {type(value)}")


def _create_backend_from_config() -> FileStorageBackend:
    """
    Create storage backend based on configuration.

    Uses standard AWS environment variables via boto3's default credential chain:
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: Credentials (or IAM roles)
    - AWS_REGION / AWS_DEFAULT_REGION: Region

    Config values (via env vars, .env, or Parameter Store):
    - WORKFLOW_FILE_S3_BUCKET: Bucket name (required)
    - WORKFLOW_FILE_S3_ENDPOINT_URL: Custom endpoint for R2/MinIO (optional)

    Returns:
        Configured FileStorageBackend instance.

    Raises:
        ValueError: If required configuration is missing.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports with config/backends
    from seer.config import config
    from seer.core.files.backends.s3 import S3FileStorage

    if not config.workflow_file_s3_bucket:
        raise ValueError(
            "Workflow file storage not configured. "
            "Set WORKFLOW_FILE_S3_BUCKET environment variable."
        )

    return S3FileStorage(
        config.workflow_file_s3_bucket,
        endpoint_url=config.workflow_file_s3_endpoint_url,
    )


def file_to_ref(file: "WorkflowFile", run_id_override: Optional[str] = None) -> WorkflowFileRef:
    """
    Convert a WorkflowFile database model to a WorkflowFileRef.

    This is a convenience function to avoid duplicate code when creating
    file references from database records.

    Args:
        file: WorkflowFile database model instance.
        run_id_override: Optional run ID to use instead of the file's workflow_run_id.
                        Useful when the run ID format differs (e.g., "run_123" vs int).

    Returns:
        WorkflowFileRef instance for use with the file system.
    """
    workflow_run_id = run_id_override if run_id_override is not None else (
        str(file.workflow_run_id) if file.workflow_run_id else ""
    )
    return WorkflowFileRef(
        file_id=file.file_id,
        storage_path=file.storage_path,
        filename=file.filename,
        mime_type=file.mime_type,
        size_bytes=file.size_bytes,
        workflow_run_id=workflow_run_id,
        created_at=file.created_at,
        md5_hash=file.md5_hash,
    )


# Re-export for convenience
__all__ = ["WorkflowFileSystem", "is_file_ref", "parse_file_ref", "WORKFLOW_FILE_REF_TYPE", "file_to_ref"]
