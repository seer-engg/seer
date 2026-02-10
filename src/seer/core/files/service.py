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
    pass

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
    ) -> str:
        """
        Get a presigned URL for direct download.

        Args:
            file_ref: File reference.
            expires_seconds: URL expiration time.

        Returns:
            Presigned URL string.
        """
        return await self._backend.get_presigned_url(file_ref, expires_seconds)

    async def file_exists(self, file_ref: WorkflowFileRef) -> bool:
        """
        Check if a file exists.

        Args:
            file_ref: File reference to check.

        Returns:
            True if exists, False otherwise.
        """
        return await self._backend.exists(file_ref)

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

    Returns:
        Configured FileStorageBackend instance.

    Raises:
        ValueError: If required configuration is missing.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports with config/backends
    from seer.config import config
    from seer.core.files.backends.s3 import S3FileStorage

    # Validate required configuration
    if not config.workflow_file_s3_bucket:
        raise ValueError(
            "Workflow file storage not configured. "
            "Set WORKFLOW_FILE_S3_BUCKET environment variable."
        )

    return S3FileStorage(
        config.workflow_file_s3_bucket,
        region=config.workflow_file_s3_region,
        endpoint_url=config.workflow_file_s3_endpoint_url,
        access_key=config.workflow_file_s3_access_key,
        secret_key=config.workflow_file_s3_secret_key,
        presigned_url_expiry=config.workflow_file_presigned_url_expiry_seconds,
    )


# Re-export for convenience
__all__ = ["WorkflowFileSystem", "is_file_ref", "parse_file_ref", "WORKFLOW_FILE_REF_TYPE"]
