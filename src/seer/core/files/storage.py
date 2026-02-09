"""
Abstract interface for file storage backends.

This module defines the FileStorageBackend ABC that all storage implementations
must follow. This allows swapping between S3, R2, or other S3-compatible storage
without changing the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Optional

if TYPE_CHECKING:
    from seer.core.files.models import WorkflowFileRef


class StorageError(Exception):
    """Raised when a storage operation fails."""


class FileNotFoundError(StorageError):  # pylint: disable=redefined-builtin  # Domain-specific exception distinct from built-in
    """Raised when a file is not found in storage."""


class FileStorageBackend(ABC):
    """
    Abstract base class for file storage backends.

    Implementations must provide methods for storing, retrieving, and deleting
    files. The interface is designed to work with S3-compatible storage systems.

    All methods are async to support non-blocking I/O.
    """

    @abstractmethod
    async def store(
        self,
        user_id: str,
        run_id: str,
        filename: str,
        data: bytes,
        *,
        mime_type: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> "WorkflowFileRef":
        """
        Store file data and return a reference.

        Args:
            user_id: User ID for scoping files (data isolation).
            run_id: Workflow run ID for organizing files.
            filename: Original filename (used in storage path).
            data: Raw file bytes to store.
            mime_type: MIME type of the file.
            metadata: Optional key-value metadata to attach to the file.

        Returns:
            WorkflowFileRef containing the storage location and metadata.

        Raises:
            StorageError: If the upload fails.
        """

    @abstractmethod
    async def retrieve(self, file_ref: "WorkflowFileRef") -> bytes:
        """
        Retrieve file data from a reference.

        Args:
            file_ref: File reference containing storage location.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            StorageError: If the download fails.
        """

    @abstractmethod
    async def retrieve_stream(
        self, file_ref: "WorkflowFileRef", chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """
        Stream file data for large files.

        Args:
            file_ref: File reference containing storage location.
            chunk_size: Size of each chunk in bytes.

        Yields:
            Chunks of file data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            StorageError: If the download fails.
        """

    @abstractmethod
    async def delete(self, file_ref: "WorkflowFileRef") -> bool:
        """
        Delete a single file.

        Args:
            file_ref: File reference to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            StorageError: If the deletion fails (other than not found).
        """

    @abstractmethod
    async def delete_by_run(self, user_id: str, run_id: str) -> int:
        """
        Delete all files for a workflow run.

        Args:
            user_id: User ID for scoping.
            run_id: Workflow run ID.

        Returns:
            Number of files deleted.

        Raises:
            StorageError: If the deletion fails.
        """

    @abstractmethod
    async def delete_by_user(self, user_id: str) -> int:
        """
        Delete all files for a user.

        Args:
            user_id: User ID.

        Returns:
            Number of files deleted.

        Raises:
            StorageError: If the deletion fails.
        """

    @abstractmethod
    async def get_presigned_url(
        self,
        file_ref: "WorkflowFileRef",
        expires_seconds: int = 3600,
    ) -> str:
        """
        Get a presigned URL for direct file download.

        Args:
            file_ref: File reference.
            expires_seconds: URL expiration time in seconds.

        Returns:
            Presigned URL string.

        Raises:
            StorageError: If URL generation fails.
        """

    @abstractmethod
    async def exists(self, file_ref: "WorkflowFileRef") -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_ref: File reference to check.

        Returns:
            True if the file exists, False otherwise.
        """
