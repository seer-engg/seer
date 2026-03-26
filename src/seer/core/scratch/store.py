"""
Scratch data storage service.

Provides a transient key-value store for large tool outputs that shouldn't
flow through the LLM context. Data is stored in S3 using the same backend
as WorkflowFileSystem, scoped by user and run for security and cleanup.

Storage path pattern: {user_id}/{run_id}/scratch/{handle}.{ext}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from seer.core.scratch.models import ScratchDataRef
from seer.logger import get_logger

if TYPE_CHECKING:
    from seer.core.files.storage import FileStorageBackend

logger = get_logger("seer.core.scratch.store")


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate a string to max_len characters, appending ellipsis if needed."""
    return s[:max_len] + "..." if len(s) > max_len else s


def _generate_preview(data: Any, data_type: str, preview_rows: int) -> str:
    """Generate a human-readable preview of the data for LLM context."""
    if data_type == "json":
        if isinstance(data, list):
            preview_str = json.dumps(data[:preview_rows], indent=2, default=str)
            if len(data) > preview_rows:
                preview_str += f"\n... and {len(data) - preview_rows} more records"
            return preview_str
        return _truncate(json.dumps(data, indent=2, default=str))
    if data_type == "csv" and isinstance(data, str):
        lines = data.split("\n")
        preview_str = "\n".join(lines[:preview_rows + 1])  # +1 for header
        if len(lines) > preview_rows + 1:
            preview_str += f"\n... and {len(lines) - preview_rows - 1} more rows"
        return preview_str
    return _truncate(str(data))


def _count_records(data: Any, data_type: str) -> int | None:
    """Count records in the data, if applicable."""
    if data_type == "json" and isinstance(data, list):
        return len(data)
    if data_type == "csv" and isinstance(data, str):
        lines = data.strip().split("\n")
        return len(lines) - 1 if len(lines) > 1 else 0  # Exclude header
    return None


class ScratchStore:
    """
    Service for storing and retrieving large tool outputs.

    Data is stored in S3 with the same backend as WorkflowFileSystem,
    but under a separate 'scratch' prefix. This keeps scratch data
    organized and easy to clean up.

    Usage:
        store = ScratchStore.instance()
        ref = await store.store(user_id, run_id, "oura_sleep", data)
        data = await store.retrieve(ref.handle, user_id, run_id)
    """

    _instance: Optional[ScratchStore] = None

    def __init__(self, backend: "FileStorageBackend"):
        """
        Initialize with a storage backend.

        Use ScratchStore.instance() to get the configured singleton.

        Args:
            backend: Storage backend implementation.
        """
        self._backend = backend
        # In-memory index for handle -> storage_path mapping within a session
        # This avoids needing to parse the handle to determine the path
        self._handle_index: dict[str, str] = {}

    @classmethod
    def instance(cls) -> ScratchStore:
        """
        Get the singleton instance with configured backend.

        Returns:
            Configured ScratchStore instance.

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
    def set_instance(cls, instance: ScratchStore) -> None:
        """Set a custom instance (useful for testing)."""
        cls._instance = instance

    async def store(
        self,
        user_id: str,
        run_id: str,
        handle: str,
        data: Any,
        *,
        data_type: str = "json",
        preview_rows: int = 5,
    ) -> ScratchDataRef:
        """
        Store data and return a lightweight reference.

        Args:
            user_id: User ID for data isolation.
            run_id: Workflow run ID for scoping and cleanup.
            handle: Unique handle for this data (e.g., "oura_sleep_abc123").
            data: The data to store (list, dict, or string).
            data_type: Format of the data ("json" or "csv").
            preview_rows: Number of rows to include in the preview.

        Returns:
            ScratchDataRef containing handle, preview, and metadata.
        """
        # Serialize data
        if data_type == "json":
            serialized = json.dumps(data, default=str).encode("utf-8")
        elif data_type == "csv":
            if isinstance(data, str):
                serialized = data.encode("utf-8")
            else:
                # Convert list of dicts to CSV
                serialized = _list_to_csv(data).encode("utf-8")
        else:
            serialized = str(data).encode("utf-8")

        # Generate preview for LLM context
        preview = _generate_preview(data, data_type, preview_rows)
        row_count = _count_records(data, data_type)

        # Determine extension
        ext = "json" if data_type == "json" else "csv" if data_type == "csv" else "txt"

        # Build storage path: {user_id}/{run_id}/scratch/{handle}.{ext}
        storage_path = f"{user_id}/{run_id}/scratch/{handle}.{ext}"

        # Store in S3
        await self._backend.store_raw(storage_path, serialized)

        # Index the handle for retrieval
        self._handle_index[handle] = storage_path

        logger.debug(
            "Stored scratch data: handle=%s path=%s size=%d rows=%s",
            handle, storage_path, len(serialized), row_count
        )

        return ScratchDataRef(
            handle=handle,
            data_type=data_type,
            preview=preview,
            row_count=row_count,
            size_bytes=len(serialized),
            storage_path=storage_path,
            workflow_run_id=run_id,
            created_at=datetime.now(timezone.utc),
        )

    async def retrieve(
        self,
        handle: str,
        user_id: str,
        run_id: str,
    ) -> bytes:
        """
        Retrieve stored data by handle.

        Args:
            handle: The data handle from a previous store call.
            user_id: User ID for access control.
            run_id: Workflow run ID.

        Returns:
            Raw data bytes.

        Raises:
            FileNotFoundError: If the data is not found.
        """
        # Check in-memory index first
        if handle in self._handle_index:
            storage_path = self._handle_index[handle]
        else:
            # Reconstruct path - try both json and csv extensions
            for ext in ["json", "csv", "txt"]:
                storage_path = f"{user_id}/{run_id}/scratch/{handle}.{ext}"
                if await self._backend.exists_raw(storage_path):
                    break
            else:
                raise FileNotFoundError(f"Scratch data not found: {handle}")

        data = await self._backend.retrieve_raw(storage_path)
        logger.debug("Retrieved scratch data: handle=%s size=%d", handle, len(data))
        return data

    async def retrieve_from_ref(self, ref: ScratchDataRef) -> bytes:
        """
        Retrieve stored data using a ScratchDataRef.

        Args:
            ref: Scratch data reference from a previous store call.

        Returns:
            Raw data bytes.

        Raises:
            FileNotFoundError: If the data is not found.
        """
        data = await self._backend.retrieve_raw(ref.storage_path)
        logger.debug("Retrieved scratch data: handle=%s size=%d", ref.handle, len(data))
        return data

    async def delete(self, handle: str, user_id: str, run_id: str) -> bool:
        """
        Delete stored data.

        Args:
            handle: The data handle.
            user_id: User ID.
            run_id: Workflow run ID.

        Returns:
            True if deleted, False if not found.
        """
        # Check index first
        if handle in self._handle_index:
            storage_path = self._handle_index.pop(handle)
        else:
            # Try to find the file
            for ext in ["json", "csv", "txt"]:
                storage_path = f"{user_id}/{run_id}/scratch/{handle}.{ext}"
                if await self._backend.exists_raw(storage_path):
                    break
            else:
                return False

        result = await self._backend.delete_raw(storage_path)
        logger.debug("Deleted scratch data: handle=%s", handle)
        return result

    async def delete_run_scratch(self, user_id: str, run_id: str) -> int:
        """
        Delete all scratch data for a workflow run.

        Args:
            user_id: User ID.
            run_id: Workflow run ID.

        Returns:
            Number of items deleted.
        """
        prefix = f"{user_id}/{run_id}/scratch/"
        count = await self._backend.delete_by_prefix(prefix)

        # Clear matching entries from index
        to_remove = [h for h, p in self._handle_index.items() if p.startswith(prefix)]
        for handle in to_remove:
            del self._handle_index[handle]

        logger.info("Deleted %d scratch items for run %s", count, run_id)
        return count


def _list_to_csv(data: List[Any]) -> str:
    """Convert a list of dicts to CSV string."""
    if not data:
        return ""
    if not isinstance(data[0], dict):
        # Simple list
        return "\n".join(str(item) for item in data)

    # List of dicts - use first item's keys as headers
    headers = list(data[0].keys())
    lines = [",".join(headers)]
    for item in data:
        values = [str(item.get(h, "")).replace(",", ";").replace("\n", " ") for h in headers]
        lines.append(",".join(values))
    return "\n".join(lines)


def _create_backend_from_config() -> "FileStorageBackend":
    """
    Create storage backend based on configuration.

    Uses the same S3 configuration as WorkflowFileSystem.

    Returns:
        Configured FileStorageBackend instance.

    Raises:
        ValueError: If required configuration is missing.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports
    from seer.config import config
    from seer.core.files.backends.s3 import S3FileStorage

    if not config.workflow_file_s3_bucket:
        raise ValueError(
            "Scratch storage not configured. "
            "Set WORKFLOW_FILE_S3_BUCKET environment variable."
        )

    return S3FileStorage(
        config.workflow_file_s3_bucket,
        endpoint_url=config.workflow_file_s3_endpoint_url,
    )


__all__ = ["ScratchStore"]
