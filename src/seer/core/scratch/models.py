"""
Data models for the scratch storage system.

ScratchDataRef is a lightweight reference that gets returned to the agent
instead of large raw data. The agent sees only the preview and metadata,
while the full data remains in S3 for direct sandbox access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Magic type marker for detecting scratch references in tool outputs
SCRATCH_DATA_REF_TYPE = "scratch_data_ref"


@dataclass(frozen=True)
class ScratchDataRef:  # pylint: disable=too-many-instance-attributes  # Pure data transfer object; all fields are required reference metadata
    """
    Reference to data stored in the scratch storage system.

    This is what tools return instead of large datasets. The agent sees
    a small preview (~5 rows) and metadata, while the full data is stored
    in S3 for efficient sandbox loading.

    Attributes:
        handle: Unique identifier for retrieving the data (e.g., "oura_sleep_abc123")
        data_type: Format of the stored data ("json" or "csv")
        preview: Human-readable preview (first N rows) for LLM context
        row_count: Number of records in the data (None if not applicable)
        size_bytes: Size of the stored data in bytes
        storage_path: Full S3 path where data is stored
        workflow_run_id: Associated workflow run ID for lifecycle management
        created_at: When the data was stored
    """

    handle: str
    data_type: str
    preview: str
    size_bytes: int
    storage_path: str
    workflow_run_id: str
    created_at: datetime
    row_count: int | None = None

    # Internal type marker - always set, used for detection
    _type: str = field(default=SCRATCH_DATA_REF_TYPE, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary for JSON storage in tool output.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "_type": self._type,
            "handle": self.handle,
            "data_type": self.data_type,
            "preview": self.preview,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "storage_path": self.storage_path,
            "workflow_run_id": self.workflow_run_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScratchDataRef:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary containing scratch reference fields.

        Returns:
            ScratchDataRef instance.

        Raises:
            ValueError: If the dictionary is not a valid scratch reference.
        """
        if data.get("_type") != SCRATCH_DATA_REF_TYPE:
            raise ValueError(f"Invalid scratch reference: expected _type='{SCRATCH_DATA_REF_TYPE}'")

        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            handle=data["handle"],
            data_type=data["data_type"],
            preview=data["preview"],
            row_count=data.get("row_count"),
            size_bytes=data["size_bytes"],
            storage_path=data["storage_path"],
            workflow_run_id=data["workflow_run_id"],
            created_at=created_at,
        )

    @property
    def size_human(self) -> str:
        """Get human-readable data size (e.g., '1.5 MB')."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


def is_scratch_ref(value: Any) -> bool:
    """
    Check if a value is a scratch data reference.

    This function detects scratch references in tool outputs by checking
    for the magic _type marker.

    Args:
        value: Any value to check.

    Returns:
        True if the value is a scratch reference dictionary.
    """
    return isinstance(value, dict) and value.get("_type") == SCRATCH_DATA_REF_TYPE


def parse_scratch_ref(value: dict[str, Any]) -> ScratchDataRef:
    """
    Parse a dictionary into a ScratchDataRef.

    Args:
        value: Dictionary containing scratch reference fields.

    Returns:
        ScratchDataRef instance.
    """
    return ScratchDataRef.from_dict(value)
