"""
Data models for the workflow file system.

The WorkflowFileRef is a lightweight reference that gets stored in workflow state
instead of raw file data. This keeps state small and efficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# Magic type marker for detecting file references in workflow state
WORKFLOW_FILE_REF_TYPE = "workflow_file_ref"


@dataclass(frozen=True)
class WorkflowFileRef:  # pylint: disable=too-many-instance-attributes  # File metadata requires these fields
    """
    Reference to a file stored in the workflow file system.

    This is what gets stored in workflow state instead of raw file data.
    Small enough to serialize efficiently (~200 bytes), contains metadata
    that tools need to work with the file.

    Attributes:
        file_id: Unique file identifier (UUID)
        storage_path: Full storage location (e.g., "s3://bucket/run_id/file_id/name.pdf")
        filename: Original filename
        mime_type: MIME type of the file
        size_bytes: File size in bytes
        workflow_run_id: Associated workflow run ID for lifecycle management
        created_at: When the file was stored
        md5_hash: Optional MD5 hash for integrity verification
    """

    file_id: str
    storage_path: str
    filename: str
    mime_type: str
    size_bytes: int
    workflow_run_id: str
    created_at: datetime
    md5_hash: Optional[str] = None

    # Internal type marker - always set, used for detection
    _type: str = field(default=WORKFLOW_FILE_REF_TYPE, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary for JSON storage in workflow state.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "_type": self._type,
            "file_id": self.file_id,
            "storage_path": self.storage_path,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "workflow_run_id": self.workflow_run_id,
            "created_at": self.created_at.isoformat(),
            "md5_hash": self.md5_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowFileRef:
        """
        Deserialize from dictionary (e.g., from workflow state).

        Args:
            data: Dictionary containing file reference fields.

        Returns:
            WorkflowFileRef instance.

        Raises:
            ValueError: If the dictionary is not a valid file reference.
        """
        if data.get("_type") != WORKFLOW_FILE_REF_TYPE:
            raise ValueError(f"Invalid file reference: expected _type='{WORKFLOW_FILE_REF_TYPE}'")

        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            file_id=data["file_id"],
            storage_path=data["storage_path"],
            filename=data["filename"],
            mime_type=data["mime_type"],
            size_bytes=data["size_bytes"],
            workflow_run_id=data["workflow_run_id"],
            created_at=created_at,
            md5_hash=data.get("md5_hash"),
        )

    @property
    def extension(self) -> str:
        """Get the file extension (e.g., '.pdf')."""
        if "." in self.filename:
            return "." + self.filename.rsplit(".", 1)[-1].lower()
        return ""

    @property
    def size_human(self) -> str:
        """Get human-readable file size (e.g., '1.5 MB')."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


def is_file_ref(value: Any) -> bool:
    """
    Check if a value is a workflow file reference.

    This function detects file references in workflow state by checking
    for the magic _type marker.

    Args:
        value: Any value to check.

    Returns:
        True if the value is a file reference dictionary.
    """
    return isinstance(value, dict) and value.get("_type") == WORKFLOW_FILE_REF_TYPE


def parse_file_ref(value: dict[str, Any]) -> WorkflowFileRef:
    """
    Parse a dictionary into a WorkflowFileRef.

    Convenience function that wraps WorkflowFileRef.from_dict().

    Args:
        value: Dictionary containing file reference fields.

    Returns:
        WorkflowFileRef instance.
    """
    return WorkflowFileRef.from_dict(value)
