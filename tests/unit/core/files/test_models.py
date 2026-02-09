"""
Unit tests for workflow file system models.

Tests WorkflowFileRef serialization and file reference detection.
"""

from datetime import datetime, timezone

import pytest

from seer.core.files.models import (
    WORKFLOW_FILE_REF_TYPE,
    WorkflowFileRef,
    is_file_ref,
    parse_file_ref,
)


# =============================================================================
# WorkflowFileRef Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowFileRef:
    """Tests for WorkflowFileRef dataclass."""

    def test_create_file_ref(self):
        """Test creating a WorkflowFileRef."""
        now = datetime.now(timezone.utc)
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/path/to/file.pdf",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=now,
        )

        assert ref.file_id == "abc-123"
        assert ref.filename == "document.pdf"
        assert ref.mime_type == "application/pdf"
        assert ref.size_bytes == 1024
        assert ref._type == WORKFLOW_FILE_REF_TYPE

    def test_file_ref_immutable(self):
        """Test that WorkflowFileRef is immutable (frozen)."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/path/to/file.pdf",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            ref.file_id = "changed"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/path/to/file.pdf",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=now,
            md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        )

        data = ref.to_dict()

        assert data["_type"] == WORKFLOW_FILE_REF_TYPE
        assert data["file_id"] == "abc-123"
        assert data["storage_path"] == "s3://bucket/path/to/file.pdf"
        assert data["filename"] == "document.pdf"
        assert data["mime_type"] == "application/pdf"
        assert data["size_bytes"] == 1024
        assert data["workflow_run_id"] == "run_456"
        assert data["created_at"] == "2026-02-09T12:00:00+00:00"
        assert data["md5_hash"] == "d41d8cd98f00b204e9800998ecf8427e"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "_type": WORKFLOW_FILE_REF_TYPE,
            "file_id": "abc-123",
            "storage_path": "s3://bucket/path/to/file.pdf",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "workflow_run_id": "run_456",
            "created_at": "2026-02-09T12:00:00+00:00",
            "md5_hash": "d41d8cd98f00b204e9800998ecf8427e",
        }

        ref = WorkflowFileRef.from_dict(data)

        assert ref.file_id == "abc-123"
        assert ref.filename == "document.pdf"
        assert ref.md5_hash == "d41d8cd98f00b204e9800998ecf8427e"

    def test_from_dict_invalid_type(self):
        """Test from_dict raises on invalid _type."""
        data = {
            "_type": "invalid_type",
            "file_id": "abc-123",
        }

        with pytest.raises(ValueError, match="Invalid file reference"):
            WorkflowFileRef.from_dict(data)

    def test_round_trip_serialization(self):
        """Test serialization round-trip preserves data."""
        now = datetime.now(timezone.utc)
        original = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/path/to/file.pdf",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=now,
        )

        data = original.to_dict()
        restored = WorkflowFileRef.from_dict(data)

        assert restored.file_id == original.file_id
        assert restored.storage_path == original.storage_path
        assert restored.filename == original.filename
        assert restored.mime_type == original.mime_type
        assert restored.size_bytes == original.size_bytes
        assert restored.workflow_run_id == original.workflow_run_id

    def test_extension_property(self):
        """Test extension property extraction."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/file.pdf",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.extension == ".pdf"

    def test_extension_no_extension(self):
        """Test extension property when no extension."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/file",
            filename="README",
            mime_type="text/plain",
            size_bytes=1024,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.extension == ""

    def test_size_human_bytes(self):
        """Test human-readable size for bytes."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/file.txt",
            filename="small.txt",
            mime_type="text/plain",
            size_bytes=512,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.size_human == "512 B"

    def test_size_human_kilobytes(self):
        """Test human-readable size for kilobytes."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/file.txt",
            filename="medium.txt",
            mime_type="text/plain",
            size_bytes=2048,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.size_human == "2.0 KB"

    def test_size_human_megabytes(self):
        """Test human-readable size for megabytes."""
        ref = WorkflowFileRef(
            file_id="abc-123",
            storage_path="s3://bucket/file.pdf",
            filename="large.pdf",
            mime_type="application/pdf",
            size_bytes=5 * 1024 * 1024,
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.size_human == "5.0 MB"


# =============================================================================
# is_file_ref Tests
# =============================================================================


@pytest.mark.unit
class TestIsFileRef:
    """Tests for is_file_ref detection function."""

    def test_is_file_ref_valid(self):
        """Test is_file_ref returns True for valid file ref dict."""
        data = {
            "_type": WORKFLOW_FILE_REF_TYPE,
            "file_id": "abc-123",
        }

        assert is_file_ref(data) is True

    def test_is_file_ref_wrong_type(self):
        """Test is_file_ref returns False for wrong _type."""
        data = {
            "_type": "something_else",
            "file_id": "abc-123",
        }

        assert is_file_ref(data) is False

    def test_is_file_ref_no_type(self):
        """Test is_file_ref returns False when _type missing."""
        data = {
            "file_id": "abc-123",
        }

        assert is_file_ref(data) is False

    def test_is_file_ref_not_dict(self):
        """Test is_file_ref returns False for non-dict values."""
        assert is_file_ref("string") is False
        assert is_file_ref(123) is False
        assert is_file_ref(None) is False
        assert is_file_ref([]) is False


# =============================================================================
# parse_file_ref Tests
# =============================================================================


@pytest.mark.unit
class TestParseFileRef:
    """Tests for parse_file_ref function."""

    def test_parse_file_ref_valid(self):
        """Test parse_file_ref returns WorkflowFileRef."""
        data = {
            "_type": WORKFLOW_FILE_REF_TYPE,
            "file_id": "abc-123",
            "storage_path": "s3://bucket/path/to/file.pdf",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "workflow_run_id": "run_456",
            "created_at": "2026-02-09T12:00:00+00:00",
        }

        ref = parse_file_ref(data)

        assert isinstance(ref, WorkflowFileRef)
        assert ref.file_id == "abc-123"

    def test_parse_file_ref_invalid(self):
        """Test parse_file_ref raises for invalid data."""
        data = {
            "_type": "invalid",
        }

        with pytest.raises(ValueError):
            parse_file_ref(data)
