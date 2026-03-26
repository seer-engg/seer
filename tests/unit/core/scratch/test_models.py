"""
Unit tests for scratch storage models.

Tests ScratchDataRef serialization and scratch reference detection.
"""
from datetime import datetime, timezone

import pytest

from seer.core.scratch.models import (
    SCRATCH_DATA_REF_TYPE,
    ScratchDataRef,
    is_scratch_ref,
    parse_scratch_ref,
)


# =============================================================================
# ScratchDataRef Tests
# =============================================================================


@pytest.mark.unit
class TestScratchDataRef:
    """Tests for ScratchDataRef dataclass."""

    def test_create_scratch_ref(self):
        """Test creating a ScratchDataRef."""
        now = datetime.now(timezone.utc)
        ref = ScratchDataRef(
            handle="oura_sleep_abc123",
            data_type="json",
            preview='[{"date": "2026-03-25", "score": 85}]',
            row_count=30,
            size_bytes=50000,
            storage_path="user_123/run_456/scratch/oura_sleep_abc123.json",
            workflow_run_id="run_456",
            created_at=now,
        )

        assert ref.handle == "oura_sleep_abc123"
        assert ref.data_type == "json"
        assert ref.row_count == 30
        assert ref.size_bytes == 50000
        assert ref._type == SCRATCH_DATA_REF_TYPE

    def test_scratch_ref_immutable(self):
        """Test that ScratchDataRef is immutable (frozen)."""
        ref = ScratchDataRef(
            handle="test_handle",
            data_type="json",
            preview="preview",
            size_bytes=1024,
            storage_path="path/to/data.json",
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            ref.handle = "changed"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        ref = ScratchDataRef(
            handle="oura_sleep_abc123",
            data_type="json",
            preview='[{"date": "2026-03-25", "score": 85}]',
            row_count=30,
            size_bytes=50000,
            storage_path="user_123/run_456/scratch/oura_sleep_abc123.json",
            workflow_run_id="run_456",
            created_at=now,
        )

        data = ref.to_dict()

        assert data["_type"] == SCRATCH_DATA_REF_TYPE
        assert data["handle"] == "oura_sleep_abc123"
        assert data["data_type"] == "json"
        assert data["preview"] == '[{"date": "2026-03-25", "score": 85}]'
        assert data["row_count"] == 30
        assert data["size_bytes"] == 50000
        assert data["storage_path"] == "user_123/run_456/scratch/oura_sleep_abc123.json"
        assert data["workflow_run_id"] == "run_456"
        assert data["created_at"] == "2026-03-25T12:00:00+00:00"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "_type": SCRATCH_DATA_REF_TYPE,
            "handle": "oura_sleep_abc123",
            "data_type": "json",
            "preview": '[{"date": "2026-03-25", "score": 85}]',
            "row_count": 30,
            "size_bytes": 50000,
            "storage_path": "user_123/run_456/scratch/oura_sleep_abc123.json",
            "workflow_run_id": "run_456",
            "created_at": "2026-03-25T12:00:00+00:00",
        }

        ref = ScratchDataRef.from_dict(data)

        assert ref.handle == "oura_sleep_abc123"
        assert ref.data_type == "json"
        assert ref.row_count == 30

    def test_from_dict_invalid_type(self):
        """Test from_dict raises on invalid _type."""
        data = {
            "_type": "invalid_type",
            "handle": "abc-123",
        }

        with pytest.raises(ValueError, match="Invalid scratch reference"):
            ScratchDataRef.from_dict(data)

    def test_from_dict_without_row_count(self):
        """Test from_dict handles optional row_count."""
        data = {
            "_type": SCRATCH_DATA_REF_TYPE,
            "handle": "test_handle",
            "data_type": "json",
            "preview": "{}",
            "size_bytes": 100,
            "storage_path": "path/to/data.json",
            "workflow_run_id": "run_456",
            "created_at": "2026-03-25T12:00:00+00:00",
        }

        ref = ScratchDataRef.from_dict(data)
        assert ref.row_count is None

    def test_round_trip_serialization(self):
        """Test serialization round-trip preserves data."""
        now = datetime.now(timezone.utc)
        original = ScratchDataRef(
            handle="oura_sleep_abc123",
            data_type="json",
            preview='[{"date": "2026-03-25", "score": 85}]',
            row_count=30,
            size_bytes=50000,
            storage_path="user_123/run_456/scratch/oura_sleep_abc123.json",
            workflow_run_id="run_456",
            created_at=now,
        )

        data = original.to_dict()
        restored = ScratchDataRef.from_dict(data)

        assert restored.handle == original.handle
        assert restored.data_type == original.data_type
        assert restored.preview == original.preview
        assert restored.row_count == original.row_count
        assert restored.size_bytes == original.size_bytes
        assert restored.storage_path == original.storage_path
        assert restored.workflow_run_id == original.workflow_run_id

    def test_size_human_bytes(self):
        """Test human-readable size for bytes."""
        ref = ScratchDataRef(
            handle="small_data",
            data_type="json",
            preview="{}",
            size_bytes=512,
            storage_path="path/to/data.json",
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.size_human == "512 B"

    def test_size_human_kilobytes(self):
        """Test human-readable size for kilobytes."""
        ref = ScratchDataRef(
            handle="medium_data",
            data_type="json",
            preview="{}",
            size_bytes=50000,
            storage_path="path/to/data.json",
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        assert ref.size_human == "48.8 KB"


# =============================================================================
# is_scratch_ref Tests
# =============================================================================


@pytest.mark.unit
class TestIsScratchRef:
    """Tests for is_scratch_ref detection function."""

    def test_is_scratch_ref_valid(self):
        """Test is_scratch_ref returns True for valid scratch ref dict."""
        data = {
            "_type": SCRATCH_DATA_REF_TYPE,
            "handle": "abc-123",
        }

        assert is_scratch_ref(data) is True

    def test_is_scratch_ref_wrong_type(self):
        """Test is_scratch_ref returns False for wrong _type."""
        data = {
            "_type": "workflow_file_ref",
            "handle": "abc-123",
        }

        assert is_scratch_ref(data) is False

    def test_is_scratch_ref_no_type(self):
        """Test is_scratch_ref returns False when _type missing."""
        data = {
            "handle": "abc-123",
        }

        assert is_scratch_ref(data) is False

    def test_is_scratch_ref_not_dict(self):
        """Test is_scratch_ref returns False for non-dict values."""
        assert is_scratch_ref("string") is False
        assert is_scratch_ref(123) is False
        assert is_scratch_ref(None) is False
        assert is_scratch_ref([]) is False


# =============================================================================
# parse_scratch_ref Tests
# =============================================================================


@pytest.mark.unit
class TestParseScratchRef:
    """Tests for parse_scratch_ref function."""

    def test_parse_scratch_ref_valid(self):
        """Test parse_scratch_ref returns ScratchDataRef."""
        data = {
            "_type": SCRATCH_DATA_REF_TYPE,
            "handle": "oura_sleep_abc123",
            "data_type": "json",
            "preview": '[{"date": "2026-03-25"}]',
            "row_count": 30,
            "size_bytes": 50000,
            "storage_path": "user_123/run_456/scratch/oura_sleep_abc123.json",
            "workflow_run_id": "run_456",
            "created_at": "2026-03-25T12:00:00+00:00",
        }

        ref = parse_scratch_ref(data)

        assert isinstance(ref, ScratchDataRef)
        assert ref.handle == "oura_sleep_abc123"

    def test_parse_scratch_ref_invalid(self):
        """Test parse_scratch_ref raises for invalid data."""
        data = {
            "_type": "invalid",
        }

        with pytest.raises(ValueError):
            parse_scratch_ref(data)
