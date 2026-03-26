"""
Unit tests for scratch storage service.

Tests ScratchStore store/retrieve operations with mock backend.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.scratch.models import ScratchDataRef
from seer.core.scratch.store import ScratchStore, _generate_preview, _count_records, _list_to_csv


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestGeneratePreview:
    """Tests for _generate_preview helper."""

    def test_json_list_preview(self):
        """Test preview of JSON list data."""
        data = [
            {"date": "2026-03-25", "score": 85},
            {"date": "2026-03-24", "score": 82},
            {"date": "2026-03-23", "score": 90},
        ]

        preview = _generate_preview(data, "json", preview_rows=2)

        assert '"date"' in preview
        assert '"score"' in preview
        assert "and 1 more records" in preview

    def test_json_list_preview_all_rows(self):
        """Test preview when all rows fit."""
        data = [
            {"date": "2026-03-25", "score": 85},
            {"date": "2026-03-24", "score": 82},
        ]

        preview = _generate_preview(data, "json", preview_rows=5)

        assert "more records" not in preview

    def test_json_object_preview(self):
        """Test preview of single JSON object."""
        data = {"key": "value" * 100}

        preview = _generate_preview(data, "json", preview_rows=5)

        assert len(preview) <= 503  # 500 + "..."

    def test_csv_preview(self):
        """Test preview of CSV data."""
        csv_data = "name,score\nAlice,85\nBob,82\nCharlie,90"

        preview = _generate_preview(csv_data, "csv", preview_rows=2)

        assert "name,score" in preview
        assert "Alice,85" in preview
        assert "and 1 more rows" in preview


@pytest.mark.unit
class TestCountRecords:
    """Tests for _count_records helper."""

    def test_count_json_list(self):
        """Test counting records in JSON list."""
        data = [{"a": 1}, {"a": 2}, {"a": 3}]

        count = _count_records(data, "json")

        assert count == 3

    def test_count_json_object(self):
        """Test counting records returns None for non-list."""
        data = {"key": "value"}

        count = _count_records(data, "json")

        assert count is None

    def test_count_csv_rows(self):
        """Test counting rows in CSV (excluding header)."""
        csv_data = "name,score\nAlice,85\nBob,82"

        count = _count_records(csv_data, "csv")

        assert count == 2


@pytest.mark.unit
class TestListToCsv:
    """Tests for _list_to_csv helper."""

    def test_list_of_dicts_to_csv(self):
        """Test converting list of dicts to CSV."""
        data = [
            {"name": "Alice", "score": 85},
            {"name": "Bob", "score": 82},
        ]

        csv = _list_to_csv(data)

        assert csv == "name,score\nAlice,85\nBob,82"

    def test_empty_list(self):
        """Test converting empty list."""
        csv = _list_to_csv([])
        assert csv == ""

    def test_simple_list(self):
        """Test converting list of primitives."""
        data = [1, 2, 3]

        csv = _list_to_csv(data)

        assert csv == "1\n2\n3"


# =============================================================================
# ScratchStore Tests
# =============================================================================


@pytest.fixture
def mock_backend():
    """Create a mock storage backend."""
    backend = MagicMock()
    backend.store_raw = AsyncMock()
    backend.retrieve_raw = AsyncMock(return_value=b'[{"score": 85}]')
    backend.exists_raw = AsyncMock(return_value=True)
    backend.delete_raw = AsyncMock(return_value=True)
    backend.delete_by_prefix = AsyncMock(return_value=5)
    return backend


@pytest.fixture
def scratch_store(mock_backend):
    """Create a ScratchStore with mock backend."""
    store = ScratchStore(mock_backend)
    # Reset singleton for clean tests
    ScratchStore._instance = None
    return store


@pytest.mark.unit
class TestScratchStore:
    """Tests for ScratchStore service."""

    @pytest.mark.asyncio
    async def test_store_json_data(self, scratch_store, mock_backend):
        """Test storing JSON list data."""
        data = [
            {"date": "2026-03-25", "score": 85},
            {"date": "2026-03-24", "score": 82},
        ]

        ref = await scratch_store.store(
            user_id="user_123",
            run_id="run_456",
            handle="oura_sleep_abc123",
            data=data,
            data_type="json",
            preview_rows=5,
        )

        # Verify result
        assert isinstance(ref, ScratchDataRef)
        assert ref.handle == "oura_sleep_abc123"
        assert ref.data_type == "json"
        assert ref.row_count == 2
        assert ref.workflow_run_id == "run_456"
        assert "oura_sleep_abc123.json" in ref.storage_path

        # Verify backend was called
        mock_backend.store_raw.assert_called_once()
        call_args = mock_backend.store_raw.call_args
        assert call_args[0][0] == "user_123/run_456/scratch/oura_sleep_abc123.json"

    @pytest.mark.asyncio
    async def test_store_csv_data(self, scratch_store, mock_backend):
        """Test storing CSV string data."""
        csv_data = "name,score\nAlice,85\nBob,82"

        ref = await scratch_store.store(
            user_id="user_123",
            run_id="run_456",
            handle="sheet_data",
            data=csv_data,
            data_type="csv",
        )

        assert ref.handle == "sheet_data"
        assert ref.data_type == "csv"
        assert "sheet_data.csv" in ref.storage_path

    @pytest.mark.asyncio
    async def test_retrieve_data(self, scratch_store, mock_backend):
        """Test retrieving stored data."""
        # First store something to populate the index
        await scratch_store.store(
            user_id="user_123",
            run_id="run_456",
            handle="test_handle",
            data={"key": "value"},
        )

        # Now retrieve
        data = await scratch_store.retrieve(
            handle="test_handle",
            user_id="user_123",
            run_id="run_456",
        )

        assert data == b'[{"score": 85}]'

    @pytest.mark.asyncio
    async def test_retrieve_from_ref(self, scratch_store, mock_backend):
        """Test retrieving data using a ScratchDataRef."""
        from datetime import datetime, timezone

        ref = ScratchDataRef(
            handle="test_handle",
            data_type="json",
            preview="{}",
            size_bytes=100,
            storage_path="user_123/run_456/scratch/test_handle.json",
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

        data = await scratch_store.retrieve_from_ref(ref)

        assert data == b'[{"score": 85}]'
        mock_backend.retrieve_raw.assert_called_with(ref.storage_path)

    @pytest.mark.asyncio
    async def test_retrieve_not_found(self, scratch_store, mock_backend):
        """Test retrieving non-existent data."""
        mock_backend.exists_raw.return_value = False

        with pytest.raises(FileNotFoundError):
            await scratch_store.retrieve(
                handle="nonexistent",
                user_id="user_123",
                run_id="run_456",
            )

    @pytest.mark.asyncio
    async def test_delete_data(self, scratch_store, mock_backend):
        """Test deleting stored data."""
        # Store first
        await scratch_store.store(
            user_id="user_123",
            run_id="run_456",
            handle="to_delete",
            data={"key": "value"},
        )

        result = await scratch_store.delete(
            handle="to_delete",
            user_id="user_123",
            run_id="run_456",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_run_scratch(self, scratch_store, mock_backend):
        """Test deleting all scratch data for a run."""
        count = await scratch_store.delete_run_scratch(
            user_id="user_123",
            run_id="run_456",
        )

        assert count == 5
        mock_backend.delete_by_prefix.assert_called_once_with(
            "user_123/run_456/scratch/"
        )


@pytest.mark.unit
class TestScratchStoreSingleton:
    """Tests for ScratchStore singleton pattern."""

    def test_set_instance(self, mock_backend):
        """Test setting a custom instance."""
        ScratchStore.reset_instance()

        custom_store = ScratchStore(mock_backend)
        ScratchStore.set_instance(custom_store)

        assert ScratchStore._instance is custom_store

    def test_reset_instance(self, mock_backend):
        """Test resetting the singleton."""
        ScratchStore.set_instance(ScratchStore(mock_backend))
        ScratchStore.reset_instance()

        assert ScratchStore._instance is None

    def test_instance_with_config(self):
        """Test instance() creates from config."""
        ScratchStore.reset_instance()

        with patch("seer.core.scratch.store._create_backend_from_config") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend

            store = ScratchStore.instance()

            assert store is not None
            mock_create.assert_called_once()

        ScratchStore.reset_instance()
