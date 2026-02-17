"""
Unit tests for file management API models.
"""

from datetime import datetime, timezone

import pytest

from seer.api.files.models import (
    UserFileListItem,
    UserFileListResponse,
    UserFileResponse,
    UserFileDownloadResponse,
    UserFileDeleteResponse,
    MimeTypeStats,
    ToolStats,
    UserStorageStatsResponse,
    BulkDeleteFilesRequest,
    BulkDeleteResult,
    BulkDeleteFilesResponse,
    FileSearchResponse,
    UserFileUploadResponse,
)


@pytest.mark.unit
class TestUserFileListItem:
    """Tests for UserFileListItem model."""

    def test_create_with_all_fields(self):
        """Test creating item with all fields populated."""
        item = UserFileListItem(
            file_id="abc-123",
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            size_human="1.0 KB",
            run_id="run_1",
            workflow_id="wf_1",
            workflow_name="My Workflow",
            source_node_id="node_1",
            source_tool="google_drive_download",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert item.file_id == "abc-123"
        assert item.filename == "document.pdf"
        assert item.run_id == "run_1"
        assert item.workflow_id == "wf_1"

    def test_create_with_optional_fields_null(self):
        """Test creating item with optional fields null (user upload)."""
        item = UserFileListItem(
            file_id="abc-123",
            filename="upload.txt",
            mime_type="text/plain",
            size_bytes=100,
            size_human="100 B",
            run_id=None,
            workflow_id=None,
            workflow_name=None,
            source_node_id=None,
            source_tool="user_upload",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert item.run_id is None
        assert item.workflow_id is None
        assert item.source_tool == "user_upload"


@pytest.mark.unit
class TestUserFileListResponse:
    """Tests for UserFileListResponse model."""

    def test_empty_response(self):
        """Test response with no files."""
        response = UserFileListResponse(
            files=[],
            total_count=0,
            total_size_bytes=0,
            next_cursor=None,
        )
        assert len(response.files) == 0
        assert response.total_count == 0
        assert response.next_cursor is None

    def test_response_with_pagination(self):
        """Test response with pagination cursor."""
        item = UserFileListItem(
            file_id="abc-123",
            filename="test.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            size_human="1.0 KB",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        response = UserFileListResponse(
            files=[item],
            total_count=100,
            total_size_bytes=102400,
            next_cursor="abc-123",
        )
        assert len(response.files) == 1
        assert response.total_count == 100
        assert response.next_cursor == "abc-123"


@pytest.mark.unit
class TestUserStorageStatsResponse:
    """Tests for UserStorageStatsResponse model."""

    def test_empty_stats(self):
        """Test empty storage stats."""
        stats = UserStorageStatsResponse(
            total_files=0,
            total_size_bytes=0,
            total_size_human="0 B",
            files_by_mime_type=[],
            files_by_tool=[],
        )
        assert stats.total_files == 0
        assert stats.total_size_human == "0 B"

    def test_stats_with_breakdown(self):
        """Test storage stats with breakdowns."""
        stats = UserStorageStatsResponse(
            total_files=10,
            total_size_bytes=1048576,
            total_size_human="1.0 MB",
            files_by_mime_type=[
                MimeTypeStats(
                    mime_type="application/pdf",
                    file_count=5,
                    total_size_bytes=524288,
                    total_size_human="512.0 KB",
                ),
                MimeTypeStats(
                    mime_type="image/png",
                    file_count=5,
                    total_size_bytes=524288,
                    total_size_human="512.0 KB",
                ),
            ],
            files_by_tool=[
                ToolStats(
                    source_tool="google_drive_download",
                    file_count=10,
                    total_size_bytes=1048576,
                ),
            ],
            oldest_file_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            newest_file_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        assert stats.total_files == 10
        assert len(stats.files_by_mime_type) == 2
        assert len(stats.files_by_tool) == 1


@pytest.mark.unit
class TestBulkDeleteFilesRequest:
    """Tests for BulkDeleteFilesRequest model."""

    def test_valid_request(self):
        """Test valid bulk delete request."""
        request = BulkDeleteFilesRequest(file_ids=["abc-123", "def-456"])
        assert len(request.file_ids) == 2

    def test_single_file_request(self):
        """Test bulk delete with single file."""
        request = BulkDeleteFilesRequest(file_ids=["abc-123"])
        assert len(request.file_ids) == 1

    def test_empty_list_fails_validation(self):
        """Test empty list fails validation."""
        with pytest.raises(ValueError):
            BulkDeleteFilesRequest(file_ids=[])

    def test_max_100_files(self):
        """Test 100 files is allowed."""
        file_ids = [f"file-{i}" for i in range(100)]
        request = BulkDeleteFilesRequest(file_ids=file_ids)
        assert len(request.file_ids) == 100


@pytest.mark.unit
class TestBulkDeleteFilesResponse:
    """Tests for BulkDeleteFilesResponse model."""

    def test_all_successful(self):
        """Test response when all deletes succeed."""
        response = BulkDeleteFilesResponse(
            results=[
                BulkDeleteResult(file_id="abc-123", deleted=True),
                BulkDeleteResult(file_id="def-456", deleted=True),
            ],
            deleted_count=2,
            failed_count=0,
            total_size_freed_bytes=2048,
        )
        assert response.deleted_count == 2
        assert response.failed_count == 0

    def test_partial_failure(self):
        """Test response with some failures."""
        response = BulkDeleteFilesResponse(
            results=[
                BulkDeleteResult(file_id="abc-123", deleted=True),
                BulkDeleteResult(file_id="def-456", deleted=False, error="File not found"),
            ],
            deleted_count=1,
            failed_count=1,
            total_size_freed_bytes=1024,
        )
        assert response.deleted_count == 1
        assert response.failed_count == 1
        assert response.results[1].error == "File not found"


@pytest.mark.unit
class TestFileSearchResponse:
    """Tests for FileSearchResponse model."""

    def test_search_response(self):
        """Test search response."""
        item = UserFileListItem(
            file_id="abc-123",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            size_human="1.0 KB",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        response = FileSearchResponse(
            query="report",
            results=[item],
            total_matches=5,
        )
        assert response.query == "report"
        assert len(response.results) == 1
        assert response.total_matches == 5
