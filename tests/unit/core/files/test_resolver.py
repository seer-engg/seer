"""
Unit tests for the file resolver.

Tests the unified file resolution logic that converts file inputs
(WorkflowFileRef or static_file_ref) to actual file bytes.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.files.models import WORKFLOW_FILE_REF_TYPE, WorkflowFileRef
from seer.core.files.resolver import (
    FileResolutionError,
    resolve_file_input,
    resolve_file_inputs,
    validate_file_input_format,
)
from seer.core.files.schemas import STATIC_FILE_REF_TYPE


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_workflow_file_ref() -> dict:
    """Create a sample WorkflowFileRef dict."""
    return {
        "_type": WORKFLOW_FILE_REF_TYPE,
        "file_id": "abc-123",
        "storage_path": "s3://bucket/path/to/file.pdf",
        "filename": "document.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1024,
        "workflow_run_id": "run_456",
        "created_at": "2026-02-09T12:00:00+00:00",
    }


@pytest.fixture
def sample_static_file_ref() -> dict:
    """Create a sample static_file_ref dict."""
    return {
        "_type": STATIC_FILE_REF_TYPE,
        "file_id": "user-file-789",
    }


@pytest.fixture
def mock_context():
    """Create a mock workflow runtime context."""
    context = MagicMock()
    context.has_file_system = True
    context.workflow_run_id = "run_123"
    context.user = MagicMock()
    context.user.user_id = "user_456"
    context.file_system = MagicMock()
    return context


# =============================================================================
# validate_file_input_format Tests
# =============================================================================


@pytest.mark.unit
class TestValidateFileInputFormat:
    """Tests for validate_file_input_format function."""

    def test_valid_workflow_file_ref(self, sample_workflow_file_ref):
        """Test validation passes for WorkflowFileRef."""
        assert validate_file_input_format(sample_workflow_file_ref) is True

    def test_valid_static_file_ref(self, sample_static_file_ref):
        """Test validation passes for static_file_ref."""
        assert validate_file_input_format(sample_static_file_ref) is True

    def test_invalid_string(self):
        """Test validation fails for raw strings."""
        assert validate_file_input_format("some_base64_string") is False

    def test_invalid_dict_no_type(self):
        """Test validation fails for dict without _type."""
        assert validate_file_input_format({"file_id": "abc"}) is False

    def test_invalid_none(self):
        """Test validation fails for None."""
        assert validate_file_input_format(None) is False


# =============================================================================
# resolve_file_input Tests - WorkflowFileRef
# =============================================================================


@pytest.mark.unit
class TestResolveWorkflowFileRef:
    """Tests for resolving WorkflowFileRef inputs."""

    @pytest.mark.asyncio
    async def test_resolve_workflow_file_ref_success(self, sample_workflow_file_ref, mock_context):
        """Test successful resolution of WorkflowFileRef."""
        expected_content = b"PDF content here"
        mock_context.file_system.get_file_content = AsyncMock(return_value=expected_content)
        mock_context.file_system.parse_file_ref = MagicMock(
            return_value=WorkflowFileRef(
                file_id="abc-123",
                storage_path="s3://bucket/path/to/file.pdf",
                filename="document.pdf",
                mime_type="application/pdf",
                size_bytes=1024,
                workflow_run_id="run_456",
                created_at=datetime.now(timezone.utc),
            )
        )

        content, mime_type, filename = await resolve_file_input(sample_workflow_file_ref, mock_context)

        assert content == expected_content
        assert mime_type == "application/pdf"
        assert filename == "document.pdf"
        mock_context.file_system.get_file_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_workflow_file_ref_no_context(self, sample_workflow_file_ref):
        """Test resolution fails without context."""
        with pytest.raises(FileResolutionError, match="file system not available"):
            await resolve_file_input(sample_workflow_file_ref, None)

    @pytest.mark.asyncio
    async def test_resolve_workflow_file_ref_no_file_system(self, sample_workflow_file_ref, mock_context):
        """Test resolution fails when file system not available."""
        mock_context.has_file_system = False

        with pytest.raises(FileResolutionError, match="file system not available"):
            await resolve_file_input(sample_workflow_file_ref, mock_context)


# =============================================================================
# resolve_file_input Tests - static_file_ref
# =============================================================================


@pytest.mark.unit
class TestResolveStaticFileRef:
    """Tests for resolving static_file_ref inputs."""

    @pytest.mark.asyncio
    async def test_resolve_static_file_ref_success(self, sample_static_file_ref, mock_context):
        """Test successful resolution of static_file_ref."""
        expected_content = b"Static file content"
        file_ref = WorkflowFileRef(
            file_id="user-file-789",
            storage_path="s3://bucket/user/file.txt",
            filename="userfile.txt",
            mime_type="text/plain",
            size_bytes=100,
            workflow_run_id="",
            created_at=datetime.now(timezone.utc),
        )
        mock_context.file_system.get_file_by_id = AsyncMock(return_value=(expected_content, file_ref))

        content, mime_type, filename = await resolve_file_input(sample_static_file_ref, mock_context)

        assert content == expected_content
        assert mime_type == "text/plain"
        assert filename == "userfile.txt"
        mock_context.file_system.get_file_by_id.assert_called_once_with("user-file-789", mock_context.user)

    @pytest.mark.asyncio
    async def test_resolve_static_file_ref_no_context(self, sample_static_file_ref):
        """Test resolution fails without context."""
        with pytest.raises(FileResolutionError, match="no workflow context"):
            await resolve_file_input(sample_static_file_ref, None)

    @pytest.mark.asyncio
    async def test_resolve_static_file_ref_no_file_id(self, mock_context):
        """Test resolution fails when file_id missing."""
        bad_ref = {"_type": STATIC_FILE_REF_TYPE}  # Missing file_id

        with pytest.raises(ValueError, match="must have a file_id"):
            await resolve_file_input(bad_ref, mock_context)


# =============================================================================
# resolve_file_input Tests - Error Cases
# =============================================================================


@pytest.mark.unit
class TestResolveFileInputErrors:
    """Tests for error handling in resolve_file_input."""

    @pytest.mark.asyncio
    async def test_reject_none_input(self, mock_context):
        """Test None input is rejected."""
        with pytest.raises(ValueError, match="cannot be None"):
            await resolve_file_input(None, mock_context)

    @pytest.mark.asyncio
    async def test_reject_raw_base64_string(self, mock_context):
        """Test raw base64 strings are rejected with clear message."""
        with pytest.raises(ValueError, match="Raw base64 input is not supported"):
            await resolve_file_input("SGVsbG8gV29ybGQh", mock_context)

    @pytest.mark.asyncio
    async def test_reject_invalid_format(self, mock_context):
        """Test invalid formats are rejected."""
        with pytest.raises(ValueError, match="Invalid file input format"):
            await resolve_file_input(12345, mock_context)

    @pytest.mark.asyncio
    async def test_reject_dict_without_type(self, mock_context):
        """Test dicts without _type are rejected."""
        with pytest.raises(ValueError, match="Invalid file input format"):
            await resolve_file_input({"file_id": "abc"}, mock_context)


# =============================================================================
# resolve_file_inputs Tests (Batch Resolution)
# =============================================================================


@pytest.mark.unit
class TestResolveFileInputs:
    """Tests for batch file resolution."""

    @pytest.mark.asyncio
    async def test_resolve_multiple_files(self, sample_workflow_file_ref, mock_context):
        """Test resolving multiple files."""
        mock_context.file_system.get_file_content = AsyncMock(return_value=b"content")
        mock_context.file_system.parse_file_ref = MagicMock(
            return_value=WorkflowFileRef(
                file_id="abc-123",
                storage_path="s3://bucket/path/to/file.pdf",
                filename="file.pdf",
                mime_type="application/pdf",
                size_bytes=100,
                workflow_run_id="run_456",
                created_at=datetime.now(timezone.utc),
            )
        )

        results = await resolve_file_inputs([sample_workflow_file_ref, sample_workflow_file_ref], mock_context)

        assert len(results) == 2
        assert all(r[0] == b"content" for r in results)

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self, mock_context):
        """Test resolving empty list."""
        results = await resolve_file_inputs([], mock_context)
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_error_includes_index(self, sample_workflow_file_ref, mock_context):
        """Test batch errors include the file index."""
        inputs = [
            sample_workflow_file_ref,  # First file is valid
            "invalid_string",  # This should fail at index 1
        ]

        mock_context.file_system.get_file_content = AsyncMock(return_value=b"content")
        mock_context.file_system.parse_file_ref = MagicMock(
            return_value=WorkflowFileRef(
                file_id="abc-123",
                storage_path="s3://bucket/file",
                filename="file.txt",
                mime_type="text/plain",
                size_bytes=10,
                workflow_run_id="run",
                created_at=datetime.now(timezone.utc),
            )
        )

        with pytest.raises(FileResolutionError, match="index 1"):
            await resolve_file_inputs(inputs, mock_context)
