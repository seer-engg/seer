"""
Tests for LLM block file input support.

These tests verify that:
- File references in LLM inputs are detected and resolved
- Images are converted to multimodal content blocks
- Documents have text extracted and appended to prompt
- Backward compatibility is maintained (no files = same behavior)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.files.models import WorkflowFileRef, is_file_ref
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.schema_registry import SchemaRegistry
from seer.database import User


def _create_file_ref(
    file_id: str = "test-file-123",
    filename: str = "document.pdf",
    mime_type: str = "application/pdf",
    size_bytes: int = 1024,
) -> dict:
    """Create a file reference dict for testing."""
    ref = WorkflowFileRef(
        file_id=file_id,
        storage_path=f"s3://bucket/user/run/{file_id}/{filename}",
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        workflow_run_id="run_123",
        created_at=datetime.now(timezone.utc),
    )
    return ref.to_dict()


def _create_mock_runtime_services() -> RuntimeServices:
    """Create mock runtime services for testing."""
    from seer.core.expr.typecheck import TypeEnvironment

    return RuntimeServices(
        schema_registry=SchemaRegistry(),
        tool_registry=ToolRegistry(),
        model_registry=ModelRegistry(),
        type_env=TypeEnvironment(),
    )


def _create_mock_context_with_file_system() -> MagicMock:
    """Create a mock workflow context with file system."""
    mock_user = MagicMock(spec=User)
    mock_user.user_id = "usr_test"

    # Create mock context (must be MagicMock to allow property mocking)
    context = MagicMock(spec=WorkflowRuntimeContext)
    context.user = mock_user
    context.workflow_run_id = "run_test123"

    # Mock the file system
    mock_fs = AsyncMock()
    mock_fs.get_file_content = AsyncMock(return_value=b"file content bytes")
    context.file_system = mock_fs

    # has_file_system must return True for file resolution to work
    context.has_file_system = True

    return context


class TestFileRefDetection:
    """Tests for detecting file references in values."""

    def test_detect_file_ref_dict(self):
        """File ref dict is detected correctly."""
        file_ref = _create_file_ref()
        assert is_file_ref(file_ref) is True

    def test_detect_non_file_ref(self):
        """Non-file-ref values are not detected as file refs."""
        assert is_file_ref({"foo": "bar"}) is False
        assert is_file_ref("string value") is False
        assert is_file_ref(123) is False
        assert is_file_ref(None) is False
        assert is_file_ref([1, 2, 3]) is False

    def test_detect_file_ref_requires_type_marker(self):
        """File ref must have _type marker to be detected."""
        incomplete = {
            "file_id": "123",
            "filename": "test.pdf",
            # Missing _type
        }
        assert is_file_ref(incomplete) is False


class TestResolveLLMFileInputs:
    """Tests for _resolve_llm_file_inputs method."""

    @pytest.mark.asyncio
    async def test_resolve_single_file_ref(self):
        """Single file reference is resolved correctly."""
        runtime = NodeRuntime(_create_mock_runtime_services())
        context = _create_mock_context_with_file_system()

        file_ref = _create_file_ref(
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=2048,
        )

        auxiliary = {
            "document": file_ref,
            "other_param": "string value",
        }

        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        # File ref should be replaced with metadata
        assert "_resolved_file" in resolved["document"]
        assert resolved["document"]["_resolved_file"] == "report.pdf"
        assert resolved["document"]["mime_type"] == "application/pdf"
        assert resolved["document"]["size_bytes"] == 2048

        # Other params unchanged
        assert resolved["other_param"] == "string value"

        # File content should be in file_contents list
        assert len(file_contents) == 1
        assert file_contents[0]["filename"] == "report.pdf"
        assert file_contents[0]["mime_type"] == "application/pdf"
        assert file_contents[0]["content"] == b"file content bytes"

    @pytest.mark.asyncio
    async def test_resolve_list_of_file_refs(self):
        """List of file references is resolved correctly."""
        runtime = NodeRuntime(_create_mock_runtime_services())
        context = _create_mock_context_with_file_system()

        file_ref1 = _create_file_ref(filename="image1.png", mime_type="image/png")
        file_ref2 = _create_file_ref(filename="image2.png", mime_type="image/png")

        auxiliary = {
            "attachments": [file_ref1, file_ref2],
        }

        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        # List items should be resolved
        assert len(resolved["attachments"]) == 2
        assert resolved["attachments"][0]["_resolved_file"] == "image1.png"
        assert resolved["attachments"][1]["_resolved_file"] == "image2.png"

        # Both files in content list
        assert len(file_contents) == 2

    @pytest.mark.asyncio
    async def test_resolve_mixed_list(self):
        """List with mix of file refs and other values is handled."""
        runtime = NodeRuntime(_create_mock_runtime_services())
        context = _create_mock_context_with_file_system()

        file_ref = _create_file_ref(filename="doc.pdf")

        auxiliary = {
            "items": [file_ref, "string", 123],
        }

        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        # Mixed list resolved
        assert "_resolved_file" in resolved["items"][0]
        assert resolved["items"][1] == "string"
        assert resolved["items"][2] == 123

        # Only file ref in content list
        assert len(file_contents) == 1

    @pytest.mark.asyncio
    async def test_no_file_refs_returns_original(self):
        """When no file refs, original inputs returned unchanged."""
        runtime = NodeRuntime(_create_mock_runtime_services())
        context = _create_mock_context_with_file_system()

        auxiliary = {
            "param1": "value1",
            "param2": 123,
            "nested": {"key": "value"},
        }

        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        assert resolved == auxiliary
        assert file_contents == []

    @pytest.mark.asyncio
    async def test_no_context_returns_original(self):
        """Without context, original inputs returned unchanged."""
        runtime = NodeRuntime(_create_mock_runtime_services())

        file_ref = _create_file_ref()
        auxiliary = {"document": file_ref}

        # No context provided
        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, None)

        # File ref not resolved (no file system)
        assert resolved == auxiliary
        assert file_contents == []

    @pytest.mark.asyncio
    async def test_context_without_file_system(self):
        """Context without file system returns original inputs."""
        runtime = NodeRuntime(_create_mock_runtime_services())

        mock_user = MagicMock(spec=User)
        context = WorkflowRuntimeContext(user=mock_user)

        file_ref = _create_file_ref()
        auxiliary = {"document": file_ref}

        # Mock has_file_system property to return False
        with patch.object(WorkflowRuntimeContext, "has_file_system", new_callable=lambda: property(lambda self: False)):
            resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        assert resolved == auxiliary
        assert file_contents == []


class TestFileContentProcessing:
    """Tests for file content processing in handlers."""

    @pytest.mark.asyncio
    async def test_image_creates_content_block(self):
        """Image file creates base64 content block."""
        from seer.core.runtime.file_processor import FileContentProcessor

        processor = FileContentProcessor()
        image_bytes = b"\x89PNG\r\n\x1a\ntest image data"

        file_contents = [
            {
                "key": "image",
                "mime_type": "image/png",
                "filename": "test.png",
                "content": image_bytes,
            }
        ]

        result = await processor.process_files(file_contents)

        assert len(result["image_blocks"]) == 1
        assert result["image_blocks"][0]["type"] == "image_url"
        assert "base64" in result["image_blocks"][0]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_document_extracts_text(self):
        """Document file has text extracted."""
        from seer.core.runtime.file_processor import FileContentProcessor

        processor = FileContentProcessor()

        with patch.object(processor, "_get_document_processor") as mock_get:
            mock_dp = AsyncMock()
            mock_dp.extract_text = AsyncMock(return_value="This is the document content.")
            mock_get.return_value = mock_dp

            file_contents = [
                {
                    "key": "doc",
                    "mime_type": "application/pdf",
                    "filename": "report.pdf",
                    "content": b"PDF bytes",
                }
            ]

            result = await processor.process_files(file_contents)

            assert "report.pdf" in result["extracted_text"]
            assert "This is the document content." in result["extracted_text"]


class TestBackwardCompatibility:
    """Tests for backward compatibility when no files are present."""

    @pytest.mark.asyncio
    async def test_no_files_same_behavior(self):
        """When no file_contents in invocation, behavior unchanged."""
        # This is implicitly tested by other tests, but making it explicit
        runtime = NodeRuntime(_create_mock_runtime_services())

        auxiliary = {"regular": "input"}
        context = None

        resolved, file_contents = await runtime._resolve_llm_file_inputs(auxiliary, context)

        assert resolved == auxiliary
        assert file_contents == []
