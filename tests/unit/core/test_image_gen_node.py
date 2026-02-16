# pylint: disable=unused-argument
# Reason: Mock handlers require specific function signatures even if not all params are used
"""
Tests for ImageGenNode - image generation node type.

Tests Pydantic model validation, spec parsing, compilation, and execution
with mocked OpenRouter API calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.models import ImageGenNode
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pydantic model validation tests
# ---------------------------------------------------------------------------

class TestImageGenNodeModel:
    """Tests for ImageGenNode Pydantic model validation."""

    def test_valid_image_gen_node(self):
        """ImageGenNode with required inputs should parse correctly."""
        node = ImageGenNode(
            id="img-1",
            inputs={"model": "sourceful/riverflow-v2-fast", "prompt": "A cat"},
        )
        assert node.type == "image_gen"
        assert node.inputs["model"] == "sourceful/riverflow-v2-fast"
        assert node.inputs["prompt"] == "A cat"

    def test_image_gen_node_with_optional_fields(self):
        """ImageGenNode should accept optional size and num_images inputs."""
        node = ImageGenNode(
            id="img-2",
            inputs={
                "model": "google/gemini-2.5-flash-image",
                "prompt": "A sunset over mountains",
                "size": "1024x768",
                "num_images": 1,
            },
        )
        assert node.inputs["size"] == "1024x768"
        assert node.inputs["num_images"] == 1

    def test_image_gen_node_missing_model(self):
        """ImageGenNode should fail validation when model is missing."""
        with pytest.raises(ValueError, match="model"):
            ImageGenNode(id="img-3", inputs={"prompt": "A cat"})

    def test_image_gen_node_missing_prompt(self):
        """ImageGenNode should fail validation when prompt is missing."""
        with pytest.raises(ValueError, match="prompt"):
            ImageGenNode(id="img-4", inputs={"model": "sourceful/riverflow-v2-fast"})

    def test_image_gen_node_missing_both(self):
        """ImageGenNode should fail validation when both required inputs are missing."""
        with pytest.raises(ValueError, match="model.*prompt|prompt.*model"):
            ImageGenNode(id="img-5", inputs={})

    def test_image_gen_node_type_literal(self):
        """ImageGenNode type should be 'image_gen'."""
        node = ImageGenNode(
            id="img-6",
            inputs={"model": "test-model", "prompt": "test"},
        )
        assert node.type == "image_gen"


# ---------------------------------------------------------------------------
# Workflow spec parsing tests
# ---------------------------------------------------------------------------

class TestImageGenNodeParsing:
    """Tests for parsing workflow specs containing image_gen nodes."""

    def test_parse_spec_with_image_gen_node(self):
        """Should parse a workflow spec containing an image_gen node."""
        spec_payload = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "Generate a landscape painting",
                        "size": "1024x1024",
                    },
                }
            ],
            "edges": [],
        }
        spec = parse_workflow_spec(spec_payload)
        assert len(spec.nodes) == 1
        assert spec.nodes[0].type == "image_gen"
        assert isinstance(spec.nodes[0], ImageGenNode)

    def test_parse_spec_image_gen_with_template_prompt(self):
        """Should parse image_gen node with template expressions in prompt."""
        spec_payload = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "google/gemini-2.5-flash-image",
                        "prompt": "Generate an image of ${trigger.subject}",
                    },
                }
            ],
            "edges": [],
        }
        spec = parse_workflow_spec(spec_payload)
        node = spec.nodes[0]
        assert isinstance(node, ImageGenNode)
        assert "${trigger.subject}" in str(node.inputs["prompt"])


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------

async def _compile_image_gen_workflow(spec_payload: dict) -> CompiledWorkflow:
    """Helper to compile a workflow containing image_gen nodes."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    model_registry = ModelRegistry()

    spec = parse_workflow_spec(spec_payload)
    type_env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)

    runtime = NodeRuntime(
        RuntimeServices(
            schema_registry=schema_registry,
            tool_registry=tool_registry,
            model_registry=model_registry,
            type_env=type_env,
        )
    )
    graph = await emit_langgraph(plan, runtime)
    return CompiledWorkflow(
        spec=spec,
        type_env=type_env.as_dict(),
        graph=graph,
        runtime=runtime,
    )


class TestImageGenNodeCompilation:
    """Tests for compiling workflows with image_gen nodes."""

    @pytest.mark.asyncio
    async def test_compile_image_gen_workflow(self):
        """Should compile a workflow with an image_gen node."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A beautiful sunset",
                        "size": "1024x1024",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)
        assert compiled is not None
        assert compiled.spec.nodes[0].type == "image_gen"

    @pytest.mark.asyncio
    async def test_compile_image_gen_type_env_registered(self):
        """Type environment should contain image_gen node output schema."""
        spec = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A cat",
                    },
                }
            ],
            "edges": [],
        }

        compiled = await _compile_image_gen_workflow(spec)
        # Type environment should have the image_gen node's output schema
        assert "img-1" in compiled.type_env
        schema = compiled.type_env["img-1"]
        assert schema["type"] == "object"
        assert "image_url" in schema["properties"]


# ---------------------------------------------------------------------------
# Execution tests (with mocked OpenRouter API)
# ---------------------------------------------------------------------------

class TestImageGenNodeExecution:
    """Tests for image_gen node execution with mocked OpenRouter API."""

    @pytest.mark.asyncio
    async def test_execute_image_gen_node(self):
        """Should execute an image_gen node and return image URL."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A cat sitting on a chair",
                        "size": "1024x1024",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock the OpenRouter API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "https://example.com/generated-image.png",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            "model": "sourceful/riverflow-v2-fast",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
            )

        # Verify image URL is in the output
        assert "img-1" in result
        img_result = result["img-1"]
        assert img_result["image_url"] == "https://example.com/generated-image.png"
        assert img_result["prompt"] == "A cat sitting on a chair"
        assert img_result["model"] == "sourceful/riverflow-v2-fast"
        assert img_result["size"] == "1024x1024"

    @pytest.mark.asyncio
    async def test_execute_image_gen_with_multimodal_response(self):
        """Should handle multimodal response format from OpenRouter."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "google/gemini-2.5-flash-image",
                        "prompt": "A dog",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock multimodal response format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/dog.png"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "model": "google/gemini-2.5-flash-image",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
            )

        assert "img-1" in result
        assert result["img-1"]["image_url"] == "https://example.com/dog.png"

    @pytest.mark.asyncio
    async def test_execute_image_gen_with_images_array(self):
        """Should handle OpenRouter response with message.images array format."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A forest scene",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock response with message.images array (actual OpenRouter format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",  # Empty content - image is in images array
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "sourceful/riverflow-v2-fast",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
            )

        assert "img-1" in result
        # Should extract the base64 data URL from message.images
        assert result["img-1"]["image_url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_execute_image_gen_with_images_array_url_format(self):
        """Should handle message.images array with direct url field."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A mountain",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock response with message.images array using direct url field
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "images": [
                            {"url": "https://cdn.example.com/generated-mountain.png"},
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            "model": "sourceful/riverflow-v2-fast",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
            )

        assert "img-1" in result
        assert result["img-1"]["image_url"] == "https://cdn.example.com/generated-mountain.png"


# ---------------------------------------------------------------------------
# Image download and storage tests
# ---------------------------------------------------------------------------

class TestImageGenNodeDownload:
    """Tests for image download functionality."""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """Should download image from URL and detect MIME type."""
        from seer.core.nodes.image_gen_node import ImageGenNodeType

        node_type = ImageGenNodeType()

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"\x89PNG\r\n\x1a\n"  # PNG header bytes
        mock_response.headers = {"content-type": "image/png"}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client):
            image_bytes, mime_type = await node_type._download_image("https://example.com/image.png")

        assert image_bytes == b"\x89PNG\r\n\x1a\n"
        assert mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_download_image_failure(self):
        """Should raise ExecutionError on HTTP failure."""
        from seer.core.errors import ExecutionError
        from seer.core.nodes.image_gen_node import ImageGenNodeType

        node_type = ImageGenNodeType()

        # Mock failed HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ExecutionError, match="Failed to download image"):
                await node_type._download_image("https://example.com/notfound.png")

    @pytest.mark.asyncio
    async def test_download_data_url(self):
        """Should parse data URL and extract bytes."""
        import base64

        from seer.core.nodes.image_gen_node import ImageGenNodeType

        node_type = ImageGenNodeType()

        # Create a data URL
        image_data = b"fake-image-data"
        b64_data = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_data}"

        image_bytes, mime_type = await node_type._download_image(data_url)

        assert image_bytes == image_data
        assert mime_type == "image/jpeg"

    def test_detect_mime_from_url(self):
        """Should detect MIME type from URL extension."""
        from seer.core.nodes.image_gen_node import ImageGenNodeType

        assert ImageGenNodeType._detect_mime_from_url("https://example.com/image.png") == "image/png"
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/photo.jpg") == "image/jpeg"
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/photo.jpeg") == "image/jpeg"
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/anim.gif") == "image/gif"
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/image.webp") == "image/webp"
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/unknown") == "image/png"  # default
        # Should handle query params
        assert ImageGenNodeType._detect_mime_from_url("https://example.com/image.png?token=abc") == "image/png"


class TestImageGenNodeStorage:
    """Tests for image storage functionality."""

    @pytest.mark.asyncio
    async def test_store_image_without_runtime_context(self):
        """Should return None when runtime context is not available."""
        from seer.core.nodes.image_gen_node import ImageGenNodeType
        from seer.core.nodes.base import NodeExecutionContext

        node_type = ImageGenNodeType()

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx=None,
            runtime_context=None,  # No context
            trigger={},
        )

        result = await node_type._store_image(
            image_data=b"fake-image",
            mime_type="image/png",
            node_id="img-1",
            ctx=ctx,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_store_image_without_file_system(self):
        """Should return None when file system is not configured."""
        from seer.core.nodes.image_gen_node import ImageGenNodeType
        from seer.core.nodes.base import NodeExecutionContext

        node_type = ImageGenNodeType()

        # Mock runtime context without file system
        mock_runtime_ctx = MagicMock()
        mock_runtime_ctx.workflow_run_id = "run_123"
        mock_runtime_ctx.has_file_system = False

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx=None,
            runtime_context=mock_runtime_ctx,
            trigger={},
        )

        result = await node_type._store_image(
            image_data=b"fake-image",
            mime_type="image/png",
            node_id="img-1",
            ctx=ctx,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_store_image_with_file_system(self):
        """Should store image and return WorkflowFileRef dict."""
        from seer.core.nodes.image_gen_node import ImageGenNodeType
        from seer.core.nodes.base import NodeExecutionContext

        node_type = ImageGenNodeType()

        # Mock file ref
        mock_file_ref = MagicMock()
        mock_file_ref.file_id = "file_abc123"
        mock_file_ref.to_dict.return_value = {
            "file_id": "file_abc123",
            "filename": "image_gen_img-1_12345678.png",
            "storage_path": "s3://bucket/files/file_abc123",
            "mime_type": "image/png",
            "_type": "workflow_file_ref",
        }

        # Mock file system
        mock_file_system = AsyncMock()
        mock_file_system.store_file_with_record.return_value = mock_file_ref

        # Mock runtime context
        mock_runtime_ctx = MagicMock()
        mock_runtime_ctx.workflow_run_id = "run_123"
        mock_runtime_ctx.has_file_system = True
        mock_runtime_ctx.file_system = mock_file_system
        mock_runtime_ctx.user = MagicMock()

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx=None,
            runtime_context=mock_runtime_ctx,
            trigger={},
        )

        result = await node_type._store_image(
            image_data=b"fake-image-data",
            mime_type="image/png",
            node_id="img-1",
            ctx=ctx,
        )

        assert result is not None
        assert result["file_id"] == "file_abc123"
        assert result["_type"] == "workflow_file_ref"

        # Verify store_file_with_record was called correctly
        mock_file_system.store_file_with_record.assert_called_once()
        call_kwargs = mock_file_system.store_file_with_record.call_args.kwargs
        assert call_kwargs["data"] == b"fake-image-data"
        assert call_kwargs["mime_type"] == "image/png"
        assert call_kwargs["source_tool"] == "image_gen_node"


class TestImageGenNodeExecutionWithStorage:
    """Tests for full execution with storage integration."""

    @pytest.mark.asyncio
    async def test_execute_with_storage(self):
        """Should execute image_gen node and store image to file system."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A beautiful landscape",
                        "size": "1024x1024",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock OpenRouter API response
        mock_openrouter_response = MagicMock()
        mock_openrouter_response.status_code = 200
        mock_openrouter_response.json.return_value = {
            "choices": [{"message": {"content": "https://example.com/generated.png"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "sourceful/riverflow-v2-fast",
        }

        # Mock image download response
        mock_download_response = MagicMock()
        mock_download_response.status_code = 200
        mock_download_response.content = b"\x89PNG\r\n\x1a\n"
        mock_download_response.headers = {"content-type": "image/png"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_openrouter_response
        mock_client.get.return_value = mock_download_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock file system
        mock_file_ref = MagicMock()
        mock_file_ref.file_id = "file_stored_123"
        mock_file_ref.to_dict.return_value = {
            "file_id": "file_stored_123",
            "filename": "image_gen_img-1_abcd1234.png",
            "_type": "workflow_file_ref",
        }

        mock_file_system = AsyncMock()
        mock_file_system.store_file_with_record.return_value = mock_file_ref

        # Mock runtime context with file system
        mock_runtime_ctx = MagicMock()
        mock_runtime_ctx.workflow_run_id = "run_test_123"
        mock_runtime_ctx.has_file_system = True
        mock_runtime_ctx.file_system = mock_file_system
        mock_runtime_ctx.user = MagicMock()

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
                context=mock_runtime_ctx,  # Use 'context' parameter name
            )

        # Verify result contains both image_url and file
        assert "img-1" in result
        img_result = result["img-1"]
        assert img_result["image_url"] == "https://example.com/generated.png"
        assert "file" in img_result
        assert img_result["file"]["file_id"] == "file_stored_123"
        assert img_result["file"]["_type"] == "workflow_file_ref"

    @pytest.mark.asyncio
    async def test_execute_graceful_degradation_on_download_failure(self):
        """Should continue with just URL if download fails."""
        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "test.trigger",
                    "title": "Test",
                    "provider": "test",
                    "mode": "webhook",
                    "schemas": {"event": {"type": "object"}},
                }
            ],
            "nodes": [
                {
                    "id": "img-1",
                    "type": "image_gen",
                    "inputs": {
                        "model": "sourceful/riverflow-v2-fast",
                        "prompt": "A landscape",
                    },
                }
            ],
            "edges": [
                {"source": "trigger-1", "target": "img-1", "type": "trigger"},
            ],
        }

        compiled = await _compile_image_gen_workflow(spec)

        # Mock OpenRouter API response
        mock_openrouter_response = MagicMock()
        mock_openrouter_response.status_code = 200
        mock_openrouter_response.json.return_value = {
            "choices": [{"message": {"content": "https://example.com/image.png"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "model": "sourceful/riverflow-v2-fast",
        }

        # Mock failed download
        mock_download_response = MagicMock()
        mock_download_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_openrouter_response
        mock_client.get.return_value = mock_download_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("seer.core.nodes.image_gen_node.httpx.AsyncClient", return_value=mock_client), \
             patch("seer.core.nodes.image_gen_node.config") as mock_config:
            mock_config.openrouter_api_key = "test-api-key"

            result = await compiled.ainvoke(
                workflow_input=None,
                trigger={"trigger-1": {}},
            )

        # Should still have result with image_url, but no file
        assert "img-1" in result
        img_result = result["img-1"]
        assert img_result["image_url"] == "https://example.com/image.png"
        assert "file" not in img_result  # No file since download failed
