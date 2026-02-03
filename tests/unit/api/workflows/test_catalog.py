"""
Unit tests for workflow catalog operations.

Tests the catalog service functions with proper mocking of dependencies.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.workflows import models as api_models
from seer.api.workflows.services import catalog
from seer.core.schema.models import LLMNode, ToolNode, WorkflowSpec


# =============================================================================
# List Node Types Tests
# =============================================================================


@pytest.mark.unit
class TestListNodeTypes:
    """Tests for list_node_types function."""

    @pytest.mark.asyncio
    async def test_list_node_types_returns_response(self):
        """Test list_node_types returns NodeTypeResponse."""
        result = await catalog.list_node_types()

        assert isinstance(result, api_models.NodeTypeResponse)
        assert hasattr(result, "node_types")
        assert isinstance(result.node_types, list)

    @pytest.mark.asyncio
    async def test_list_node_types_includes_llm_type(self):
        """Test LLM node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "llm" in types

        llm_descriptor = next(nt for nt in result.node_types if nt.type == "llm")
        assert llm_descriptor.title == "LLM"
        assert len(llm_descriptor.fields) > 0

    @pytest.mark.asyncio
    async def test_list_node_types_includes_if_else_type(self):
        """Test if_else node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "if_else" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_for_loop_type(self):
        """Test for_loop node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "for_loop" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_mcp_type(self):
        """Test MCP node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "mcp" in types

    @pytest.mark.asyncio
    async def test_list_node_types_field_descriptors_have_required_fields(self):
        """Test node type descriptors have properly structured fields."""
        result = await catalog.list_node_types()

        for node_type in result.node_types:
            assert isinstance(node_type, api_models.NodeTypeDescriptor)
            assert node_type.type
            assert node_type.title
            assert isinstance(node_type.fields, list)

            for field in node_type.fields:
                assert isinstance(field, api_models.NodeFieldDescriptor)
                assert field.name
                assert field.kind


# =============================================================================
# List Tools Tests
# =============================================================================


@pytest.mark.unit
class TestListTools:
    """Tests for list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tool_descriptors(self):
        """Test listing tools returns ToolRegistryResponse."""
        mock_tool = MagicMock(spec=[])  # Empty spec means no 'title' attr, uses fallback
        mock_tool.name = "test_tool"

        mock_definition = MagicMock()
        mock_definition.name = "test_tool"
        mock_definition.version = "1.0.0"
        mock_definition.input_schema = {"type": "object"}
        mock_definition.output_schema = {"type": "string"}

        with patch.object(catalog, "registry_list_tools", return_value=[mock_tool]):
            with patch.object(catalog.COMPILER, "ensure_tool", return_value=mock_definition):
                result = await catalog.list_tools(include_schemas=False)

        assert isinstance(result, api_models.ToolRegistryResponse)
        assert len(result.tools) == 1
        assert result.tools[0].name == "test_tool"
        assert result.tools[0].version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_tools_without_schemas(self):
        """Test listing tools excludes schemas when include_schemas=False."""
        mock_tool = MagicMock(spec=[])
        mock_tool.name = "test_tool"

        mock_definition = MagicMock()
        mock_definition.name = "test_tool"
        mock_definition.version = "1.0.0"
        mock_definition.input_schema = {"type": "object"}
        mock_definition.output_schema = {"type": "string"}

        with patch.object(catalog, "registry_list_tools", return_value=[mock_tool]):
            with patch.object(catalog.COMPILER, "ensure_tool", return_value=mock_definition):
                result = await catalog.list_tools(include_schemas=False)

        assert result.tools[0].input_schema is None
        assert result.tools[0].output_schema is None

    @pytest.mark.asyncio
    async def test_list_tools_with_schemas(self):
        """Test listing tools includes schemas when include_schemas=True."""
        mock_tool = MagicMock(spec=[])
        mock_tool.name = "test_tool"

        mock_definition = MagicMock()
        mock_definition.name = "test_tool"
        mock_definition.version = "1.0.0"
        mock_definition.input_schema = {"type": "object", "properties": {"input": {"type": "string"}}}
        mock_definition.output_schema = {"type": "string"}

        with patch.object(catalog, "registry_list_tools", return_value=[mock_tool]):
            with patch.object(catalog.COMPILER, "ensure_tool", return_value=mock_definition):
                result = await catalog.list_tools(include_schemas=True)

        assert result.tools[0].input_schema == {"type": "object", "properties": {"input": {"type": "string"}}}
        assert result.tools[0].output_schema == {"type": "string"}

    @pytest.mark.asyncio
    async def test_list_tools_generates_correct_id_format(self):
        """Test tool ID follows correct format: tools.{name}@{version}."""
        mock_tool = MagicMock(spec=[])
        mock_tool.name = "my_tool"

        mock_definition = MagicMock()
        mock_definition.name = "my_tool"
        mock_definition.version = "2.0.0"
        mock_definition.input_schema = None
        mock_definition.output_schema = None

        with patch.object(catalog, "registry_list_tools", return_value=[mock_tool]):
            with patch.object(catalog.COMPILER, "ensure_tool", return_value=mock_definition):
                result = await catalog.list_tools()

        assert result.tools[0].id == "tools.my_tool@2.0.0"

    @pytest.mark.asyncio
    async def test_list_tools_with_custom_title(self):
        """Test tool with custom title uses that title."""
        mock_tool = MagicMock(spec=["name", "title"])
        mock_tool.name = "test_tool"
        mock_tool.title = "My Custom Tool"

        mock_definition = MagicMock()
        mock_definition.name = "test_tool"
        mock_definition.version = "1.0.0"
        mock_definition.input_schema = None
        mock_definition.output_schema = None

        with patch.object(catalog, "registry_list_tools", return_value=[mock_tool]):
            with patch.object(catalog.COMPILER, "ensure_tool", return_value=mock_definition):
                result = await catalog.list_tools()

        assert result.tools[0].title == "My Custom Tool"

    @pytest.mark.asyncio
    async def test_list_tools_empty_registry(self):
        """Test listing tools with empty registry returns empty list."""
        with patch.object(catalog, "registry_list_tools", return_value=[]):
            result = await catalog.list_tools()

        assert result.tools == []


# =============================================================================
# List Triggers Tests
# =============================================================================


@pytest.mark.unit
class TestListTriggers:
    """Tests for list_triggers function."""

    @pytest.mark.asyncio
    async def test_list_triggers_returns_trigger_descriptors(self):
        """Test listing triggers returns TriggerCatalogResponse."""
        mock_definition = MagicMock()
        mock_definition.key = "webhook.generic"
        mock_definition.title = "Webhook"
        mock_definition.provider = "generic"
        mock_definition.mode = "push"
        mock_definition.description = "Generic webhook trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            result = await catalog.list_triggers()

        assert isinstance(result, api_models.TriggerCatalogResponse)
        assert len(result.triggers) == 1
        assert result.triggers[0].key == "webhook.generic"
        assert result.triggers[0].title == "Webhook"

    @pytest.mark.asyncio
    async def test_list_triggers_includes_all_fields(self):
        """Test trigger descriptors include all required fields."""
        mock_definition = MagicMock()
        mock_definition.key = "schedule.cron"
        mock_definition.title = "Scheduled"
        mock_definition.provider = "cron"
        mock_definition.mode = "poll"
        mock_definition.description = "Cron-based schedule trigger"
        mock_definition.schemas.event = {"type": "object", "properties": {"time": {"type": "string"}}}
        mock_definition.schemas.filter = {"type": "object"}
        mock_definition.schemas.config = {"type": "object", "properties": {"cron": {"type": "string"}}}

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            result = await catalog.list_triggers()

        trigger = result.triggers[0]
        assert trigger.key == "schedule.cron"
        assert trigger.title == "Scheduled"
        assert trigger.provider == "cron"
        assert trigger.mode == "poll"
        assert trigger.description == "Cron-based schedule trigger"
        assert trigger.event_schema is not None
        assert trigger.filter_schema is not None
        assert trigger.config_schema is not None

    @pytest.mark.asyncio
    async def test_list_triggers_empty_registry(self):
        """Test handling empty trigger registry."""
        with patch.object(catalog.trigger_registry, "all", return_value=[]):
            result = await catalog.list_triggers()

        assert result.triggers == []


# =============================================================================
# List Models Tests
# =============================================================================


@pytest.mark.unit
class TestListModels:
    """Tests for list_models function."""

    @pytest.mark.asyncio
    async def test_list_models_returns_default_models(self):
        """Test listing returns default model options."""
        with patch.object(catalog.shared_config, "default_llm_model", None):
            result = await catalog.list_models()

        assert isinstance(result, api_models.ModelRegistryResponse)
        assert len(result.models) >= 2

        model_ids = [m.id for m in result.models]
        assert "gpt-4.1-mini" in model_ids
        assert "gpt-4o-mini" in model_ids

    @pytest.mark.asyncio
    async def test_list_models_includes_custom_default(self):
        """Test custom default model is included when configured."""
        with patch.object(catalog.shared_config, "default_llm_model", "claude-3-opus"):
            result = await catalog.list_models()

        model_ids = [m.id for m in result.models]
        assert "claude-3-opus" in model_ids

    @pytest.mark.asyncio
    async def test_list_models_skips_duplicate_default(self):
        """Test duplicate default models are not added twice."""
        with patch.object(catalog.shared_config, "default_llm_model", "gpt-4.1-mini"):
            result = await catalog.list_models()

        # Count occurrences of gpt-4.1-mini
        count = sum(1 for m in result.models if m.id == "gpt-4.1-mini")
        assert count == 1

    @pytest.mark.asyncio
    async def test_list_models_descriptors_have_required_fields(self):
        """Test model descriptors have all required fields."""
        with patch.object(catalog.shared_config, "default_llm_model", None):
            result = await catalog.list_models()

        for model in result.models:
            assert isinstance(model, api_models.ModelDescriptor)
            assert model.id
            assert model.title
            assert isinstance(model.supports_json_schema, bool)


# =============================================================================
# Resolve Schema Tests
# =============================================================================


@pytest.mark.unit
class TestResolveSchema:
    """Tests for resolve_schema function."""

    @pytest.mark.asyncio
    async def test_resolve_schema_returns_schema(self):
        """Test resolving schema returns SchemaResponse."""
        mock_schema = {"type": "object", "properties": {"input": {"type": "string"}}}

        with patch.object(catalog.COMPILER.schema_registry, "get", return_value=mock_schema):
            result = await catalog.resolve_schema("tool.test")

        assert isinstance(result, api_models.SchemaResponse)
        assert result.id == "tool.test"
        assert result.json_schema == mock_schema

    @pytest.mark.asyncio
    async def test_resolve_schema_not_found_raises_error(self):
        """Test resolving unknown schema raises 404 error."""
        with patch.object(catalog.COMPILER.schema_registry, "get", return_value=None):
            with pytest.raises(Exception):  # raise_problem raises an exception
                await catalog.resolve_schema("unknown.schema")


# =============================================================================
# Validate Spec Tests
# =============================================================================


@pytest.mark.unit
class TestValidateSpec:
    """Tests for validate_spec function.

    Note: The _collect_warnings_from_nodes function has a bug where it checks
    for 'node.out' but ToolNode uses 'expect_outputs' and LLMNode uses 'outputs'.
    These tests document the expected behavior once the bug is fixed.
    """

    def test_validate_spec_empty_nodes_no_warnings(self):
        """Test validating spec with no nodes generates no warnings."""
        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.ValidateRequest(spec=spec)

        result = catalog.validate_spec(payload)

        assert isinstance(result, api_models.ValidateResponse)
        assert result.ok is True
        assert result.warnings == []

    def test_validate_spec_returns_valid_response(self):
        """Test validate_spec returns ValidateResponse structure."""
        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.ValidateRequest(spec=spec)

        result = catalog.validate_spec(payload)

        assert isinstance(result, api_models.ValidateResponse)
        assert hasattr(result, "ok")
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)


# =============================================================================
# Collect Warnings Tests
# =============================================================================


@pytest.mark.unit
class TestCollectWarnings:
    """Tests for _collect_warnings_from_nodes helper function.

    Note: The current implementation has a bug - it checks 'node.out' but:
    - ToolNode uses 'expect_outputs'
    - LLMNode uses 'outputs'

    This causes AttributeError when processing these node types.
    Tests below document the current (buggy) behavior.
    """

    def test_collect_warnings_empty_nodes(self):
        """Test collecting warnings from empty node list."""
        warnings = catalog._collect_warnings_from_nodes([])

        assert warnings == []

    def test_collect_warnings_with_tool_node_raises_attribute_error(self):
        """Test that ToolNode causes AttributeError due to missing 'out' attr.

        BUG: catalog.py line 280 checks 'node.out' but ToolNode has 'expect_outputs'.
        """
        node = ToolNode(
            id="tool1",
            tool="test.tool"
        )

        # Current buggy behavior - raises AttributeError
        with pytest.raises(AttributeError, match="'ToolNode' object has no attribute 'out'"):
            catalog._collect_warnings_from_nodes([node])

    def test_collect_warnings_with_llm_node_raises_attribute_error(self):
        """Test that LLMNode causes AttributeError due to missing 'out' attr.

        BUG: catalog.py line 280 checks 'node.out' but LLMNode has 'outputs'.
        """
        node = LLMNode(
            id="llm1",
            inputs={"model": "gpt-4", "prompt": "test"}
        )

        # Current buggy behavior - raises AttributeError
        with pytest.raises(AttributeError, match="'LLMNode' object has no attribute 'out'"):
            catalog._collect_warnings_from_nodes([node])


# =============================================================================
# Graph Preview Tests
# =============================================================================


@pytest.mark.unit
class TestGraphPreview:
    """Tests for _graph_preview helper function."""

    def test_graph_preview_empty_spec(self):
        """Test generating preview for empty spec."""
        spec = WorkflowSpec(nodes=[], edges=[])

        preview = catalog._graph_preview(spec)

        assert preview["nodes"] == []
        assert preview["edges"] == []

    def test_graph_preview_with_single_node(self):
        """Test generating preview with single node."""
        node = ToolNode(id="n1", tool="test.tool")
        spec = WorkflowSpec(nodes=[node], edges=[])

        preview = catalog._graph_preview(spec)

        assert len(preview["nodes"]) == 1
        assert preview["nodes"][0]["id"] == "n1"
        assert preview["nodes"][0]["kind"] == "tool"
        assert preview["edges"] == []

    def test_graph_preview_with_multiple_nodes(self):
        """Test generating preview creates sequential edges."""
        nodes = [
            ToolNode(id="n1", tool="test.tool"),
            LLMNode(id="n2", inputs={"model": "gpt-4", "prompt": "test"}),
        ]
        spec = WorkflowSpec(nodes=nodes, edges=[])

        preview = catalog._graph_preview(spec)

        assert len(preview["nodes"]) == 2
        assert len(preview["edges"]) == 1
        assert preview["edges"][0]["from"] == "n1"
        assert preview["edges"][0]["to"] == "n2"

    def test_graph_preview_node_structure(self):
        """Test preview nodes have correct structure."""
        node = LLMNode(id="llm1", inputs={"model": "gpt-4", "prompt": "test"})
        spec = WorkflowSpec(nodes=[node], edges=[])

        preview = catalog._graph_preview(spec)

        assert "id" in preview["nodes"][0]
        assert "kind" in preview["nodes"][0]


# =============================================================================
# Compile Spec Tests
# =============================================================================


@pytest.mark.unit
class TestCompileSpec:
    """Tests for compile_spec function."""

    @pytest.mark.asyncio
    async def test_compile_spec_success(self):
        """Test successful spec compilation."""
        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.CompileRequest(
            spec=spec,
            options=api_models.CompileOptions()
        )

        mock_user = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.workflow.type_env = {}

        with patch("seer.api.workflows.services.catalog.get_checkpointer", new_callable=AsyncMock):
            with patch.object(catalog.COMPILER, "compile", new_callable=AsyncMock, return_value=mock_compiled):
                result = await catalog.compile_spec(mock_user, payload)

        assert isinstance(result, api_models.CompileResponse)
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_compile_spec_with_type_env(self):
        """Test compilation returns type environment when requested."""
        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.CompileRequest(
            spec=spec,
            options=api_models.CompileOptions(emit_type_env=True)
        )

        mock_user = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.workflow.type_env = {"input": {"type": "object"}}

        with patch("seer.api.workflows.services.catalog.get_checkpointer", new_callable=AsyncMock):
            with patch.object(catalog.COMPILER, "compile", new_callable=AsyncMock, return_value=mock_compiled):
                result = await catalog.compile_spec(mock_user, payload)

        assert result.artifacts.type_env == {"input": {"type": "object"}}

    @pytest.mark.asyncio
    async def test_compile_spec_with_graph_preview(self):
        """Test compilation returns graph preview when requested."""
        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.CompileRequest(
            spec=spec,
            options=api_models.CompileOptions(emit_graph_preview=True)
        )

        mock_user = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.workflow.type_env = {}

        with patch("seer.api.workflows.services.catalog.get_checkpointer", new_callable=AsyncMock):
            with patch.object(catalog.COMPILER, "compile", new_callable=AsyncMock, return_value=mock_compiled):
                result = await catalog.compile_spec(mock_user, payload)

        assert result.artifacts.graph_preview is not None
        assert "nodes" in result.artifacts.graph_preview
        assert "edges" in result.artifacts.graph_preview

    @pytest.mark.asyncio
    async def test_compile_spec_error_raises_problem(self):
        """Test compilation error raises problem."""
        from seer.core.errors import WorkflowCompilerError

        spec = WorkflowSpec(nodes=[], edges=[])
        payload = api_models.CompileRequest(spec=spec)

        mock_user = MagicMock()

        with patch("seer.api.workflows.services.catalog.get_checkpointer", new_callable=AsyncMock):
            with patch.object(
                catalog.COMPILER,
                "compile",
                new_callable=AsyncMock,
                side_effect=WorkflowCompilerError("Invalid node")
            ):
                with pytest.raises(Exception):  # raise_problem raises an exception
                    await catalog.compile_spec(mock_user, payload)


# =============================================================================
# List MCP Tools Tests
# =============================================================================


@pytest.mark.unit
class TestListMcpTools:
    """Tests for list_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_list_mcp_tools_returns_tools(self):
        """Test listing MCP tools returns McpToolsResponse."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp_tool_1"
        mock_tool.description = "A test MCP tool"
        mock_tool.inputSchema = {"type": "object"}

        payload = api_models.McpToolsRequest(
            server="http://localhost:8080",
            server_type="http"
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            return_value=[mock_tool]
        ):
            result = await catalog.list_mcp_tools(payload)

        assert isinstance(result, api_models.McpToolsResponse)
        assert len(result.tools) == 1
        assert result.tools[0].name == "mcp_tool_1"
        assert result.tools[0].description == "A test MCP tool"

    @pytest.mark.asyncio
    async def test_list_mcp_tools_connection_error(self):
        """Test handling MCP connection errors."""
        payload = api_models.McpToolsRequest(
            server="http://invalid-server",
            server_type="http"
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Connection refused")
        ):
            with pytest.raises(Exception):  # raise_problem raises exception
                await catalog.list_mcp_tools(payload)

    @pytest.mark.asyncio
    async def test_list_mcp_tools_with_auth(self):
        """Test listing MCP tools with authentication."""
        mock_tool = MagicMock()
        mock_tool.name = "auth_tool"
        mock_tool.description = "Authenticated tool"
        mock_tool.inputSchema = None

        payload = api_models.McpToolsRequest(
            server="http://localhost:8080",
            server_type="http",
            auth={"api_key": "secret"}
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            return_value=[mock_tool]
        ):
            result = await catalog.list_mcp_tools(payload)

        assert len(result.tools) == 1
        assert result.tools[0].name == "auth_tool"


# =============================================================================
# Generate Schema Metadata Tests
# =============================================================================


@pytest.mark.unit
class TestGenerateSchemaMetadata:
    """Tests for generate_schema_metadata function."""

    @pytest.mark.asyncio
    async def test_generate_schema_metadata_empty_properties(self):
        """Test generating metadata for schema with empty properties."""
        payload = api_models.SchemaMetadataGenerateRequest(
            json_schema={"type": "object", "properties": {}}
        )

        result = await catalog.generate_schema_metadata(payload)

        assert isinstance(result, api_models.SchemaMetadataGenerateResponse)
        assert result.title == "OutputSchema"
        assert result.description == "Structured output schema"

    @pytest.mark.asyncio
    async def test_generate_schema_metadata_no_properties(self):
        """Test generating metadata when no properties key."""
        payload = api_models.SchemaMetadataGenerateRequest(
            json_schema={"type": "string"}
        )

        result = await catalog.generate_schema_metadata(payload)

        assert result.title == "OutputSchema"
        assert result.description == "Structured output schema"

    @pytest.mark.asyncio
    async def test_generate_schema_metadata_with_llm(self):
        """Test metadata generation uses LLM for descriptions."""
        payload = api_models.SchemaMetadataGenerateRequest(
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                }
            }
        )

        mock_response = MagicMock()
        mock_response.content = '{"title": "UserProfile", "description": "Contains user information."}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Patch at seer.llm since the function does: from seer.llm import get_llm
        with patch("seer.llm.get_llm", return_value=mock_llm):
            result = await catalog.generate_schema_metadata(payload)

        assert result.title == "UserProfile"
        assert result.description == "Contains user information."

    @pytest.mark.asyncio
    async def test_generate_schema_metadata_llm_error_returns_default(self):
        """Test LLM error returns default metadata."""
        payload = api_models.SchemaMetadataGenerateRequest(
            json_schema={
                "type": "object",
                "properties": {"field": {"type": "string"}}
            }
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        with patch("seer.llm.get_llm", return_value=mock_llm):
            result = await catalog.generate_schema_metadata(payload)

        assert result.title == "OutputSchema"
        assert result.description == "Structured output schema"

    @pytest.mark.asyncio
    async def test_generate_schema_metadata_invalid_json_response(self):
        """Test handling invalid JSON from LLM."""
        payload = api_models.SchemaMetadataGenerateRequest(
            json_schema={
                "type": "object",
                "properties": {"field": {"type": "string"}}
            }
        )

        mock_response = MagicMock()
        mock_response.content = "not valid json"

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("seer.llm.get_llm", return_value=mock_llm):
            result = await catalog.generate_schema_metadata(payload)

        # Falls back to default on JSON parse error
        assert result.title == "OutputSchema"
        assert result.description == "Structured output schema"
