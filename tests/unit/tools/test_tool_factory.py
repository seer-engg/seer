"""
Unit tests for the Tool Factory pattern — ToolDefinition, ToolSurface, UnifiedToolRegistry.

Verifies:
- ToolDefinition.to_langgraph_tool() produces correct StructuredTool with name/description/schema
- ToolDefinition.to_mcp_tool() produces correct FunctionTool with name/description/params
- Surface filtering: MCP-only tools don't appear in get_langgraph_tools()
- MCP tracking decorator is applied
- Idempotent registration
- register_unified_tools() registers all 6 tools
- Canonical impls with reasoning param
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from seer.tools.tool_factory import ToolDefinition, ToolSurface, UnifiedToolRegistry, unified_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _dummy_impl(query: str, reasoning: str = "", top_k: int = 5) -> str:
    """A dummy implementation for testing."""
    return json.dumps({"query": query, "reasoning": reasoning, "top_k": top_k})


async def _mcp_only_impl() -> str:
    """An MCP-only tool."""
    return json.dumps({"status": "ok"})


# ---------------------------------------------------------------------------
# ToolSurface tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolSurface:
    """Tests for ToolSurface Flag enum."""

    def test_both_includes_nexus(self):
        assert ToolSurface.NEXUS in ToolSurface.BOTH

    def test_both_includes_mcp(self):
        assert ToolSurface.MCP in ToolSurface.BOTH

    def test_nexus_not_in_mcp(self):
        assert ToolSurface.NEXUS not in ToolSurface.MCP

    def test_mcp_not_in_nexus(self):
        assert ToolSurface.MCP not in ToolSurface.NEXUS


# ---------------------------------------------------------------------------
# ToolDefinition tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_default_surface_is_both(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl)
        assert td.surface == ToolSurface.BOTH

    def test_resolve_mcp_name_defaults_to_name(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl)
        assert td.resolve_mcp_name() == "test"

    def test_resolve_mcp_name_override(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl, mcp_name="mcp_test")
        assert td.resolve_mcp_name() == "mcp_test"

    def test_resolve_nexus_name_defaults_to_name(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl)
        assert td._resolve_nexus_name() == "test"

    def test_resolve_nexus_name_override(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl, nexus_name="nexus_test")
        assert td._resolve_nexus_name() == "nexus_test"

    def test_resolve_mcp_tracking_name_defaults_to_mcp_name(self):
        td = ToolDefinition(name="test", description="desc", implementation=_dummy_impl, mcp_name="mcp_test")
        assert td._resolve_mcp_tracking_name() == "mcp_test"

    def test_resolve_mcp_tracking_name_override(self):
        td = ToolDefinition(
            name="test", description="desc", implementation=_dummy_impl,
            mcp_name="mcp_test", mcp_tracking_name="track_test"
        )
        assert td._resolve_mcp_tracking_name() == "track_test"


# ---------------------------------------------------------------------------
# to_langgraph_tool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToLanggraphTool:
    """Tests for ToolDefinition.to_langgraph_tool()."""

    def test_produces_structured_tool_with_correct_name(self):
        td = ToolDefinition(name="my_tool", description="My tool", implementation=_dummy_impl)
        tool = td.to_langgraph_tool()
        assert tool.name == "my_tool"

    def test_uses_nexus_name_override(self):
        td = ToolDefinition(
            name="my_tool", description="My tool", implementation=_dummy_impl,
            nexus_name="my_nexus_tool"
        )
        tool = td.to_langgraph_tool()
        assert tool.name == "my_nexus_tool"

    def test_uses_implementation_docstring_as_description(self):
        td = ToolDefinition(name="my_tool", description="Canonical desc", implementation=_dummy_impl)
        tool = td.to_langgraph_tool()
        # When no nexus_wrapper, uses implementation's docstring
        assert "dummy implementation" in tool.description

    def test_uses_canonical_description_when_no_docstring(self):
        async def no_doc_fn(query: str) -> str:
            return query

        td = ToolDefinition(name="my_tool", description="Canonical desc", implementation=no_doc_fn)
        tool = td.to_langgraph_tool()
        assert tool.description == "Canonical desc"

    def test_schema_includes_reasoning_param(self):
        """Implementation's reasoning param should appear in the tool's input schema."""
        td = ToolDefinition(name="my_tool", description="desc", implementation=_dummy_impl)
        tool = td.to_langgraph_tool()
        schema = tool.args_schema.model_json_schema()
        assert "reasoning" in schema.get("properties", {})

    @pytest.mark.asyncio
    async def test_langgraph_tool_is_callable(self):
        td = ToolDefinition(name="my_tool", description="desc", implementation=_dummy_impl)
        tool = td.to_langgraph_tool()
        result = await tool.ainvoke({"query": "test"})
        data = json.loads(result)
        assert data["query"] == "test"
        assert data["reasoning"] == ""
        assert data["top_k"] == 5


# ---------------------------------------------------------------------------
# to_mcp_tool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToMcpTool:
    """Tests for ToolDefinition.to_mcp_tool()."""

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_produces_function_tool_with_correct_name(self, mock_track):
        mock_track.return_value = lambda fn: fn  # passthrough decorator
        td = ToolDefinition(name="my_tool", description="My tool desc", implementation=_dummy_impl)
        tool = td.to_mcp_tool()
        assert tool.name == "my_tool"

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_uses_mcp_name_override(self, mock_track):
        mock_track.return_value = lambda fn: fn
        td = ToolDefinition(
            name="my_tool", description="desc", implementation=_dummy_impl,
            mcp_name="mcp_my_tool"
        )
        tool = td.to_mcp_tool()
        assert tool.name == "mcp_my_tool"

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_uses_canonical_description(self, mock_track):
        mock_track.return_value = lambda fn: fn
        td = ToolDefinition(name="my_tool", description="Canonical description here", implementation=_dummy_impl)
        tool = td.to_mcp_tool()
        assert tool.description == "Canonical description here"

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_tracking_decorator_applied_with_correct_name(self, mock_track):
        mock_track.return_value = lambda fn: fn
        td = ToolDefinition(
            name="my_tool", description="desc", implementation=_dummy_impl,
            mcp_tracking_name="tracked_name"
        )
        td.to_mcp_tool()
        mock_track.assert_called_once_with("tracked_name")

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_tracking_name_defaults_to_mcp_name(self, mock_track):
        mock_track.return_value = lambda fn: fn
        td = ToolDefinition(
            name="my_tool", description="desc", implementation=_dummy_impl,
            mcp_name="mcp_custom"
        )
        td.to_mcp_tool()
        mock_track.assert_called_once_with("mcp_custom")


# ---------------------------------------------------------------------------
# UnifiedToolRegistry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnifiedToolRegistry:
    """Tests for UnifiedToolRegistry."""

    def _make_registry(self) -> UnifiedToolRegistry:
        return UnifiedToolRegistry()

    def test_register_and_count(self):
        reg = self._make_registry()
        td = ToolDefinition(name="tool1", description="desc", implementation=_dummy_impl)
        reg.register(td)
        assert reg.tool_count == 1

    def test_idempotent_registration(self):
        """Re-registering the same name should not duplicate."""
        reg = self._make_registry()
        td1 = ToolDefinition(name="tool1", description="desc1", implementation=_dummy_impl)
        td2 = ToolDefinition(name="tool1", description="desc2", implementation=_dummy_impl)
        reg.register(td1)
        reg.register(td2)
        assert reg.tool_count == 1
        # First registration wins
        assert reg.get("tool1").description == "desc1"

    def test_get_returns_none_for_missing(self):
        reg = self._make_registry()
        assert reg.get("nonexistent") is None

    def test_clear(self):
        reg = self._make_registry()
        reg.register(ToolDefinition(name="t", description="d", implementation=_dummy_impl))
        reg.clear()
        assert reg.tool_count == 0

    def test_get_langgraph_tools_filters_by_surface(self):
        """MCP-only tools should not appear in get_langgraph_tools()."""
        reg = self._make_registry()
        reg.register(ToolDefinition(
            name="both_tool", description="d", implementation=_dummy_impl,
            surface=ToolSurface.BOTH,
        ))
        reg.register(ToolDefinition(
            name="mcp_only", description="d", implementation=_mcp_only_impl,
            surface=ToolSurface.MCP,
        ))
        reg.register(ToolDefinition(
            name="nexus_only", description="d", implementation=_dummy_impl,
            surface=ToolSurface.NEXUS,
        ))

        tools = reg.get_langgraph_tools()
        names = [t.name for t in tools]
        assert "both_tool" in names
        assert "nexus_only" in names
        assert "mcp_only" not in names

    @patch("seer.mcp.tracking.track_mcp_tool")
    def test_register_mcp_tools_filters_by_surface(self, mock_track):
        """NEXUS-only tools should not be registered on MCP."""
        mock_track.return_value = lambda fn: fn
        reg = self._make_registry()
        reg.register(ToolDefinition(
            name="both_tool", description="d", implementation=_dummy_impl,
            surface=ToolSurface.BOTH,
        ))
        reg.register(ToolDefinition(
            name="nexus_only", description="d", implementation=_dummy_impl,
            surface=ToolSurface.NEXUS,
        ))

        mock_mcp = MagicMock()
        reg.register_mcp_tools(mock_mcp)

        # Only "both_tool" should be added to MCP
        assert mock_mcp.add_tool.call_count == 1
        added_tool = mock_mcp.add_tool.call_args[0][0]
        assert added_tool.name == "both_tool"


# ---------------------------------------------------------------------------
# register_unified_tools() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterUnifiedTools:
    """Tests for the register_unified_tools() entrypoint."""

    def setup_method(self):
        """Reset registry state before each test."""
        import seer.tools.unified_tools as ut
        ut._REGISTERED = False
        unified_registry.clear()

    def teardown_method(self):
        """Reset registry state after each test."""
        import seer.tools.unified_tools as ut
        ut._REGISTERED = False
        unified_registry.clear()

    def test_registers_all_six_tools(self):
        from seer.tools.unified_tools import register_unified_tools
        register_unified_tools()
        assert unified_registry.tool_count == 6

    def test_expected_tool_names(self):
        from seer.tools.unified_tools import register_unified_tools
        register_unified_tools()
        expected = {"search_tools", "list_tools", "search_triggers", "list_triggers", "get_workflow_template", "list_workflow_templates"}
        actual = {td.name for td in unified_registry._tools.values()}
        assert actual == expected

    def test_idempotent_call(self):
        from seer.tools.unified_tools import register_unified_tools
        register_unified_tools()
        register_unified_tools()
        assert unified_registry.tool_count == 6

    def test_list_tools_nexus_name(self):
        from seer.tools.unified_tools import register_unified_tools
        register_unified_tools()
        td = unified_registry.get("list_tools")
        assert td._resolve_nexus_name() == "list_available_tools"

    def test_list_triggers_nexus_name(self):
        from seer.tools.unified_tools import register_unified_tools
        register_unified_tools()
        td = unified_registry.get("list_triggers")
        assert td._resolve_nexus_name() == "list_available_triggers"


# ---------------------------------------------------------------------------
# Canonical implementation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchToolsImpl:
    """Tests for the canonical search_tools_impl function."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_returns_valid_json(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_create_draft", "description": "Create a Gmail draft", "integration_type": "gmail", "parameters": {}},
        ]

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("create draft")
        data = json.loads(result)
        assert "query" in data
        assert data["query"] == "create draft"

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_no_results(self, mock_get_tools):
        mock_get_tools.return_value = []

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("nonexistent")
        data = json.loads(result)
        assert data["top_match"] is None

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_includes_resource_pickers(self, mock_get_tools):
        mock_get_tools.return_value = [
            {
                "name": "gmail_create_draft", "description": "Create a Gmail draft",
                "integration_type": "gmail", "parameters": {},
                "resource_pickers": {"to": {"type": "email"}},
            },
        ]

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("create draft")
        data = json.loads(result)
        assert "resource_pickers" in data["top_match"]

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_integration_uses_title_case(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_create_draft", "description": "Create a draft email", "integration_type": "gmail", "parameters": {}},
        ]

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("create draft")
        data = json.loads(result)
        assert data["top_match"]["integration"] == "Gmail"


@pytest.mark.unit
class TestSearchToolsReasoning:
    """Tests for reasoning param in search_tools_impl."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_reasoning_included_in_results(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_create_draft", "description": "Create a draft", "integration_type": "gmail", "parameters": {}},
        ]

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("create draft", reasoning="finding tools for email")
        data = json.loads(result)
        assert data["reasoning"] == "finding tools for email"

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_reasoning_included_in_no_results(self, mock_get_tools):
        mock_get_tools.return_value = []

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("nonexistent", reasoning="testing empty")
        data = json.loads(result)
        assert data["reasoning"] == "testing empty"
        assert data["top_match"] is None

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_reasoning_defaults_to_empty(self, mock_get_tools):
        mock_get_tools.return_value = []

        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("test")
        data = json.loads(result)
        assert data["reasoning"] == ""


@pytest.mark.unit
class TestSearchTriggersReasoning:
    """Tests for reasoning param in search_triggers_impl."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.trigger_registry")
    async def test_reasoning_included_in_results(self, mock_registry):
        mock_trigger = MagicMock()
        mock_trigger.key = "poll.gmail.email_received"
        mock_trigger.title = "Gmail Email Received"
        mock_trigger.provider = "gmail"
        mock_trigger.mode = "polling"
        mock_trigger.description = "Triggered when new email arrives"
        mock_trigger.schemas = MagicMock()
        mock_trigger.schemas.config = None
        mock_trigger.schemas.event = None
        mock_trigger.meta = MagicMock()
        mock_trigger.meta.sample_event = None
        mock_trigger.meta.requires_connection = True
        mock_registry.all.return_value = [mock_trigger]

        from seer.tools.unified_tools import search_triggers_impl
        result = await search_triggers_impl("gmail email", reasoning="need trigger for email")
        data = json.loads(result)
        assert data["reasoning"] == "need trigger for email"

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.trigger_registry")
    async def test_reasoning_defaults_to_empty(self, mock_registry):
        mock_registry.all.return_value = []

        from seer.tools.unified_tools import search_triggers_impl
        result = await search_triggers_impl("test")
        data = json.loads(result)
        assert data["reasoning"] == ""
        assert "triggers" in data
