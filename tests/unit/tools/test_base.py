"""
Unit tests for tools.base module.

Tests BaseTool class, tool registry, and helper functions.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
import inspect
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest


# =============================================================================
# _make_json_safe Tests
# =============================================================================


@pytest.mark.unit
class TestMakeJsonSafe:
    """Tests for _make_json_safe function."""

    def test_make_json_safe_primitives(self):
        """Test _make_json_safe passes through primitives unchanged."""
        from seer.tools.base import _make_json_safe

        assert _make_json_safe("string") == "string"
        assert _make_json_safe(42) == 42
        assert _make_json_safe(3.14) == 3.14
        assert _make_json_safe(True) is True
        assert _make_json_safe(False) is False
        assert _make_json_safe(None) is None

    def test_make_json_safe_dict(self):
        """Test _make_json_safe clones dictionaries."""
        from seer.tools.base import _make_json_safe

        original = {"key": "value", "nested": {"inner": 123}}
        result = _make_json_safe(original)

        assert result == original
        assert result is not original  # Different object
        assert result["nested"] is not original["nested"]  # Deep clone

    def test_make_json_safe_list(self):
        """Test _make_json_safe clones lists."""
        from seer.tools.base import _make_json_safe

        original = [1, 2, {"key": "value"}]
        result = _make_json_safe(original)

        assert result == original
        assert result is not original
        assert result[2] is not original[2]

    def test_make_json_safe_tuple(self):
        """Test _make_json_safe converts tuples to lists."""
        from seer.tools.base import _make_json_safe

        original = (1, 2, 3)
        result = _make_json_safe(original)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_make_json_safe_set(self):
        """Test _make_json_safe converts sets to lists."""
        from seer.tools.base import _make_json_safe

        original = {1, 2, 3}
        result = _make_json_safe(original)

        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_make_json_safe_nested_structure(self):
        """Test _make_json_safe handles deeply nested structures."""
        from seer.tools.base import _make_json_safe

        original = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"level4": "value"}]
                }
            }
        }
        result = _make_json_safe(original)

        assert result["level1"]["level2"]["level3"][2]["level4"] == "value"

    def test_make_json_safe_circular_reference_raises(self):
        """Test _make_json_safe raises on circular references."""
        from seer.tools.base import _make_json_safe

        circular = {"key": "value"}
        circular["self"] = circular  # Circular reference

        with pytest.raises(ValueError, match="Circular reference"):
            _make_json_safe(circular)

    def test_make_json_safe_non_json_types_passthrough(self):
        """Test _make_json_safe passes through unknown types."""
        from seer.tools.base import _make_json_safe

        class CustomObject:
            pass

        obj = CustomObject()
        result = _make_json_safe(obj)

        assert result is obj


# =============================================================================
# BaseTool Tests
# =============================================================================


@pytest.mark.unit
class TestBaseTool:
    """Tests for BaseTool abstract class."""

    @pytest.fixture
    def concrete_tool(self):
        """Create a concrete BaseTool implementation."""
        from seer.tools.base import BaseTool

        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool for unit testing"
            required_scopes = ["read", "write"]
            integration_type = "test"
            provider = "test_provider"

            async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
                return {"status": "executed", "token": access_token, "args": arguments}

            def get_parameters_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "First parameter"},
                        "param2": {"type": "integer"},
                    },
                    "required": ["param1"]
                }

            def get_output_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    }
                }

        return TestTool()

    def test_tool_basic_attributes(self, concrete_tool):
        """Test BaseTool basic attributes."""
        assert concrete_tool.name == "test_tool"
        assert concrete_tool.description == "A test tool for unit testing"
        assert concrete_tool.required_scopes == ["read", "write"]
        assert concrete_tool.integration_type == "test"
        assert concrete_tool.provider == "test_provider"

    def test_tool_default_attributes(self):
        """Test BaseTool default attributes."""
        from seer.tools.base import BaseTool

        class MinimalTool(BaseTool):
            name = "minimal"
            description = "Minimal tool"

            async def execute(self, access_token, arguments):
                return {}

        tool = MinimalTool()

        assert tool.required_scopes == []
        assert tool.integration_type is None
        assert tool.provider is None
        assert tool.required_secrets == []
        assert tool.default_resource is None

    def test_tool_get_parameters_schema_default(self):
        """Test BaseTool default parameters schema."""
        from seer.tools.base import BaseTool

        class ToolWithNoSchema(BaseTool):
            name = "no_schema"
            description = "Tool without custom schema"

            async def execute(self, access_token, arguments):
                return {}

        tool = ToolWithNoSchema()
        schema = tool.get_parameters_schema()

        assert schema == {"type": "object", "properties": {}, "required": []}

    def test_tool_get_resource_pickers_default(self, concrete_tool):
        """Test BaseTool default resource pickers."""
        pickers = concrete_tool.get_resource_pickers()
        assert pickers == {}

    def test_tool_get_metadata(self, concrete_tool):
        """Test BaseTool get_metadata returns complete metadata."""
        metadata = concrete_tool.get_metadata()

        assert metadata["name"] == "test_tool"
        assert metadata["description"] == "A test tool for unit testing"
        assert metadata["required_scopes"] == ["read", "write"]
        assert metadata["integration_type"] == "test"
        assert metadata["provider"] == "test_provider"
        assert "parameters" in metadata
        assert "output_schema" in metadata
        assert metadata["parameters"]["properties"]["param1"]["type"] == "string"

    def test_tool_get_metadata_with_resource_pickers(self):
        """Test BaseTool metadata includes resource picker configs."""
        from seer.tools.base import BaseTool

        class ToolWithPicker(BaseTool):
            name = "picker_tool"
            description = "Tool with resource picker"

            async def execute(self, access_token, arguments):
                return {}

            def get_parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string", "description": "File ID"}
                    }
                }

            def get_resource_pickers(self):
                return {
                    "file_id": {
                        "resource_type": "google_drive_file",
                        "display_field": "name",
                        "value_field": "id",
                    }
                }

        tool = ToolWithPicker()
        metadata = tool.get_metadata()

        assert "x-resource-picker" in metadata["parameters"]["properties"]["file_id"]
        assert metadata["resource_pickers"]["file_id"]["resource_type"] == "google_drive_file"

    @pytest.mark.asyncio
    async def test_tool_execute(self, concrete_tool):
        """Test BaseTool execute method."""
        result = await concrete_tool.execute(
            access_token="token_123",
            arguments={"param1": "value1", "param2": 42}
        )

        assert result["status"] == "executed"
        assert result["token"] == "token_123"
        assert result["args"]["param1"] == "value1"


# =============================================================================
# Tool Registry Tests
# =============================================================================


@pytest.mark.unit
class TestToolRegistry:
    """Tests for tool registry functions."""

    def test_register_tool(self):
        """Test registering a tool."""
        from seer.tools.base import BaseTool, register_tool, get_tool, clear_registry

        clear_registry()

        class MyTool(BaseTool):
            name = "my_tool"
            description = "My tool"

            async def execute(self, access_token, arguments):
                return {}

        tool = MyTool()
        register_tool(tool)

        result = get_tool("my_tool")
        assert result is tool

        clear_registry()

    def test_register_tool_idempotent(self):
        """Test registering same tool instance is idempotent."""
        from seer.tools.base import BaseTool, register_tool, list_tools, clear_registry

        clear_registry()

        class IdempotentTool(BaseTool):
            name = "idempotent"
            description = "Test"

            async def execute(self, access_token, arguments):
                return {}

        tool = IdempotentTool()
        register_tool(tool)
        register_tool(tool)  # Register again

        tools = list_tools()
        count = sum(1 for t in tools if t.name == "idempotent")
        assert count == 1

        clear_registry()

    def test_get_tool_not_found(self):
        """Test getting non-existent tool returns None."""
        from seer.tools.base import get_tool, clear_registry

        clear_registry()

        result = get_tool("nonexistent_tool")
        assert result is None

        clear_registry()

    def test_list_tools(self):
        """Test listing all registered tools."""
        from seer.tools.base import BaseTool, register_tool, list_tools, clear_registry

        clear_registry()

        class Tool1(BaseTool):
            name = "tool1"
            description = "Tool 1"

            async def execute(self, access_token, arguments):
                return {}

        class Tool2(BaseTool):
            name = "tool2"
            description = "Tool 2"

            async def execute(self, access_token, arguments):
                return {}

        register_tool(Tool1())
        register_tool(Tool2())

        tools = list_tools()
        names = [t.name for t in tools]

        assert "tool1" in names
        assert "tool2" in names

        clear_registry()

    def test_clear_registry(self):
        """Test clearing the registry."""
        from seer.tools.base import BaseTool, register_tool, list_tools, clear_registry

        class ClearTestTool(BaseTool):
            name = "clear_test"
            description = "Test"

            async def execute(self, access_token, arguments):
                return {}

        register_tool(ClearTestTool())
        assert len(list_tools()) > 0

        clear_registry()
        assert len(list_tools()) == 0


# =============================================================================
# Tool Signature Verification Tests
# =============================================================================


@pytest.mark.unit
class TestToolSignatureStandard:
    """Tests to verify all tools have the standardized execute() signature.

    This test class was added as part of the 2024-02 RCA to prevent future
    regressions where tools are added/modified without the correct signature.

    Required signature:
        async def execute(
            self,
            access_token: Optional[str],
            arguments: Dict[str, Any],
            *,
            credentials: Optional["ResolvedCredentials"] = None,
            context: Optional["WorkflowRuntimeContext"] = None,
        ) -> Any
    """

    @staticmethod
    def _load_all_tools():
        """Import all tool modules to trigger registration."""
        # pylint: disable=import-outside-toplevel,unused-import
        import seer.tools.google  # noqa: F401
        import seer.tools.slack  # noqa: F401
        import seer.tools.discord  # noqa: F401
        import seer.tools.knowledge  # noqa: F401
        import seer.tools.github  # noqa: F401
        import seer.tools.supabase  # noqa: F401

    def test_all_tools_have_credentials_parameter(self):
        """Verify all registered tools have 'credentials' as a keyword parameter."""
        from seer.tools.base import list_tools

        # Ensure tools are loaded
        self._load_all_tools()

        tools = list_tools()
        missing_credentials = []

        for tool in tools:
            sig = inspect.signature(tool.execute)
            params = list(sig.parameters.keys())

            if "credentials" not in params:
                missing_credentials.append(tool.name)

        if missing_credentials:
            pytest.fail(
                f"The following tools are missing 'credentials' parameter: {missing_credentials}"
            )

    def test_all_tools_have_context_parameter(self):
        """Verify all registered tools have 'context' as a keyword parameter."""
        from seer.tools.base import list_tools

        # Ensure tools are loaded
        self._load_all_tools()

        tools = list_tools()
        missing_context = []

        for tool in tools:
            sig = inspect.signature(tool.execute)
            params = list(sig.parameters.keys())

            if "context" not in params:
                missing_context.append(tool.name)

        if missing_context:
            pytest.fail(
                f"The following tools are missing 'context' parameter: {missing_context}"
            )

    def test_all_tools_credentials_is_keyword_only(self):
        """Verify 'credentials' parameter is keyword-only (after *)."""
        from seer.tools.base import list_tools

        # Ensure tools are loaded
        self._load_all_tools()

        tools = list_tools()
        not_keyword_only = []

        for tool in tools:
            sig = inspect.signature(tool.execute)
            param = sig.parameters.get("credentials")

            if param and param.kind != inspect.Parameter.KEYWORD_ONLY:
                not_keyword_only.append(tool.name)

        if not_keyword_only:
            pytest.fail(
                f"The following tools have 'credentials' as positional (should be keyword-only after *): {not_keyword_only}"
            )
