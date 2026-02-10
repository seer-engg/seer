# pylint: disable=too-many-lines
# Reason: Comprehensive test coverage for tool registry requires many test cases
"""
Unit tests for tool registry and base tool functionality.

Tests tool registration, retrieval, metadata generation, and resource pickers.
Target coverage: 90%+
"""
from typing import Any, Dict, Optional

import pytest

from seer.tools.base import (
    BaseTool,
    ResourcePickerConfig,
    _make_json_safe,
    register_tool,
    get_tool,
    list_tools,
    clear_registry,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test Tool Implementations
# =============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    required_scopes = ["scope1", "scope2"]
    integration_type = "test_integration"
    provider = "test_provider"

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        return {"status": "success", "data": arguments}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "number", "description": "Second parameter"}
            },
            "required": ["param1"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "object"}
            }
        }


class MockToolWithResourcePicker(BaseTool):
    """Mock tool with resource picker."""

    name = "mock_tool_with_picker"
    description = "A mock tool with resource picker"
    required_scopes = []
    integration_type = "googledrive"
    provider = "google"

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        return {"file_id": arguments.get("file_id")}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Google Drive file ID"
                }
            },
            "required": ["file_id"]
        }

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "filter": {},
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True
            }
        }


class MinimalTool(BaseTool):
    """Minimal tool with defaults."""

    name = "minimal_tool"
    description = "A minimal tool"

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        return "minimal_result"


# =============================================================================
# Registry Management Tests
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_register_tool():
    """Test registering a tool."""
    tool = MockTool()
    register_tool(tool)

    assert get_tool("mock_tool") is tool
    assert "mock_tool" in [t.name for t in list_tools()]


def test_register_tool_idempotent():
    """Test that registering the same tool instance twice is idempotent."""
    tool = MockTool()
    register_tool(tool)
    register_tool(tool)  # Second registration

    assert len(list_tools()) == 1
    assert get_tool("mock_tool") is tool


def test_register_multiple_tools():
    """Test registering multiple tools."""
    tool1 = MockTool()
    tool2 = MinimalTool()

    register_tool(tool1)
    register_tool(tool2)

    assert len(list_tools()) == 2
    assert get_tool("mock_tool") is tool1
    assert get_tool("minimal_tool") is tool2


def test_get_tool_not_found():
    """Test getting non-existent tool returns None."""
    assert get_tool("nonexistent_tool") is None


def test_list_tools_empty():
    """Test listing tools when registry is empty."""
    assert not list_tools()


def test_list_tools_returns_copies():
    """Test that list_tools returns a list of registered tools."""
    tool1 = MockTool()
    tool2 = MinimalTool()

    register_tool(tool1)
    register_tool(tool2)

    tools = list_tools()
    assert len(tools) == 2
    assert tool1 in tools
    assert tool2 in tools


def test_clear_registry():
    """Test clearing the registry."""
    register_tool(MockTool())
    register_tool(MinimalTool())

    assert len(list_tools()) == 2

    clear_registry()

    assert len(list_tools()) == 0
    assert get_tool("mock_tool") is None


# =============================================================================
# BaseTool Metadata Tests
# =============================================================================


def test_get_metadata_full():
    """Test getting full metadata from tool."""
    tool = MockTool()
    metadata = tool.get_metadata()

    assert metadata["name"] == "mock_tool"
    assert metadata["description"] == "A mock tool for testing"
    assert metadata["required_scopes"] == ["scope1", "scope2"]
    assert metadata["integration_type"] == "test_integration"
    assert metadata["provider"] == "test_provider"
    assert not metadata["required_secrets"]
    assert "parameters" in metadata
    assert "output_schema" in metadata


def test_get_metadata_parameters_schema():
    """Test that metadata includes parameters schema."""
    tool = MockTool()
    metadata = tool.get_metadata()

    params = metadata["parameters"]
    assert params["type"] == "object"
    assert "param1" in params["properties"]
    assert "param2" in params["properties"]
    assert params["required"] == ["param1"]


def test_get_metadata_output_schema():
    """Test that metadata includes output schema."""
    tool = MockTool()
    metadata = tool.get_metadata()

    output = metadata["output_schema"]
    assert output["type"] == "object"
    assert "status" in output["properties"]
    assert "data" in output["properties"]


def test_get_metadata_with_resource_pickers():
    """Test that metadata includes resource picker configs."""
    tool = MockToolWithResourcePicker()
    metadata = tool.get_metadata()

    # Resource pickers should be in the metadata
    assert "resource_pickers" in metadata
    assert "file_id" in metadata["resource_pickers"]

    # x-resource-picker should be injected into schema
    assert "x-resource-picker" in metadata["parameters"]["properties"]["file_id"]
    picker = metadata["parameters"]["properties"]["file_id"]["x-resource-picker"]
    assert picker["resource_type"] == "google_drive_file"
    assert picker["display_field"] == "name"
    assert picker["search_enabled"] is True


def test_get_metadata_minimal_tool():
    """Test metadata for minimal tool with defaults."""
    tool = MinimalTool()
    metadata = tool.get_metadata()

    assert metadata["name"] == "minimal_tool"
    assert metadata["description"] == "A minimal tool"
    assert not metadata["required_scopes"]
    assert metadata["integration_type"] is None
    assert metadata["provider"] is None
    assert not metadata["required_secrets"]


# =============================================================================
# Default Schema Tests
# =============================================================================


def test_default_parameters_schema():
    """Test default parameters schema."""
    tool = MinimalTool()
    schema = tool.get_parameters_schema()

    assert schema["type"] == "object"
    assert not schema["properties"]
    assert not schema["required"]


def test_default_output_schema():
    """Test default output schema."""
    tool = MinimalTool()
    schema = tool.get_output_schema()

    assert schema["type"] == "object"
    assert not schema["properties"]
    assert not schema["required"]


def test_default_resource_pickers():
    """Test default resource pickers is empty dict."""
    tool = MinimalTool()
    pickers = tool.get_resource_pickers()

    assert not pickers


# =============================================================================
# JSON Safety Tests
# =============================================================================


def test_make_json_safe_primitives():
    """Test _make_json_safe with primitive values."""
    assert _make_json_safe("string") == "string"
    assert _make_json_safe(42) == 42
    assert _make_json_safe(3.14) == 3.14
    assert _make_json_safe(True) is True
    assert _make_json_safe(None) is None


def test_make_json_safe_dict():
    """Test _make_json_safe with dict."""
    input_dict = {"key": "value", "nested": {"inner": 123}}
    result = _make_json_safe(input_dict)

    assert result == {"key": "value", "nested": {"inner": 123}}
    assert id(result) != id(input_dict)  # Should be a copy


def test_make_json_safe_list():
    """Test _make_json_safe with list."""
    input_list = [1, 2, {"key": "value"}]
    result = _make_json_safe(input_list)

    assert result == [1, 2, {"key": "value"}]
    assert id(result) != id(input_list)  # Should be a copy


def test_make_json_safe_tuple():
    """Test _make_json_safe with tuple (converts to list)."""
    input_tuple = (1, 2, 3)
    result = _make_json_safe(input_tuple)

    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_make_json_safe_set():
    """Test _make_json_safe with set (converts to list)."""
    input_set = {1, 2, 3}
    result = _make_json_safe(input_set)

    assert isinstance(result, list)
    assert set(result) == {1, 2, 3}


def test_make_json_safe_circular_reference():
    """Test _make_json_safe detects circular references."""
    circular_dict: Dict[str, Any] = {"key": "value"}
    circular_dict["self"] = circular_dict

    with pytest.raises(ValueError, match="Circular reference detected"):
        _make_json_safe(circular_dict)


def test_make_json_safe_nested_structure():
    """Test _make_json_safe with deeply nested structure."""
    nested = {
        "level1": {
            "level2": {
                "level3": [1, 2, {"level4": "deep"}]
            }
        }
    }
    result = _make_json_safe(nested)

    assert result == nested
    assert id(result) != id(nested)


def test_make_json_safe_mixed_types():
    """Test _make_json_safe with mixed types."""
    mixed = {
        "string": "text",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"}
    }
    result = _make_json_safe(mixed)

    assert result == mixed


# =============================================================================
# Tool Execution Interface Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_execute_basic():
    """Test basic tool execution."""
    tool = MockTool()
    result = await tool.execute(
        access_token="test_token",
        arguments={"param1": "value1", "param2": 42}
    )

    assert result["status"] == "success"
    assert result["data"]["param1"] == "value1"
    assert result["data"]["param2"] == 42


@pytest.mark.asyncio
async def test_tool_execute_no_token():
    """Test tool execution without access token."""
    tool = MinimalTool()
    result = await tool.execute(access_token=None, arguments={})

    assert result == "minimal_result"


@pytest.mark.asyncio
async def test_tool_execute_empty_arguments():
    """Test tool execution with empty arguments."""
    tool = MockTool()
    result = await tool.execute(access_token="token", arguments={})

    assert result["status"] == "success"
    assert result["data"] == {}


# =============================================================================
# Tool Attributes Tests
# =============================================================================


def test_tool_required_scopes():
    """Test tool required_scopes attribute."""
    tool = MockTool()
    assert tool.required_scopes == ["scope1", "scope2"]


def test_tool_integration_type():
    """Test tool integration_type attribute."""
    tool = MockTool()
    assert tool.integration_type == "test_integration"


def test_tool_provider():
    """Test tool provider attribute."""
    tool = MockTool()
    assert tool.provider == "test_provider"


def test_tool_default_required_secrets():
    """Test tool default required_secrets is empty list."""
    tool = MinimalTool()
    assert not tool.required_secrets


def test_tool_default_resource_requirement():
    """Test tool default_resource is None by default."""
    tool = MinimalTool()
    assert tool.default_resource is None


# =============================================================================
# Resource Picker Configuration Tests
# =============================================================================


def test_resource_picker_config_structure():
    """Test resource picker config structure."""
    tool = MockToolWithResourcePicker()
    pickers = tool.get_resource_pickers()

    picker = pickers["file_id"]
    assert picker["resource_type"] == "google_drive_file"
    assert picker["display_field"] == "name"
    assert picker["value_field"] == "id"
    assert picker["search_enabled"] is True
    assert picker["hierarchy"] is True


def test_resource_picker_injected_into_schema():
    """Test that resource picker is injected into parameter schema."""
    tool = MockToolWithResourcePicker()
    metadata = tool.get_metadata()

    file_id_prop = metadata["parameters"]["properties"]["file_id"]
    assert "x-resource-picker" in file_id_prop

    picker = file_id_prop["x-resource-picker"]
    assert picker["resource_type"] == "google_drive_file"


def test_resource_picker_not_injected_for_missing_param():
    """Test that resource picker is not injected if parameter doesn't exist."""

    class ToolWithMismatchedPicker(BaseTool):
        name = "mismatched"
        description = "Tool with mismatched picker"

        async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
            return {}

        def get_parameters_schema(self) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "actual_param": {"type": "string"}
                }
            }

        def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
            return {
                "nonexistent_param": {
                    "resource_type": "some_type",
                    "display_field": "name",
                    "value_field": "id"
                }
            }

    tool = ToolWithMismatchedPicker()
    metadata = tool.get_metadata()

    # Picker should not be injected since parameter doesn't exist
    assert "x-resource-picker" not in metadata["parameters"]["properties"].get("actual_param", {})


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_tool_name_uniqueness_enforcement():
    """Test that registering different tool instances with same name is handled."""

    class Tool1(BaseTool):
        name = "duplicate_name"
        description = "First tool"

        async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
            return "tool1"

    class Tool2(BaseTool):
        name = "duplicate_name"
        description = "Second tool"

        async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
            return "tool2"

    tool1 = Tool1()
    tool2 = Tool2()

    register_tool(tool1)
    register_tool(tool2)  # Should be skipped with debug log

    # First tool should remain registered
    assert get_tool("duplicate_name") is tool1
    assert len(list_tools()) == 1


def test_tool_with_empty_scopes():
    """Test tool with empty required_scopes."""
    tool = MinimalTool()
    assert not tool.required_scopes
    metadata = tool.get_metadata()
    assert not metadata["required_scopes"]


def test_tool_metadata_required_secrets_as_list():
    """Test that required_secrets is always returned as a list in metadata."""

    class ToolWithSecrets(BaseTool):
        name = "secrets_tool"
        description = "Tool with secrets"
        required_secrets = ["SECRET_KEY"]

        async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
            return {}

    tool = ToolWithSecrets()
    metadata = tool.get_metadata()

    assert isinstance(metadata["required_secrets"], list)
    assert metadata["required_secrets"] == ["SECRET_KEY"]


def test_tool_metadata_default_resource():
    """Test that default_resource is included in metadata."""

    class ToolWithDefaultResource(BaseTool):
        name = "resource_tool"
        description = "Tool with default resource"
        default_resource = {
            "resource_type": "google_drive_file",
            "provider": "google",
            "required": True
        }

        async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
            return {}

    tool = ToolWithDefaultResource()
    metadata = tool.get_metadata()

    assert metadata["default_resource"] == {
        "resource_type": "google_drive_file",
        "provider": "google",
        "required": True
    }


@pytest.mark.parametrize("tool_class,expected_name", [
    (MockTool, "mock_tool"),
    (MinimalTool, "minimal_tool"),
    (MockToolWithResourcePicker, "mock_tool_with_picker"),
])
def test_tool_names(tool_class, expected_name):
    """Test various tool names."""
    tool = tool_class()
    assert tool.name == expected_name
