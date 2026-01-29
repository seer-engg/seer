"""
Unit tests for MCP node functionality.

Tests schema validation, registry operations, and node configuration.
"""
import pytest
from pydantic import ValidationError

from seer.core.schema.models import MCPNode, OutputContract, OutputMode


# =============================================================================
# Schema Validation Tests
# =============================================================================


def test_mcp_node_basic():
    """Test basic MCP node creation."""
    node = MCPNode(
        id="test_node",
        server="http://localhost:8080/mcp",
        server_type="http",
        tool="test_tool",
    )

    assert node.id == "test_node"
    assert node.type == "mcp"
    assert node.server == "http://localhost:8080/mcp"
    assert node.server_type == "http"
    assert node.tool == "test_tool"
    assert node.auth is None
    assert node.inputs == {}
    assert node.expect_outputs is None


def test_mcp_node_with_auth_headers():
    """Test MCP node with authentication headers."""
    node = MCPNode(
        id="auth_node",
        server="https://api.example.com/mcp",
        server_type="http",
        tool="secure_tool",
        auth={
            "headers": {
                "Authorization": "Bearer ${secrets.api_token}",
                "X-API-Key": "${secrets.api_key}",
            }
        },
    )

    assert node.auth is not None
    assert "headers" in node.auth
    assert node.auth["headers"]["Authorization"] == "Bearer ${secrets.api_token}"
    assert node.auth["headers"]["X-API-Key"] == "${secrets.api_key}"


def test_mcp_node_with_auth_env():
    """Test MCP node with environment variable auth (stdio)."""
    node = MCPNode(
        id="stdio_node",
        server="npx mcp-server@latest",
        server_type="stdio",
        tool="list_files",
        auth={"env": {"MCP_API_KEY": "${secrets.mcp_key}"}},
    )

    assert node.server_type == "stdio"
    assert node.auth is not None
    assert "env" in node.auth
    assert node.auth["env"]["MCP_API_KEY"] == "${secrets.mcp_key}"


def test_mcp_node_with_inputs():
    """Test MCP node with input parameters."""
    node = MCPNode(
        id="search_node",
        server="http://localhost:8080/mcp",
        server_type="http",
        tool="search",
        inputs={"query": "${trigger.data.search_query}", "limit": 10, "offset": 0},
    )

    assert len(node.inputs) == 3
    assert node.inputs["query"] == "${trigger.data.search_query}"
    assert node.inputs["limit"] == 10
    assert node.inputs["offset"] == 0


def test_mcp_node_with_expect_outputs():
    """Test MCP node with expected output contract."""
    from seer.core.schema.models import InlineSchema

    node = MCPNode(
        id="structured_node",
        server="http://localhost:8080/mcp",
        server_type="http",
        tool="get_data",
        expect_outputs=OutputContract(
            mode=OutputMode.json,
            schema=InlineSchema(
                schema={
                    "type": "object",
                    "properties": {"results": {"type": "array"}, "count": {"type": "integer"}},
                    "required": ["results"],
                }
            ),
        ),
    )

    assert node.expect_outputs is not None
    assert node.expect_outputs.mode == OutputMode.json
    # Access the nested schema through the InlineSchema wrapper
    schema_dict = node.expect_outputs.schema.json_schema
    assert "results" in schema_dict["properties"]

# =============================================================================
# not testing this as from frontend we get empty strings initaly in autosave
# =============================================================================
# def test_mcp_node_missing_server():
#     """Test that MCP node requires server field."""
#     with pytest.raises(ValidationError) as exc_info:
#         MCPNode(
#             id="bad_node",
#             server="",  # Empty server
#             server_type="http",
#             tool="test_tool",
#         )

#     errors = exc_info.value.errors()
#     assert any(
#         err["loc"] == ("server",) and "at least 1 character" in str(err["msg"]).lower()
#         for err in errors
#     )


# def test_mcp_node_missing_tool():
#     """Test that MCP node requires tool field."""
#     with pytest.raises(ValidationError) as exc_info:
#         MCPNode(
#             id="bad_node",
#             server="http://localhost:8080/mcp",
#             server_type="http",
#             tool="",  # Empty tool
#         )

#     errors = exc_info.value.errors()
#     assert any(
#         err["loc"] == ("tool",) and "at least 1 character" in str(err["msg"]).lower()
#         for err in errors
#     )


def test_mcp_node_invalid_server_type():
    """Test that MCP node validates server_type."""
    with pytest.raises(ValidationError) as exc_info:
        MCPNode(
            id="bad_node",
            server="http://localhost:8080/mcp",
            server_type="invalid_type",  # Should be "http" or "stdio"
            tool="test_tool",
        )

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("server_type",) and "literal_error" in err["type"]
        for err in errors
    )


def test_mcp_node_default_server_type():
    """Test that MCP node defaults to http server type."""
    node = MCPNode(
        id="default_node",
        server="http://localhost:8080/mcp",
        tool="test_tool",
    )

    assert node.server_type == "http"


def test_mcp_node_http_server():
    """Test HTTP MCP server configuration."""
    node = MCPNode(
        id="http_node",
        server="https://mcp.example.com/api",
        server_type="http",
        tool="get_resource",
        auth={"headers": {"Authorization": "Bearer token"}},
    )

    assert node.server_type == "http"
    assert node.server.startswith("https://")


def test_mcp_node_stdio_server():
    """Test stdio MCP server configuration."""
    node = MCPNode(
        id="stdio_node",
        server="npx @modelcontextprotocol/server-everything",
        server_type="stdio",
        tool="execute",
    )

    assert node.server_type == "stdio"
    assert "npx" in node.server


def test_mcp_node_complex_auth():
    """Test MCP node with both headers and env auth."""
    node = MCPNode(
        id="complex_auth_node",
        server="http://localhost:8080/mcp",
        server_type="http",
        tool="complex_tool",
        auth={
            "headers": {
                "Authorization": "Bearer ${secrets.token}",
                "X-Custom-Header": "value",
            },
            "env": {"API_KEY": "${secrets.env_key}"},
        },
    )

    assert "headers" in node.auth
    assert "env" in node.auth
    assert len(node.auth["headers"]) == 2
    assert len(node.auth["env"]) == 1


def test_mcp_node_ui_metadata():
    """Test that MCP node preserves UI metadata."""
    node = MCPNode(
        id="ui_node",
        server="http://localhost:8080/mcp",
        tool="test_tool",
        ui={"x": 100, "y": 200, "color": "blue"},
    )

    assert node.ui is not None
    assert node.ui["x"] == 100
    assert node.ui["y"] == 200
    assert node.ui["color"] == "blue"


# =============================================================================
# Edge Cases
# =============================================================================


def test_mcp_node_special_characters_in_server():
    """Test MCP node with special characters in server URL."""
    node = MCPNode(
        id="special_node",
        server="http://localhost:8080/mcp?key=value&foo=bar",
        tool="test_tool",
    )

    assert "?" in node.server
    assert "&" in node.server


def test_mcp_node_auth_with_empty_dicts():
    """Test MCP node with empty auth dictionaries."""
    node = MCPNode(
        id="empty_auth_node",
        server="http://localhost:8080/mcp",
        tool="test_tool",
        auth={"headers": {}, "env": {}},
    )

    assert node.auth is not None
    assert node.auth["headers"] == {}
    assert node.auth["env"] == {}


def test_mcp_node_with_nested_input_expressions():
    """Test MCP node with nested input expressions."""
    node = MCPNode(
        id="nested_node",
        server="http://localhost:8080/mcp",
        tool="process",
        inputs={
            "data": {
                "user": "${trigger.data.user}",
                "filters": {"status": "active", "limit": "${config.max_results}"},
            }
        },
    )

    assert isinstance(node.inputs["data"], dict)
    assert node.inputs["data"]["user"] == "${trigger.data.user}"
    assert isinstance(node.inputs["data"]["filters"], dict)


def test_mcp_node_serialization():
    """Test that MCP node can be serialized to dict."""
    node = MCPNode(
        id="serialize_node",
        server="http://localhost:8080/mcp",
        server_type="http",
        tool="test_tool",
        inputs={"param": "value"},
        auth={"headers": {"Authorization": "Bearer token"}},
    )

    data = node.model_dump()

    assert data["id"] == "serialize_node"
    assert data["type"] == "mcp"
    assert data["server"] == "http://localhost:8080/mcp"
    assert data["server_type"] == "http"
    assert data["tool"] == "test_tool"
    assert data["inputs"] == {"param": "value"}
    assert data["auth"] == {"headers": {"Authorization": "Bearer token"}}


def test_mcp_node_from_dict():
    """Test creating MCP node from dictionary."""
    data = {
        "id": "from_dict_node",
        "type": "mcp",
        "server": "http://localhost:8080/mcp",
        "server_type": "stdio",
        "tool": "execute",
        "inputs": {"command": "ls"},
        "auth": {"env": {"PATH": "/usr/bin"}},
    }

    node = MCPNode(**data)

    assert node.id == "from_dict_node"
    assert node.server_type == "stdio"
    assert node.tool == "execute"
    assert node.inputs["command"] == "ls"
    assert node.auth["env"]["PATH"] == "/usr/bin"
