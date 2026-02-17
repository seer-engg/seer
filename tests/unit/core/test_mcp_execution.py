"""
Integration tests for MCP node execution.

Tests full workflow compilation and execution with MCP nodes.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment_async
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


async def _compile_workflow_with_mcp(spec_payload: dict) -> CompiledWorkflow:
    """Compile a workflow with MCP support."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    model_registry = ModelRegistry()
    mcp_client_registry = MCPClientRegistry()

    spec = parse_workflow_spec(spec_payload)
    type_env = await build_type_environment_async(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        mcp_client_registry=mcp_client_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)

    runtime = NodeRuntime(
        RuntimeServices(
            schema_registry=schema_registry,
            tool_registry=tool_registry,
            model_registry=model_registry,
            type_env=type_env,
            mcp_client_registry=mcp_client_registry,
        )
    )
    graph = await emit_langgraph(plan, runtime)
    return CompiledWorkflow(
        spec=spec,
        type_env=type_env.as_dict(),
        graph=graph,
        runtime=runtime,
    )


@pytest.mark.asyncio
async def test_mcp_node_basic_execution() -> None:
    """Test basic MCP node execution with mocked MCP server."""

    # Mock MCP tool
    mock_tool = MagicMock()
    mock_tool.name = "search_database"
    mock_tool.description = "Search the database"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }

    # Mock MCP call result
    mock_content = MagicMock()
    mock_content.text = '{"results": ["item1", "item2"], "count": 2}'
    mock_result = MagicMock()
    mock_result.content = [mock_content]

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {"search_term": {"type": "string"}},
                        }
                    },
                },
                "meta": {"sample_event": {"data": {"search_term": "test"}}},
            }
        ],
        "nodes": [
            {
                "id": "mcp_search",
                "type": "mcp",
                "server": "http://localhost:8080/mcp",
                "server_type": "http",
                "tool": "search_database",
                "inputs": {"query": "test query"},  # Use literal value to avoid trigger binding issues
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "mcp_search", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate, patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.invoke_tool"
    ) as mock_invoke:

        # Setup mocks
        mock_validate.return_value = {
            "name": "search_database",
            "description": "Search the database",
            "input_schema": mock_tool.inputSchema,
        }
        # Return Python dict, not JSON string
        mock_invoke.return_value = {"results": ["item1", "item2"], "count": 2}

        # Compile workflow
        workflow = await _compile_workflow_with_mcp(spec)

        # Execute workflow
        trigger_event = {}  # No trigger data needed since we use literal values
        result = await workflow.ainvoke(trigger=trigger_event)

        # Verify MCP tool was called
        mock_validate.assert_called_once()
        mock_invoke.assert_called_once()

        # Verify inputs were passed correctly
        call_args = mock_invoke.call_args
        assert call_args[0][2]["query"] == "test query"  # arguments parameter

        # Verify result
        assert "mcp_search" in result
        assert result["mcp_search"] == {"results": ["item1", "item2"], "count": 2}


@pytest.mark.asyncio
async def test_mcp_node_with_auth_resolution() -> None:
    """Test MCP node with authentication headers."""

    mock_tool = MagicMock()
    mock_tool.name = "secure_api"
    mock_tool.description = "Secure API"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object", "properties": {}},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "secure_call",
                "type": "mcp",
                "server": "https://api.example.com/mcp",
                "server_type": "http",
                "tool": "secure_api",
                "auth": {
                    "headers": {"Authorization": "Bearer test_token"}
                },
                "inputs": {},
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "secure_call", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate, patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.invoke_tool"
    ) as mock_invoke:

        mock_validate.return_value = {
            "name": "secure_api",
            "description": "Secure API",
            "input_schema": {},
        }
        mock_invoke.return_value = {"status": "success"}

        workflow = await _compile_workflow_with_mcp(spec)

        # Execute workflow
        result = await workflow.ainvoke(trigger={})

        # Verify the auth was passed to the registry
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        server_config = call_args[0][0]

        # Auth should be present
        assert server_config.auth is not None
        assert "headers" in server_config.auth
        assert server_config.auth["headers"]["Authorization"] == "Bearer test_token"


@pytest.mark.asyncio
async def test_mcp_node_stdio_type() -> None:
    """Test MCP node with stdio server type."""

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object"},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "stdio_node",
                "type": "mcp",
                "server": "npx @modelcontextprotocol/server-everything",
                "server_type": "stdio",
                "tool": "list_files",
                "inputs": {"path": "/tmp"},
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "stdio_node", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate, patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.invoke_tool"
    ) as mock_invoke:

        mock_validate.return_value = {
            "name": "list_files",
            "description": "List files",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
        # Return object instead of array to match default schema expectation
        mock_invoke.return_value = {"files": ["file1.txt", "file2.txt"]}

        workflow = await _compile_workflow_with_mcp(spec)
        result = await workflow.ainvoke(trigger={})

        # Verify stdio server type was used
        call_args = mock_invoke.call_args
        server_config = call_args[0][0]
        assert server_config.server_type == "stdio"
        assert "npx" in server_config.server


@pytest.mark.asyncio
async def test_mcp_node_with_output_validation() -> None:
    """Test MCP node with expect_outputs validation."""

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object"},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "validated_node",
                "type": "mcp",
                "server": "http://localhost:8080/mcp",
                "server_type": "http",
                "tool": "get_user",
                "inputs": {"user_id": "123"},
                "expect_outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                            "required": ["id", "name"],
                        }
                    },
                },
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "validated_node", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate, patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.invoke_tool"
    ) as mock_invoke:

        mock_validate.return_value = {
            "name": "get_user",
            "description": "Get user",
            "input_schema": {},
        }
        # Return valid data matching the schema
        mock_invoke.return_value = {
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
        }

        workflow = await _compile_workflow_with_mcp(spec)
        result = await workflow.ainvoke(trigger={})

        # Should execute successfully with valid data
        assert "validated_node" in result
        assert result["validated_node"]["id"] == "123"


@pytest.mark.asyncio
async def test_mcp_node_error_handling() -> None:
    """Test MCP node error handling when server is unreachable."""

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object"},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "failing_node",
                "type": "mcp",
                "server": "http://invalid-server:9999/mcp",
                "server_type": "http",
                "tool": "test_tool",
                "inputs": {},
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "failing_node", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate:
        # Simulate connection error during compilation
        mock_validate.side_effect = ConnectionError("Connection refused")

        # Should fail during compilation with helpful error
        with pytest.raises(Exception) as exc_info:
            await _compile_workflow_with_mcp(spec)

        assert "Connection" in str(exc_info.value) or "MCP" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_node_tool_not_found() -> None:
    """Test MCP node error when tool doesn't exist on server."""

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object"},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "invalid_tool_node",
                "type": "mcp",
                "server": "http://localhost:8080/mcp",
                "server_type": "http",
                "tool": "nonexistent_tool",
                "inputs": {},
            }
        ],
        "edges": [
            {"source": "manual_trigger", "target": "invalid_tool_node", "type": "trigger"}
        ],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate:
        # Simulate tool not found error
        mock_validate.side_effect = ValueError(
            "Tool 'nonexistent_tool' not found on MCP server. Available tools: search, get_user"
        )

        # Should fail during compilation
        with pytest.raises(Exception) as exc_info:
            await _compile_workflow_with_mcp(spec)

        error_msg = str(exc_info.value)
        assert "nonexistent_tool" in error_msg or "not found" in error_msg.lower()


@pytest.mark.asyncio
async def test_mcp_node_trace_data() -> None:
    """Test that MCP node stores trace data correctly."""

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "manual_trigger",
                "key": "manual.trigger",
                "mode": "manual",
                "event_schema": {"type": "object"},
                "meta": {},
            }
        ],
        "nodes": [
            {
                "id": "traced_node",
                "type": "mcp",
                "server": "http://localhost:8080/mcp",
                "server_type": "http",
                "tool": "test_tool",
                "auth": {"headers": {"Authorization": "Bearer secret"}},
                "inputs": {"param": "value"},
            }
        ],
        "edges": [{"source": "manual_trigger", "target": "traced_node", "type": "trigger"}],
    }

    with patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.validate_tool"
    ) as mock_validate, patch(
        "seer.core.registry.mcp_client_registry.MCPClientRegistry.invoke_tool"
    ) as mock_invoke:

        mock_validate.return_value = {
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {},
        }
        mock_invoke.return_value = {"result": "success"}

        workflow = await _compile_workflow_with_mcp(spec)
        result = await workflow.ainvoke(trigger={})

        # Check trace data is stored
        trace_key = "_trace_traced_node"
        assert trace_key in result
        trace_data = result[trace_key]

        assert trace_data["node_id"] == "traced_node"
        assert trace_data["node_type"] == "mcp"
        assert trace_data["server"] == "http://localhost:8080/mcp"
        assert trace_data["tool"] == "test_tool"
        assert trace_data["inputs"] == {"param": "value"}
        assert trace_data["output"] == {"result": "success"}

        # Auth should be redacted in trace
        assert trace_data["auth"] is not None
        assert trace_data["auth"]["headers"]["Authorization"] == "***REDACTED***"
