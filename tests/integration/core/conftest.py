# pylint: disable=redefined-outer-name
# Reason: pytest fixtures intentionally shadow outer names for dependency injection
"""
Core-specific fixtures for integration tests.

These fixtures provide real registries and compilation helpers for testing
the workflow compiler and runtime in an integrated manner.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from seer.core.compiler.context import CompilerContext
from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry


# =============================================================================
# Tracking Tools - For verifying execution order and state
# =============================================================================


def create_tracking_tool(call_tracker: List[Any]) -> ToolDefinition:
    """
    Create a tool that tracks all calls for verification.

    The tool appends its input value to the call_tracker list,
    allowing tests to verify execution order and data flow.

    Args:
        call_tracker: List to append tracked values to

    Returns:
        ToolDefinition configured for tracking
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        value = inputs.get("value", "")
        call_tracker.append(value)
        return value

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        value = inputs.get("value", "")
        call_tracker.append(value)
        return value

    return ToolDefinition(
        name="test.tracker",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": ["string", "array", "object", "number", "boolean", "null"]}
            },
            "additionalProperties": False,
        },
        output_schema={"type": ["string", "array", "object", "number", "boolean", "null"]},
        handler=handler,
        async_handler=async_handler,
    )


def create_echo_tool() -> ToolDefinition:
    """
    Create a tool that echoes its inputs back as output.

    Useful for testing state propagation through the workflow.

    Returns:
        ToolDefinition configured to return its inputs
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return inputs

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return inputs

    return ToolDefinition(
        name="test.echo",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "data": {"type": "object"},
            },
            "additionalProperties": True,
        },
        output_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "data": {"type": "object"},
            },
            "additionalProperties": True,
        },
        handler=handler,
        async_handler=async_handler,
    )


def create_transform_tool() -> ToolDefinition:
    """
    Create a tool that transforms its input by adding a prefix.

    Useful for testing data transformations through tool chains.

    Returns:
        ToolDefinition that transforms inputs
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> str:
        value = inputs.get("value", "")
        return f"transformed_{value}"

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> str:
        value = inputs.get("value", "")
        return f"transformed_{value}"

    return ToolDefinition(
        name="test.transform",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        output_schema={"type": "string"},
        handler=handler,
        async_handler=async_handler,
    )


def create_error_tool() -> ToolDefinition:
    """
    Create a tool that always raises an error.

    Useful for testing error handling and propagation.

    Returns:
        ToolDefinition that raises ExecutionError
    """
    from seer.core.errors import ExecutionError  # pylint: disable=import-outside-toplevel

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        message = inputs.get("message", "Test error")
        raise ExecutionError(message)

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        message = inputs.get("message", "Test error")
        raise ExecutionError(message)

    return ToolDefinition(
        name="test.error",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
        output_schema={"type": "null"},
        handler=handler,
        async_handler=async_handler,
    )


def create_aggregator_tool() -> ToolDefinition:
    """
    Create a tool that aggregates array inputs.

    Useful for testing aggregation after loops.

    Returns:
        ToolDefinition that counts and summarizes
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        items = inputs.get("items", [])
        return {"count": len(items), "items": items}

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        items = inputs.get("items", [])
        return {"count": len(items), "items": items}

    return ToolDefinition(
        name="test.aggregator",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"items": {"type": "array"}},
        },
        output_schema={
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "items": {"type": "array"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )


# =============================================================================
# Mock LLM Handlers - Deterministic responses for testing
# =============================================================================


def create_mock_text_llm_handler(response: str = "mock response"):
    """
    Create a deterministic text LLM handler.

    The handler returns a tuple of (result, usage_metadata) as expected
    by the runtime.

    Args:
        response: The response to return

    Returns:
        Callable that returns the fixed response with empty usage metadata
    """

    def handler(invocation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        # Return tuple of (result, usage_metadata)
        return response, {}

    return handler


def create_mock_json_llm_handler(response: Any = None):
    """
    Create a deterministic JSON LLM handler.

    The handler returns a tuple of (result, usage_metadata) as expected
    by the runtime.

    Args:
        response: The JSON response to return (defaults to {"result": "mock"})

    Returns:
        Callable that returns the fixed JSON response with empty usage metadata
    """
    if response is None:
        response = {"result": "mock"}

    def handler(invocation: Dict[str, Any], schema: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        # Return tuple of (result, usage_metadata)
        return response, {}

    return handler


def create_mock_array_llm_handler(items: List[Any] | None = None):
    """
    Create a deterministic LLM handler that returns an array.

    The handler returns a tuple of (result, usage_metadata) as expected
    by the runtime.

    Args:
        items: The array to return (defaults to ["item1", "item2", "item3"])

    Returns:
        Callable that returns the fixed array with empty usage metadata
    """
    if items is None:
        items = ["item1", "item2", "item3"]

    def handler(invocation: Dict[str, Any], schema: Dict[str, Any]) -> tuple[List[Any], Dict[str, Any]]:
        # Return tuple of (result, usage_metadata)
        return items, {}

    return handler


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def call_tracker() -> List[Any]:
    """Provide a fresh call tracker list for each test."""
    return []


@pytest.fixture
def tracking_tool(call_tracker: List[Any]) -> ToolDefinition:
    """Provide a tracking tool connected to the call_tracker fixture."""
    return create_tracking_tool(call_tracker)


@pytest.fixture
def echo_tool() -> ToolDefinition:
    """Provide an echo tool for state propagation tests."""
    return create_echo_tool()


@pytest.fixture
def transform_tool() -> ToolDefinition:
    """Provide a transform tool for data transformation tests."""
    return create_transform_tool()


@pytest.fixture
def error_tool() -> ToolDefinition:
    """Provide an error tool for error handling tests."""
    return create_error_tool()


@pytest.fixture
def aggregator_tool() -> ToolDefinition:
    """Provide an aggregator tool for loop tests."""
    return create_aggregator_tool()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Provide a fresh ToolRegistry."""
    return ToolRegistry()


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Provide a fresh ModelRegistry."""
    return ModelRegistry()


@pytest.fixture
def schema_registry() -> SchemaRegistry:
    """Provide a fresh SchemaRegistry."""
    return SchemaRegistry()


@pytest.fixture
def mcp_client_registry() -> MCPClientRegistry:
    """Provide a fresh MCPClientRegistry."""
    return MCPClientRegistry()


@pytest.fixture
def compiler_context(
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelRegistry,
    mcp_client_registry: MCPClientRegistry,
) -> CompilerContext:
    """Provide a complete CompilerContext with all registries."""
    return CompilerContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
        mcp_client_registry=mcp_client_registry,
    )


@pytest.fixture
def mock_llm_model(model_registry: ModelRegistry) -> ModelDefinition:
    """Register and return a mock LLM model with deterministic handlers."""
    model = ModelDefinition(
        model_id="qwen/qwen3-235b-a22b-2507",
        text_handler=create_mock_text_llm_handler("mock llm response"),
        json_handler=create_mock_json_llm_handler({"result": "mock json"}),
    )
    model_registry.register(model)
    return model


# =============================================================================
# Workflow Compilation Helper
# =============================================================================


async def compile_workflow(
    spec_payload: Dict[str, Any],
    tool_defs: List[ToolDefinition] | None = None,
    model_defs: List[ModelDefinition] | None = None,
    schema_registry: SchemaRegistry | None = None,
) -> CompiledWorkflow:
    """
    Helper to compile a workflow spec into an executable graph.

    This function performs the complete 5-stage compilation pipeline:
    1. Parse - Convert JSON payload to WorkflowSpec
    2. Type Environment - Build type information for all nodes
    3. Reference Validation - Verify ${...} expressions
    4. Control Flow Lowering - Build ExecutionPlan
    5. LangGraph Emission - Create executable graph

    Args:
        spec_payload: The workflow specification as a dict
        tool_defs: Optional list of tool definitions to register
        model_defs: Optional list of model definitions to register
        schema_registry: Optional schema registry (creates new if not provided)

    Returns:
        CompiledWorkflow ready for execution via ainvoke()
    """
    if schema_registry is None:
        schema_registry = SchemaRegistry()

    tool_registry = ToolRegistry()
    if tool_defs:
        for tool in tool_defs:
            tool_registry.register(tool)

    model_registry = ModelRegistry()
    if model_defs:
        for model in model_defs:
            model_registry.register(model)

    # Stage 1: Parse
    spec = parse_workflow_spec(spec_payload)

    # Stage 2: Build type environment
    type_env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    # Stage 3: Validate references
    validate_references(spec, type_env)

    # Stage 4: Build execution plan (control flow lowering)
    plan = build_execution_plan(spec)

    # Stage 5: Emit LangGraph
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


# =============================================================================
# Common Test Workflow Specs
# =============================================================================


def simple_trigger_spec() -> Dict[str, Any]:
    """Return a minimal trigger specification for testing."""
    return {
        "id": "test_trigger",
        "key": "test.trigger",
        "mode": "webhook",
        "event_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "data": {"type": "object"},
            },
        },
    }


def simple_tool_workflow_spec(tool_name: str = "test.tracker") -> Dict[str, Any]:
    """Return a minimal single-tool workflow spec."""
    return {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": tool_name,
                "inputs": {"value": "${test_trigger.message}"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }


def linear_workflow_spec(num_nodes: int = 3) -> Dict[str, Any]:
    """
    Return a linear workflow spec with N sequential nodes.

    Creates: node_0 -> node_1 -> ... -> node_{n-1}
    """
    nodes = []
    edges = [{"source": "test_trigger", "target": "node_0", "type": "trigger"}]

    for i in range(num_nodes):
        if i == 0:
            inputs = {"value": "${test_trigger.message}"}
        else:
            inputs = {"value": f"${{node_{i-1}}}"}

        nodes.append({
            "id": f"node_{i}",
            "type": "tool",
            "tool": "test.tracker",
            "inputs": inputs,
        })

        if i > 0:
            edges.append({
                "source": f"node_{i-1}",
                "target": f"node_{i}",
                "type": "default",
            })

    return {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": nodes,
        "edges": edges,
    }
