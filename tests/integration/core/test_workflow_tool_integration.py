# pylint: disable=redefined-outer-name
# Reason: pytest fixtures intentionally shadow outer names for dependency injection
"""
End-to-end tests for workflow + tool integration.

Verifies the full path: compile workflow -> execute -> tool called -> output in state.

These tests address Gap 2 from the RCA: ensuring the complete workflow execution
path correctly integrates with tools and captures their output in workflow state.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from seer.core.errors import ExecutionError
from seer.core.registry.tool_registry import ToolDefinition

from .conftest import (
    compile_workflow,
    create_tracking_tool,
    simple_trigger_spec,
)


# =============================================================================
# WORKFLOW -> TOOL -> STATE E2E TESTS (Gap 2)
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_executes_tool_and_captures_output() -> None:
    """
    Verify workflow execution calls tool and captures output in state.

    This is the fundamental E2E test: a workflow with a tool node should
    execute the tool and store its output accessible via ${node_id}.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    # Create a custom tool with predictable output structure
    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        message = inputs.get("message", "")
        return {"result": f"processed: {message}", "original": message}

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        message = inputs.get("message", "")
        return {"result": f"processed: {message}", "original": message}

    output_tool = ToolDefinition(
        name="test.output_tool",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "original": {"type": "string"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "tool_node",
                "type": "tool",
                "tool": "test.output_tool",
                "inputs": {"message": "${test_trigger.message}"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "tool_node", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[output_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "hello_world",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify tool output is captured in state
    assert "tool_node" in result
    assert result["tool_node"]["result"] == "processed: hello_world"
    assert result["tool_node"]["original"] == "hello_world"


@pytest.mark.asyncio
async def test_workflow_tool_output_available_to_next_node() -> None:
    """
    Verify tool output can be referenced by subsequent nodes via ${node_id.field}.

    This tests the critical data flow: source_tool outputs data, and
    consumer_tool can access it via expression syntax.
    """
    # Source tool that produces data
    def source_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {"data": "source_value", "count": 42}

    async def source_async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {"data": "source_value", "count": 42}

    source_tool = ToolDefinition(
        name="test.source",
        version="v1",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "count": {"type": "integer"},
            },
        },
        handler=source_handler,
        async_handler=source_async_handler,
    )

    # Consumer tool that receives the source output
    received_inputs: Dict[str, Any] = {}

    def consumer_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        received_inputs.update(inputs)
        return {"received_data": inputs.get("input_data"), "received_count": inputs.get("input_count")}

    async def consumer_async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        received_inputs.update(inputs)
        return {"received_data": inputs.get("input_data"), "received_count": inputs.get("input_count")}

    consumer_tool = ToolDefinition(
        name="test.consumer",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "input_data": {"type": "string"},
                "input_count": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "received_data": {"type": "string"},
                "received_count": {"type": "integer"},
            },
        },
        handler=consumer_handler,
        async_handler=consumer_async_handler,
    )

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "source",
                "type": "tool",
                "tool": "test.source",
                "inputs": {},
            },
            {
                "id": "consumer",
                "type": "tool",
                "tool": "test.consumer",
                "inputs": {
                    "input_data": "${source.data}",
                    "input_count": "${source.count}",
                },
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "source", "type": "trigger"},
            {"source": "source", "target": "consumer", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[source_tool, consumer_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify data flowed from source to consumer
    assert result["source"]["data"] == "source_value"
    assert result["source"]["count"] == 42
    assert result["consumer"]["received_data"] == "source_value"
    assert result["consumer"]["received_count"] == 42

    # Verify the consumer tool received the correct inputs
    assert received_inputs["input_data"] == "source_value"
    assert received_inputs["input_count"] == 42


@pytest.mark.asyncio
async def test_workflow_tool_error_captured_in_state() -> None:
    """
    Verify tool failures are properly captured in workflow state with trace data.

    When a tool raises an exception, the workflow should capture error details
    in the trace data for debugging purposes.
    """

    def failing_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        raise ExecutionError("Tool failed intentionally")

    async def failing_async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Any:
        raise ExecutionError("Tool failed intentionally")

    failing_tool = ToolDefinition(
        name="test.failing",
        version="v1",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "null"},
        handler=failing_handler,
        async_handler=failing_async_handler,
    )

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "will_fail",
                "type": "tool",
                "tool": "test.failing",
                "inputs": {},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "will_fail", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[failing_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    # Tool failure should propagate as ExecutionError
    with pytest.raises(ExecutionError) as exc_info:
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify error message contains tool info
    assert "test.failing" in str(exc_info.value) or "will_fail" in str(exc_info.value)
    assert "failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_workflow_with_multiple_tools_in_sequence() -> None:
    """
    Verify multiple tools execute in correct order with state accumulation.

    Tests a 3-tool pipeline where each tool transforms the data.
    """
    execution_order: List[str] = []

    def create_tool(name: str, prefix: str) -> ToolDefinition:
        def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
            execution_order.append(name)
            value = inputs.get("value", "")
            return {"output": f"{prefix}_{value}"}

        async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
            execution_order.append(name)
            value = inputs.get("value", "")
            return {"output": f"{prefix}_{value}"}

        return ToolDefinition(
            name=name,
            version="v1",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
            handler=handler,
            async_handler=async_handler,
        )

    tool_a = create_tool("test.tool_a", "A")
    tool_b = create_tool("test.tool_b", "B")
    tool_c = create_tool("test.tool_c", "C")

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "step_a",
                "type": "tool",
                "tool": "test.tool_a",
                "inputs": {"value": "${test_trigger.message}"},
            },
            {
                "id": "step_b",
                "type": "tool",
                "tool": "test.tool_b",
                "inputs": {"value": "${step_a.output}"},
            },
            {
                "id": "step_c",
                "type": "tool",
                "tool": "test.tool_c",
                "inputs": {"value": "${step_b.output}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "step_a", "type": "trigger"},
            {"source": "step_a", "target": "step_b", "type": "default"},
            {"source": "step_b", "target": "step_c", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tool_a, tool_b, tool_c])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "start",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify execution order
    assert execution_order == ["test.tool_a", "test.tool_b", "test.tool_c"]

    # Verify state accumulation
    assert result["step_a"]["output"] == "A_start"
    assert result["step_b"]["output"] == "B_A_start"
    assert result["step_c"]["output"] == "C_B_A_start"


@pytest.mark.asyncio
async def test_workflow_tool_receives_trigger_data() -> None:
    """
    Verify tools can access trigger data via ${trigger_id.field} expressions.
    """
    received_trigger_data: Dict[str, Any] = {}

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        received_trigger_data.update(inputs)
        return {"processed": True}

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        received_trigger_data.update(inputs)
        return {"processed": True}

    tool = ToolDefinition(
        name="test.trigger_consumer",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "user_id": {"type": "string"},
                "action": {"type": "string"},
            },
        },
        output_schema={"type": "object", "properties": {"processed": {"type": "boolean"}}},
        handler=handler,
        async_handler=async_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "custom_trigger",
                "key": "test.custom",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "user_id": {"type": "string"},
                        "action": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "test.trigger_consumer",
                "inputs": {
                    "message": "${custom_trigger.message}",
                    "user_id": "${custom_trigger.user_id}",
                    "action": "${custom_trigger.action}",
                },
            }
        ],
        "edges": [
            {"source": "custom_trigger", "target": "process", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tool])

    trigger_envelope = {
        "trigger_id": "custom_trigger",
        "trigger_key": "test.custom",
        "message": "Hello from trigger",
        "user_id": "user_123",
        "action": "create",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify tool received trigger data
    assert received_trigger_data["message"] == "Hello from trigger"
    assert received_trigger_data["user_id"] == "user_123"
    assert received_trigger_data["action"] == "create"
    assert result["process"]["processed"] is True


@pytest.mark.asyncio
async def test_workflow_tool_with_nested_object_output() -> None:
    """
    Verify tools can return nested objects and they're accessible via dot notation.
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {
            "user": {
                "profile": {
                    "name": "John Doe",
                    "email": "john@example.com",
                },
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                },
            },
            "metadata": {"version": "1.0"},
        }

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return handler(inputs, config, context)

    # Full nested schema required for workflow compiler validation
    nested_tool = ToolDefinition(
        name="test.nested_output",
        version="v1",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        },
                        "settings": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "notifications": {"type": "boolean"},
                            },
                        },
                    },
                },
                "metadata": {
                    "type": "object",
                    "properties": {"version": {"type": "string"}},
                },
            },
        },
        handler=handler,
        async_handler=async_handler,
    )

    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "fetch_user",
                "type": "tool",
                "tool": "test.nested_output",
                "inputs": {},
            },
            {
                "id": "use_nested",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${fetch_user.user.profile.name}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fetch_user", "type": "trigger"},
            {"source": "fetch_user", "target": "use_nested", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[nested_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify nested access worked
    assert "John Doe" in call_tracker
    assert result["fetch_user"]["user"]["profile"]["name"] == "John Doe"
    assert result["fetch_user"]["user"]["settings"]["theme"] == "dark"


@pytest.mark.asyncio
async def test_workflow_tool_with_array_output() -> None:
    """
    Verify tools can return arrays and they're accessible for iteration.
    """

    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {
            "items": [
                {"id": 1, "name": "First"},
                {"id": 2, "name": "Second"},
                {"id": 3, "name": "Third"},
            ],
            "total": 3,
        }

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return handler(inputs, config, context)

    # Full nested schema with array item properties for workflow compiler validation
    array_tool = ToolDefinition(
        name="test.array_output",
        version="v1",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                },
                "total": {"type": "integer"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )

    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "fetch_items",
                "type": "tool",
                "tool": "test.array_output",
                "inputs": {},
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${fetch_items.items}",
            },
            {
                "id": "process_item",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item.name}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "completed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fetch_items", "type": "trigger"},
            {"source": "fetch_items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[array_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all items were processed
    assert "First" in call_tracker
    assert "Second" in call_tracker
    assert "Third" in call_tracker
    assert "completed" in call_tracker

    # Verify array data in state
    assert result["fetch_items"]["total"] == 3
    assert len(result["fetch_items"]["items"]) == 3
