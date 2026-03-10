# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for workflow execution
"""
Integration tests for workflow execution.

Tests verify that compiled workflows execute correctly with real
state propagation between nodes.
"""
from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from seer.core.registry.model_registry import ModelDefinition

from .conftest import (
    compile_workflow,
    create_echo_tool,
    create_tracking_tool,
    simple_trigger_spec,
)


# =============================================================================
# BASIC EXECUTION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_execute_simple_linear_workflow() -> None:
    """
    Test execution of A -> B -> C linear workflow.

    Verifies that state propagates correctly through sequential nodes.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "node_a",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${test_trigger.message}_A"},
            },
            {
                "id": "node_b",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${node_a}_B"},
            },
            {
                "id": "node_c",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${node_b}_C"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "node_a", "type": "trigger"},
            {"source": "node_a", "target": "node_b", "type": "default"},
            {"source": "node_b", "target": "node_c", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "hello",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify execution order
    assert "hello_A" in call_tracker
    assert "hello_A_B" in call_tracker
    assert "hello_A_B_C" in call_tracker

    # Verify call order
    assert call_tracker.index("hello_A") < call_tracker.index("hello_A_B")
    assert call_tracker.index("hello_A_B") < call_tracker.index("hello_A_B_C")

    # Verify final result
    assert result["node_c"] == "hello_A_B_C"


@pytest.mark.asyncio
async def test_execute_workflow_with_tool_chain() -> None:
    """
    Test tool output from node A is input to node B via ${A.field}.

    Verifies that complex tool outputs can be referenced by subsequent nodes.
    """
    echo_tool = create_echo_tool()
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "fetch",
                "type": "tool",
                "tool": "test.echo",
                "inputs": {
                    "message": "${test_trigger.message}",
                    "data": {"count": 42},
                },
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${fetch.message}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fetch", "type": "trigger"},
            {"source": "fetch", "target": "process", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[echo_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test_message",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify fetch output is accessible to process
    assert "test_message" in call_tracker
    assert result["process"] == "test_message"


@pytest.mark.asyncio
async def test_execute_workflow_preserves_all_trace_data() -> None:
    """
    Test that all _trace_* keys are present in final state after execution.

    Trace data is critical for debugging and should always be preserved.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "node_a",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "a"},
            },
            {
                "id": "node_b",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "b"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "node_a", "type": "trigger"},
            {"source": "node_a", "target": "node_b", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trace keys are present
    trace_keys = [k for k in result.keys() if k.startswith("_trace_")]
    assert len(trace_keys) >= 2  # At least one trace per node

    # Verify trace for each node
    node_a_traces = [k for k in trace_keys if "node_a" in k]
    node_b_traces = [k for k in trace_keys if "node_b" in k]
    assert len(node_a_traces) >= 1
    assert len(node_b_traces) >= 1


@pytest.mark.asyncio
async def test_execute_workflow_ainvoke_returns_filtered_state() -> None:
    """
    Test that internal state keys (double underscore) are filtered except __interrupt__.

    CompiledWorkflow.ainvoke should filter out internal state to return
    a clean output to callers.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${test_trigger.message}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify no double-underscore keys except __interrupt__
    internal_keys = [k for k in result.keys() if k.startswith("__") and k != "__interrupt__"]
    assert len(internal_keys) == 0, f"Found internal keys: {internal_keys}"


@pytest.mark.asyncio
async def test_execute_workflow_with_trigger_data() -> None:
    """
    Test that trigger envelope is accessible via ${trigger_id.field}.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "my_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": "string"},
                        "action": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "greet",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "Hello ${my_trigger.user_name}"},
            },
            {
                "id": "action",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "Action: ${my_trigger.action}"},
            },
        ],
        "edges": [
            {"source": "my_trigger", "target": "greet", "type": "trigger"},
            {"source": "greet", "target": "action", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "my_trigger",
        "trigger_key": "test.trigger",
        "user_name": "Alice",
        "action": "login",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trigger data was accessible
    assert "Hello Alice" in call_tracker
    assert "Action: login" in call_tracker
    assert result["greet"] == "Hello Alice"
    assert result["action"] == "Action: login"


@pytest.mark.asyncio
async def test_execute_workflow_accumulates_state_across_nodes() -> None:
    """
    Test that state from all executed nodes accumulates in final result.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "step1",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "one"},
            },
            {
                "id": "step2",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "two"},
            },
            {
                "id": "step3",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "three"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "step1", "type": "trigger"},
            {"source": "step1", "target": "step2", "type": "default"},
            {"source": "step2", "target": "step3", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all node outputs are in result
    assert "step1" in result
    assert "step2" in result
    assert "step3" in result
    assert result["step1"] == "one"
    assert result["step2"] == "two"
    assert result["step3"] == "three"


# =============================================================================
# LLM EXECUTION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_execute_workflow_with_agent_text_output() -> None:
    """
    Test execution with agent node in text mode.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="test-model",
        chat_model_factory=lambda: MagicMock(),
    )

    # Agent node uses inputs for model/prompt and outputs for mode
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "generate",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Generate something for: ${test_trigger.message}",
                },
                "outputs": {"mode": "text"},
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${generate}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "generate", "type": "trigger"},
            {"source": "generate", "target": "process", "type": "default"},
        ],
    }

    compiled = await compile_workflow(
        spec,
        tool_defs=[tracking_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test input",
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Generated text response")]}
    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify agent output was captured and passed to next node
    assert "Generated text response" in call_tracker
    assert result["generate"] == "Generated text response"


@pytest.mark.asyncio
async def test_execute_workflow_with_agent_json_output() -> None:
    """
    Test execution with agent node in JSON mode with schema validation.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="test-model",
        chat_model_factory=lambda: MagicMock(),
    )

    # Agent node uses inputs for model/prompt and outputs for mode/schema
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "generate",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Extract data from: ${test_trigger.message}",
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "score": {"type": "integer"},
                            },
                            "required": ["name", "score"],
                        }
                    },
                },
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${generate.name}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "generate", "type": "trigger"},
            {"source": "generate", "target": "process", "type": "default"},
        ],
    }

    compiled = await compile_workflow(
        spec,
        tool_defs=[tracking_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "some text",
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content='{"name": "Test Name", "score": 95}')]
    }
    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify agent JSON output structure
    assert result["generate"]["name"] == "Test Name"
    assert result["generate"]["score"] == 95

    # Verify downstream access to JSON fields
    assert "Test Name" in call_tracker


# =============================================================================
# CONTROL FLOW EXECUTION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_execute_workflow_with_if_true_branch() -> None:
    """
    Test conditional execution takes true branch when condition is true.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                },
            }
        ],
        "nodes": [
            {
                "id": "check",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "true_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "true_branch"},
            },
            {
                "id": "false_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "false_branch"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check", "type": "trigger"},
            {"source": "check", "target": "true_path", "type": "conditional_true"},
            {"source": "check", "target": "false_path", "type": "conditional_false"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": True,
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify true branch was taken
    assert "true_branch" in call_tracker
    assert "false_branch" not in call_tracker


@pytest.mark.asyncio
async def test_execute_workflow_with_for_each_loop() -> None:
    """
    Test for_each loop iterates over all items.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "completed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["apple", "banana", "cherry"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all items were processed
    assert "apple" in call_tracker
    assert "banana" in call_tracker
    assert "cherry" in call_tracker
    assert "completed" in call_tracker

    # Verify done comes after all items
    assert call_tracker.index("completed") > call_tracker.index("cherry")


@pytest.mark.asyncio
async def test_execute_workflow_with_loop_index_access() -> None:
    """
    Test that loop index variable is accessible within loop body.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
                "item_var": "item",
                "index_var": "idx",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${idx}_${item}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify index was accessible
    assert "0_a" in call_tracker
    assert "1_b" in call_tracker
    assert "2_c" in call_tracker
