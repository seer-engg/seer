# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for multi-node interactions
"""
Integration tests for multiple node types working together.

Tests verify that different node types (Tool, LLM, If, ForEach)
interact correctly in complex workflow scenarios.
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
    create_transform_tool,
    simple_trigger_spec,
)


# =============================================================================
# TOOL CHAIN TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_tool_to_tool_pipeline() -> None:
    """
    Test tool chain passing data through multiple tools.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)
    transform_tool = create_transform_tool()

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "transform",
                "type": "tool",
                "tool": "test.transform",
                "inputs": {"value": "${test_trigger.message}"},
            },
            {
                "id": "track",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${transform}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "transform", "type": "trigger"},
            {"source": "transform", "target": "track", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[transform_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "input",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify transformation chain
    assert result["transform"] == "transformed_input"
    assert "transformed_input" in call_tracker


@pytest.mark.asyncio
async def test_tool_to_agent_to_tool_pipeline() -> None:
    """
    Test mixed node type chain: Tool -> Agent -> Tool.

    Verifies data flows correctly through different node types.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)
    echo_tool = create_echo_tool()

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
                "id": "fetch",
                "type": "tool",
                "tool": "test.echo",
                "inputs": {"message": "${test_trigger.message}"},
            },
            {
                "id": "process_llm",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Process: ${fetch.message}",
                },
                "outputs": {"mode": "text"},
            },
            {
                "id": "final",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${process_llm}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fetch", "type": "trigger"},
            {"source": "fetch", "target": "process_llm", "type": "default"},
            {"source": "process_llm", "target": "final", "type": "default"},
        ],
    }

    compiled = await compile_workflow(
        spec,
        tool_defs=[echo_tool, tracking_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "initial data",
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="LLM processed the data")]}
    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify chain executed correctly
    assert result["fetch"]["message"] == "initial data"
    assert result["process_llm"] == "LLM processed the data"
    assert "LLM processed the data" in call_tracker


@pytest.mark.asyncio
async def test_agent_generates_data_for_loop_iteration() -> None:
    """
    Test Agent generates object with array -> ForEach iterates -> Tool processes each.

    Note: OpenAI structured outputs require root type 'object', not 'array'.
    So we wrap the array in an object with a 'tasks' property.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="test-model",
        chat_model_factory=lambda: MagicMock(),
    )

    # Agent node uses inputs for model/prompt and outputs for mode/schema
    # Schema must have root type 'object' per OpenAI constraints
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "generate_tasks",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Generate tasks for: ${test_trigger.message}",
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "tasks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                        }
                    },
                },
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${generate_tasks.tasks}",  # Access the tasks array property
            },
            {
                "id": "process_task",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "processing_${item}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "all_done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "generate_tasks", "type": "trigger"},
            {"source": "generate_tasks", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_task", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
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
        "message": "project",
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content='{"tasks": ["task1", "task2", "task3"]}')]
    }
    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify agent generated object with tasks array
    assert result["generate_tasks"]["tasks"] == ["task1", "task2", "task3"]

    # Verify each task was processed
    assert "processing_task1" in call_tracker
    assert "processing_task2" in call_tracker
    assert "processing_task3" in call_tracker
    assert "all_done" in call_tracker


@pytest.mark.asyncio
async def test_conditional_routes_to_different_tool_chains() -> None:
    """
    Test If condition routes to different tool sequences.
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
                    "properties": {
                        "priority": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "check_priority",
                "type": "if",
                "condition": "${test_trigger.priority} == 'high'",
            },
            {
                "id": "urgent_step1",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "urgent_1"},
            },
            {
                "id": "urgent_step2",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "urgent_2"},
            },
            {
                "id": "normal_step",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "normal"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_priority", "type": "trigger"},
            {"source": "check_priority", "target": "urgent_step1", "type": "conditional_true"},
            {"source": "check_priority", "target": "normal_step", "type": "conditional_false"},
            {"source": "urgent_step1", "target": "urgent_step2", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Test high priority path
    call_tracker.clear()
    trigger_high = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "priority": "high",
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_high)

    assert "urgent_1" in call_tracker
    assert "urgent_2" in call_tracker
    assert "normal" not in call_tracker

    # Test normal priority path
    call_tracker.clear()
    trigger_normal = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "priority": "low",
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_normal)

    assert "urgent_1" not in call_tracker
    assert "urgent_2" not in call_tracker
    assert "normal" in call_tracker


@pytest.mark.asyncio
async def test_nested_loops_with_tool_in_inner_loop() -> None:
    """
    Test nested loops: Outer loop -> Inner loop -> Tool.
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
                    "properties": {
                        "rows": {"type": "array", "items": {"type": "string"}},
                        "cols": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.rows}",
                "item_var": "row",
                "index_var": "row_idx",
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.cols}",
                "item_var": "col",
                "index_var": "col_idx",
            },
            {
                "id": "process_cell",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${row}_${col}"},
            },
            {
                "id": "after_inner",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${row}_row_done"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "all_done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_loop", "type": "trigger"},
            {"source": "outer_loop", "target": "inner_loop", "type": "loop_body"},
            {"source": "inner_loop", "target": "process_cell", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "rows": ["R1", "R2"],
        "cols": ["C1", "C2"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all cell combinations processed
    assert "R1_C1" in call_tracker
    assert "R1_C2" in call_tracker
    assert "R2_C1" in call_tracker
    assert "R2_C2" in call_tracker

    # Verify row completions
    assert "R1_row_done" in call_tracker
    assert "R2_row_done" in call_tracker

    # Verify final done
    assert "all_done" in call_tracker
    assert result["done"] == "all_done"


@pytest.mark.asyncio
async def test_complex_branching_with_join() -> None:
    """
    Test If branches -> different processing -> same end node.

    Note: LangGraph handles convergent branches differently -
    typically only one branch executes based on condition.
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
                    "properties": {"type": {"type": "string"}},
                },
            }
        ],
        "nodes": [
            {
                "id": "check_type",
                "type": "if",
                "condition": "${test_trigger.type} == 'A'",
            },
            {
                "id": "process_a",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "processed_A"},
            },
            {
                "id": "process_b",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "processed_B"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_type", "type": "trigger"},
            {"source": "check_type", "target": "process_a", "type": "conditional_true"},
            {"source": "check_type", "target": "process_b", "type": "conditional_false"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Test type A
    call_tracker.clear()
    trigger_a = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "type": "A",
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_a)

    assert "processed_A" in call_tracker
    assert "processed_B" not in call_tracker

    # Test type B
    call_tracker.clear()
    trigger_b = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "type": "B",
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_b)

    assert "processed_A" not in call_tracker
    assert "processed_B" in call_tracker


@pytest.mark.asyncio
async def test_multiple_independent_tool_nodes() -> None:
    """
    Test multiple independent tool nodes process correctly.

    When multiple entry points exist (e.g., multiple triggers),
    each trigger activates its own subtree.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_x",
                "key": "test.x",
                "mode": "webhook",
                "event_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
            },
        ],
        "nodes": [
            {
                "id": "step_1",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "step1_${trigger_x.data}"},
            },
            {
                "id": "step_2",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "step2_${step_1}"},
            },
        ],
        "edges": [
            {"source": "trigger_x", "target": "step_1", "type": "trigger"},
            {"source": "step_1", "target": "step_2", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "trigger_x",
        "trigger_key": "test.x",
        "data": "payload",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "step1_payload" in call_tracker
    assert "step2_step1_payload" in call_tracker
    assert result["step_1"] == "step1_payload"


@pytest.mark.asyncio
async def test_for_each_with_conditional_inside() -> None:
    """
    Test ForEach loop with If node inside the loop body.
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
                "id": "check_even",
                "type": "if",
                "condition": "${idx} % 2 == 0",
            },
            {
                "id": "even_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_even"},
            },
            {
                "id": "odd_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_odd"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "loop_done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "check_even", "type": "loop_body"},
            {"source": "check_even", "target": "even_handler", "type": "conditional_true"},
            {"source": "check_even", "target": "odd_handler", "type": "conditional_false"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c", "d"],  # indices 0, 1, 2, 3
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify even indices (0, 2) went to even_handler
    assert "a_even" in call_tracker  # index 0
    assert "c_even" in call_tracker  # index 2

    # Verify odd indices (1, 3) went to odd_handler
    assert "b_odd" in call_tracker  # index 1
    assert "d_odd" in call_tracker  # index 3

    # Verify loop completion
    assert "loop_done" in call_tracker
