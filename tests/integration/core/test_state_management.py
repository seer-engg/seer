# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for state management
"""
Integration tests for complex state propagation and isolation.

Tests verify that workflow state is correctly managed across nodes,
loops, and conditional branches.
"""
from __future__ import annotations

from typing import Any, List

import pytest

from .conftest import (
    compile_workflow,
    create_echo_tool,
    create_tracking_tool,
    simple_trigger_spec,
)


# =============================================================================
# LOOP STATE ISOLATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_loop_state_isolation_between_iterations() -> None:
    """
    Test that each iteration writes to its own trace key, no collisions.

    Trace keys should be unique per iteration to avoid overwriting
    previous iteration results.
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
                "inputs": {"value": "complete"},
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
        "items": ["x", "y", "z"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Count trace keys for process node
    process_traces = [k for k in result.keys() if "_trace_process" in k]
    assert len(process_traces) == 3, f"Expected 3 iteration traces, got {len(process_traces)}"

    # Verify all items were processed
    assert "x" in call_tracker
    assert "y" in call_tracker
    assert "z" in call_tracker


@pytest.mark.asyncio
async def test_nested_loop_state_reset_on_parent_iteration() -> None:
    """
    Test that inner loop resets when outer loop advances.

    The inner loop should execute fresh for each outer iteration.
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
                        "outer": {"type": "array", "items": {"type": "string"}},
                        "inner": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.outer}",
                "item_var": "outer_item",
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner}",
                "item_var": "inner_item",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {
                "id": "after_inner",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_inner_done"},
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
            {"source": "inner_loop", "target": "process", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer": ["A", "B"],
        "inner": ["1", "2"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all combinations were processed
    expected = ["A_1", "A_2", "B_1", "B_2"]
    for combo in expected:
        assert combo in call_tracker, f"Missing combination: {combo}"

    # Verify inner loop completed for each outer
    assert "A_inner_done" in call_tracker
    assert "B_inner_done" in call_tracker

    assert result["done"] == "all_done"


@pytest.mark.asyncio
async def test_if_branch_state_available_to_continuation() -> None:
    """
    Test that state from executed branch is accessible after if completes.
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
                "inputs": {"value": "true_value"},
            },
            {
                "id": "false_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "false_value"},
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

    # Verify true branch result is in final state
    assert "true_path" in result
    assert result["true_path"] == "true_value"


@pytest.mark.asyncio
async def test_loop_variable_scope_in_body_nodes() -> None:
    """
    Test that ${item} and ${index} are correctly scoped within loop body.
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
                "item_var": "data",
                "index_var": "pos",
            },
            {
                "id": "step_a",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "a_${pos}_${data}"},
            },
            {
                "id": "step_b",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "b_${pos}_${data}"},
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
            {"source": "loop", "target": "step_a", "type": "loop_body"},
            {"source": "step_a", "target": "step_b", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["foo", "bar"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop variables were correctly scoped in both nodes
    assert "a_0_foo" in call_tracker
    assert "b_0_foo" in call_tracker
    assert "a_1_bar" in call_tracker
    assert "b_1_bar" in call_tracker


@pytest.mark.asyncio
async def test_state_merge_preserves_all_node_outputs() -> None:
    """
    Test that state merge reducer keeps all keys from parallel updates.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "node_1",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "one"},
            },
            {
                "id": "node_2",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "two"},
            },
            {
                "id": "node_3",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "three"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "node_1", "type": "trigger"},
            {"source": "node_1", "target": "node_2", "type": "default"},
            {"source": "node_2", "target": "node_3", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all node outputs preserved
    assert "node_1" in result
    assert "node_2" in result
    assert "node_3" in result
    assert result["node_1"] == "one"
    assert result["node_2"] == "two"
    assert result["node_3"] == "three"


@pytest.mark.asyncio
async def test_trigger_data_immutable_during_execution() -> None:
    """
    Test that trigger data is not mutated by node execution.
    """
    echo_tool = create_echo_tool()
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
                        "original": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "echo_trigger",
                "type": "tool",
                "tool": "test.echo",
                "inputs": {"message": "${test_trigger.original}"},
            },
            {
                "id": "verify",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${test_trigger.original}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "echo_trigger", "type": "trigger"},
            {"source": "echo_trigger", "target": "verify", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[echo_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "original": "initial_value",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trigger data still accessible unchanged
    assert "initial_value" in call_tracker
    assert result["verify"] == "initial_value"


@pytest.mark.asyncio
async def test_deep_nested_expression_resolution() -> None:
    """
    Test that ${node.field.subfield[0].name} resolves correctly.
    """
    echo_tool = create_echo_tool()
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
                        "nested": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "extract",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${test_trigger.nested.items[0].name}"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "extract", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[echo_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "nested": {
            "items": [
                {"name": "first_item"},
                {"name": "second_item"},
            ]
        },
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify deep nested access worked
    assert "first_item" in call_tracker
    assert result["extract"] == "first_item"


@pytest.mark.asyncio
async def test_trace_keys_unique_across_all_iterations() -> None:
    """
    Test that trace keys are unique - no collisions across iterations.
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
                "inputs": {"value": "finished"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Use many items to stress uniqueness
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": [f"item_{i}" for i in range(10)],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Count unique trace keys
    process_traces = [k for k in result.keys() if "_trace_process" in k]

    # Should have exactly 10 unique trace keys for 10 iterations
    assert len(process_traces) == 10, f"Expected 10 traces, got {len(process_traces)}"

    # Verify uniqueness
    assert len(set(process_traces)) == len(process_traces), "Trace keys are not unique"


@pytest.mark.asyncio
async def test_loop_variables_available_in_both_if_branches() -> None:
    """
    Test that loop variables (item, index) are accessible in both if branches.
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
                "item_var": "val",
                "index_var": "i",
            },
            {
                "id": "check",
                "type": "if",
                "condition": "${i} == 0",
            },
            {
                "id": "first_item",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "first_${i}_${val}"},
            },
            {
                "id": "other_item",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "other_${i}_${val}"},
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
            {"source": "loop", "target": "check", "type": "loop_body"},
            {"source": "check", "target": "first_item", "type": "conditional_true"},
            {"source": "check", "target": "other_item", "type": "conditional_false"},
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

    # Verify loop variables were accessible in BOTH branches
    assert "first_0_a" in call_tracker  # First item, true branch
    assert "other_1_b" in call_tracker  # Second item, false branch
    assert "other_2_c" in call_tracker  # Third item, false branch
    assert result["done"] == "done"
