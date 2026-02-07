# pylint: disable=too-many-lines,unused-argument
# Reason: Comprehensive test coverage for nested control flow requires many test cases; mock functions have required signatures
"""
Comprehensive tests for nested control flow execution in the workflow compiler.

Tests cover:
- For-each containing if (loop with conditional branches)
- If containing for-each (conditional containing loop)
- Multiple levels of nesting (3+ levels)
- State isolation across nested control flow
- Trace key generation for nested scenarios

This test file addresses the critical gap identified in the Code Review & Test Gap Analysis
where nested control flow execution had ~10% coverage, and specifically:
- If within loop execution (~20% coverage)
- Loop within if execution (0% coverage - NOT TESTED AT ALL)
- Multiple nesting levels (0% coverage)
"""

from __future__ import annotations

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry


def _create_tracking_tool(call_tracker: list) -> ToolDefinition:
    """Create a tool that tracks all calls for verification."""
    def handler(inputs, config, context):
        call_tracker.append(inputs.get("value", ""))
        return inputs.get("value", "")

    async def async_handler(inputs, config, context):
        call_tracker.append(inputs.get("value", ""))
        return inputs.get("value", "")

    return ToolDefinition(
        name="test.tracker",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": ["string", "array", "object", "number", "boolean", "null"]}},
            "additionalProperties": False,
        },
        output_schema={"type": ["string", "array", "object", "number", "boolean", "null"]},
        handler=handler,
        async_handler=async_handler,
    )


async def _compile_workflow(
    spec_payload: dict,
    tool_defs: list[ToolDefinition],
) -> CompiledWorkflow:
    """Helper to compile a workflow spec into an executable graph."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    for tool in tool_defs:
        tool_registry.register(tool)
    model_registry = ModelRegistry()

    spec = parse_workflow_spec(spec_payload)
    type_env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)

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
# FOR-EACH CONTAINING IF (LOOP WITH CONDITIONAL BRANCHES)
# =============================================================================


@pytest.mark.asyncio
async def test_for_each_with_conditional_true_branch() -> None:
    """Test for_each loop with if node where condition is always true.

    This validates the bug fix for loop body detection when control flow
    nodes are present within the loop.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "check_item",
                "type": "if",
                "condition": "1 == 1",  # Always true for this test
            },
            {
                "id": "process_true",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_true"},
            },
            {
                "id": "process_false",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_false"},
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
            {"source": "loop", "target": "check_item", "type": "loop_body"},
            {"source": "check_item", "target": "process_true", "type": "conditional_true"},
            {"source": "check_item", "target": "process_false", "type": "conditional_false"},
            # NOTE: No explicit back-edges - implicit edges are added by the compiler
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all items went through true branch
    assert "a_true" in call_tracker
    assert "b_true" in call_tracker
    assert "c_true" in call_tracker
    # Verify false branch was never executed
    assert "a_false" not in call_tracker
    assert "b_false" not in call_tracker
    assert "c_false" not in call_tracker
    # Verify loop completed
    assert "completed" in call_tracker
    assert result["done"] == "completed"


@pytest.mark.asyncio
async def test_for_each_with_conditional_false_branch() -> None:
    """Test for_each loop with if node where condition is always false.

    This tests the false branch execution within a loop, which was
    identified as having minimal coverage.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "check_item",
                "type": "if",
                "condition": "1 == 0",  # Always false for this test
            },
            {
                "id": "process_true",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_true"},
            },
            {
                "id": "process_false",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_false"},
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
            {"source": "loop", "target": "check_item", "type": "loop_body"},
            {"source": "check_item", "target": "process_true", "type": "conditional_true"},
            {"source": "check_item", "target": "process_false", "type": "conditional_false"},
            # NOTE: No explicit back-edges - implicit edges are added by the compiler
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["x", "y", "z"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all items went through false branch
    assert "x_false" in call_tracker
    assert "y_false" in call_tracker
    assert "z_false" in call_tracker
    # Verify true branch was never executed
    assert "x_true" not in call_tracker
    assert "y_true" not in call_tracker
    assert "z_true" not in call_tracker
    # Verify loop completed
    assert "completed" in call_tracker
    assert result["done"] == "completed"


@pytest.mark.asyncio
async def test_for_each_with_conditional_index_based() -> None:
    """Test for_each loop where condition varies per iteration based on index.

    This is a critical test that validates:
    1. Each iteration can take a different branch based on index
    2. Loop state is properly maintained across different branches
    3. The bug fix for loop body detection works for real conditional logic
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "check_index",
                "type": "if",
                "condition": "${index} % 2 == 0",  # Even indices go true, odd go false
            },
            {
                "id": "process_even",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_even"},
            },
            {
                "id": "process_odd",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_odd"},
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
            {"source": "loop", "target": "check_index", "type": "loop_body"},
            {"source": "check_index", "target": "process_even", "type": "conditional_true"},
            {"source": "check_index", "target": "process_odd", "type": "conditional_false"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c", "d"]  # indices 0,1,2,3
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify correct branch was taken for each item based on index
    assert "a_even" in call_tracker  # index 0 (even)
    assert "b_odd" in call_tracker   # index 1 (odd)
    assert "c_even" in call_tracker  # index 2 (even)
    assert "d_odd" in call_tracker   # index 3 (odd)
    # Verify wrong branches were not taken
    assert "a_odd" not in call_tracker
    assert "b_even" not in call_tracker
    assert "c_odd" not in call_tracker
    assert "d_even" not in call_tracker
    # Verify loop completed
    assert result["done"] == "completed"


@pytest.mark.asyncio
async def test_for_each_with_if_and_continuation() -> None:
    """Test for_each loop with if node followed by continuation node.

    This specifically tests the RCA scenario where nodes after an if
    within a loop should have iteration-specific trace keys.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "check_item",
                "type": "if",
                "condition": "${index} == 0",  # Only first item goes to true branch
            },
            {
                "id": "transform",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_transformed"},
            },
            {
                "id": "log_status",  # This is the node that was not in loop_body_nodes
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_logged"},
            },
            {
                "id": "skip_log",  # False branch
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_skipped"},
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
            {"source": "loop", "target": "check_item", "type": "loop_body"},
            {"source": "check_item", "target": "transform", "type": "conditional_true"},
            {"source": "check_item", "target": "skip_log", "type": "conditional_false"},
            {"source": "transform", "target": "log_status", "type": "default"},
            # log_status and skip_log are terminal nodes - implicit back-edges
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["first", "second", "third"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify true branch (transform + log_status) for first item
    assert "first_transformed" in call_tracker
    assert "first_logged" in call_tracker

    # Verify false branch (skip_log) for other items
    assert "second_skipped" in call_tracker
    assert "third_skipped" in call_tracker

    # Verify loop completed
    assert result["done"] == "completed"


# =============================================================================
# IF CONTAINING FOR-EACH (CONDITIONAL CONTAINING LOOP)
# =============================================================================


@pytest.mark.asyncio
async def test_if_true_branch_contains_for_each() -> None:
    """Test conditional where true branch contains a for_each loop.

    CRITICAL: This was NOT TESTED AT ALL (0% coverage).
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "should_loop": {"type": "boolean"},
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_loop",
                "type": "if",
                "condition": "${test_trigger.should_loop}",
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "process_item",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_processed"},
            },
            {
                "id": "after_loop",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "loop_done"},
            },
            {
                "id": "skip_loop",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "skipped"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_loop", "type": "trigger"},
            {"source": "check_loop", "target": "loop", "type": "conditional_true"},
            {"source": "check_loop", "target": "skip_loop", "type": "conditional_false"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "loop", "target": "after_loop", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test when condition is true (loop should execute)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "should_loop": True,
        "items": ["a", "b", "c"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "a_processed" in call_tracker
    assert "b_processed" in call_tracker
    assert "c_processed" in call_tracker
    assert "loop_done" in call_tracker
    assert "skipped" not in call_tracker

    # Test when condition is false (loop should be skipped)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "should_loop": False,
        "items": ["a", "b", "c"]
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "a_processed" not in call_tracker
    assert "b_processed" not in call_tracker
    assert "c_processed" not in call_tracker
    assert "loop_done" not in call_tracker
    assert "skipped" in call_tracker


@pytest.mark.asyncio
async def test_if_false_branch_contains_for_each() -> None:
    """Test conditional where false branch contains a for_each loop.

    CRITICAL: This was NOT TESTED AT ALL (0% coverage).
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "skip_loop": {"type": "boolean"},
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_skip",
                "type": "if",
                "condition": "${test_trigger.skip_loop}",
            },
            {
                "id": "direct_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "direct"},
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "process_item",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_processed"},
            },
            {
                "id": "after_loop",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "loop_done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_skip", "type": "trigger"},
            {"source": "check_skip", "target": "direct_path", "type": "conditional_true"},
            {"source": "check_skip", "target": "loop", "type": "conditional_false"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "loop", "target": "after_loop", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test when condition is false (loop should execute in false branch)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "skip_loop": False,
        "items": ["x", "y"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "x_processed" in call_tracker
    assert "y_processed" in call_tracker
    assert "loop_done" in call_tracker
    assert "direct" not in call_tracker

    # Test when condition is true (direct path, no loop)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "skip_loop": True,
        "items": ["x", "y"]
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "x_processed" not in call_tracker
    assert "y_processed" not in call_tracker
    assert "direct" in call_tracker


# =============================================================================
# DEEPLY NESTED CONTROL FLOW (3+ LEVELS)
# =============================================================================


@pytest.mark.asyncio
async def test_for_each_containing_if_containing_for_each() -> None:
    """Test 3 levels of nesting: loop -> if -> loop.

    CRITICAL: This was NOT TESTED AT ALL (0% coverage for 3+ levels).

    KNOWN BUG: This test exposes a state isolation issue where inner loop
    item/index variables are not properly isolated between outer loop iterations.
    When outer_idx=0 (A), the inner loop should process ['x', 'y'], but due to
    the bug, the inner loop state from one outer iteration leaks into the next.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "outer_items": {"type": "array", "items": {"type": "string"}},
                        "inner_items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.outer_items}",
                "item_var": "outer_item",
                "index_var": "outer_idx",
            },
            {
                "id": "check_outer",
                "type": "if",
                "condition": "${outer_idx} % 2 == 0",  # Even outer indices run inner loop
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner_items}",
                "item_var": "inner_item",
                "index_var": "inner_idx",
            },
            {
                "id": "process_inner",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {
                "id": "after_inner_loop",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_inner_done"},
            },
            {
                "id": "skip_inner",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_skipped"},
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
            {"source": "outer_loop", "target": "check_outer", "type": "loop_body"},
            {"source": "check_outer", "target": "inner_loop", "type": "conditional_true"},
            {"source": "check_outer", "target": "skip_inner", "type": "conditional_false"},
            {"source": "inner_loop", "target": "process_inner", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner_loop", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_items": ["A", "B", "C"],  # indices 0, 1, 2
        "inner_items": ["x", "y"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify outer index 0 (A) processed inner loop
    assert "A_x" in call_tracker
    assert "A_y" in call_tracker
    assert "A_inner_done" in call_tracker

    # Verify outer index 1 (B) was skipped
    assert "B_skipped" in call_tracker
    assert "B_x" not in call_tracker
    assert "B_y" not in call_tracker

    # Verify outer index 2 (C) processed inner loop
    assert "C_x" in call_tracker
    assert "C_y" in call_tracker
    assert "C_inner_done" in call_tracker

    # Verify completion
    assert "all_done" in call_tracker
    assert result["done"] == "all_done"


@pytest.mark.asyncio
async def test_if_containing_for_each_containing_if() -> None:
    """Test 3 levels of nesting: if -> loop -> if.

    CRITICAL: This was NOT TESTED AT ALL (0% coverage for 3+ levels).
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "run_batch": {"type": "boolean"},
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_check",
                "type": "if",
                "condition": "${test_trigger.run_batch}",
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "inner_check",
                "type": "if",
                "condition": "${index} % 2 == 0",  # Even indices
            },
            {
                "id": "process_even",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_even"},
            },
            {
                "id": "process_odd",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_odd"},
            },
            {
                "id": "after_loop",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "batch_done"},
            },
            {
                "id": "skip_all",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "all_skipped"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_check", "type": "trigger"},
            {"source": "outer_check", "target": "loop", "type": "conditional_true"},
            {"source": "outer_check", "target": "skip_all", "type": "conditional_false"},
            {"source": "loop", "target": "inner_check", "type": "loop_body"},
            {"source": "inner_check", "target": "process_even", "type": "conditional_true"},
            {"source": "inner_check", "target": "process_odd", "type": "conditional_false"},
            {"source": "loop", "target": "after_loop", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test when outer condition is true
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "run_batch": True,
        "items": ["a", "b", "c"]  # indices 0, 1, 2
    }
    _ = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "a_even" in call_tracker  # index 0
    assert "b_odd" in call_tracker   # index 1
    assert "c_even" in call_tracker  # index 2
    assert "batch_done" in call_tracker
    assert "all_skipped" not in call_tracker

    # Test when outer condition is false (skip everything)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "run_batch": False,
        "items": ["a", "b", "c"]
    }
    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "a_even" not in call_tracker
    assert "a_odd" not in call_tracker
    assert "batch_done" not in call_tracker
    assert "all_skipped" in call_tracker


# =============================================================================
# STATE ISOLATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_loop_variables_available_in_both_if_branches() -> None:
    """Test that loop variables (item, index) are accessible in both if branches."""
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
                "item_var": "data",
                "index_var": "idx",
            },
            {
                "id": "check",
                "type": "if",
                "condition": "${idx} == 0",  # First item goes to true branch
            },
            {
                "id": "true_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "idx${idx}_${data}_true"},
            },
            {
                "id": "false_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "idx${idx}_${data}_false"},
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
            {"source": "loop", "target": "check", "type": "loop_body"},
            {"source": "check", "target": "true_handler", "type": "conditional_true"},
            {"source": "check", "target": "false_handler", "type": "conditional_false"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["first", "second"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop variables (idx and data) were accessible in both branches
    assert "idx0_first_true" in call_tracker
    assert "idx1_second_false" in call_tracker
    assert result["done"] == "complete"


@pytest.mark.asyncio
async def test_nested_loops_variable_isolation() -> None:
    """Test that inner and outer loop variables don't conflict.

    KNOWN BUG: The inner loop only executes once per outer iteration instead of
    the full number of items. This appears to be related to how the inner loop
    state (_loop_inner_loop) is reset between outer loop iterations.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

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
                        "inner": {"type": "array", "items": {"type": "string"}}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.outer}",
                "item_var": "outer_item",
                "index_var": "outer_idx",
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner}",
                "item_var": "inner_item",
                "index_var": "inner_idx",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "o${outer_idx}_i${inner_idx}_${outer_item}_${inner_item}"},
            },
            {
                "id": "after_inner",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "o${outer_idx}_${outer_item}_inner_done"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "complete"},
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

    compiled = await _compile_workflow(spec, [tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer": ["A", "B"],
        "inner": ["x", "y"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify each combination of outer and inner loop
    assert "o0_i0_A_x" in call_tracker
    assert "o0_i1_A_y" in call_tracker
    assert "o0_A_inner_done" in call_tracker
    assert "o1_i0_B_x" in call_tracker
    assert "o1_i1_B_y" in call_tracker
    assert "o1_B_inner_done" in call_tracker
    assert result["done"] == "complete"


# =============================================================================
# GENERALIZED N-LEVEL NESTING TESTS
# =============================================================================
# These tests dynamically generate workflow specs with n levels of nested loops
# to catch bugs that only manifest at specific nesting depths.
# =============================================================================


def _generate_n_level_nested_loop_spec(n: int, items_per_level: int = 2) -> tuple[dict, dict]:
    """
    Generate a workflow spec with n levels of nested for_each loops.

    Structure:
        loop_0 (level 0)
            └─> loop_1 (level 1)
                └─> loop_2 (level 2)
                    └─> ... (level n-1)
                        └─> process (innermost node)
                └─> after_1 (after level 2 completes)
            └─> after_0 (after level 1 completes)
        └─> done (after level 0 completes)

    Each level iterates over items like ["L0_0", "L0_1"] for level 0.

    Args:
        n: Number of nested loop levels (must be >= 1)
        items_per_level: Number of items each loop iterates over

    Returns:
        Workflow spec dict ready for compilation
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Generate item arrays for each level
    # Level 0: ["L0_0", "L0_1"], Level 1: ["L1_0", "L1_1"], etc.
    items_arrays = {
        f"level_{i}_items": [f"L{i}_{j}" for j in range(items_per_level)]
        for i in range(n)
    }

    # Build trigger with all item arrays
    trigger_schema_props = {
        f"level_{i}_items": {"type": "array", "items": {"type": "string"}}
        for i in range(n)
    }

    trigger = {
        "id": "test_trigger",
        "key": "test.trigger",
        "mode": "webhook",
        "event_schema": {
            "type": "object",
            "properties": trigger_schema_props
        }
    }

    nodes = []
    edges = []

    # Create nested loop nodes
    for level in range(n):
        loop_id = f"loop_{level}"
        items_expr = f"${{test_trigger.level_{level}_items}}"

        nodes.append({
            "id": loop_id,
            "type": "for_each",
            "items": items_expr,
            "item_var": f"item_{level}",
            "index_var": f"idx_{level}",
        })

    # Innermost processing node - captures all loop variables
    # Creates a string like "L0_0.L1_0.L2_0" to track the path
    item_refs = ".".join([f"${{item_{i}}}" for i in range(n)])
    nodes.append({
        "id": "process",
        "type": "tool",
        "tool": "test.tracker",
        "inputs": {"value": item_refs},
    })

    # After-loop nodes for each level (except outermost)
    for level in range(n - 1, 0, -1):
        nodes.append({
            "id": f"after_{level}",
            "type": "tool",
            "tool": "test.tracker",
            "inputs": {"value": f"after_level_{level}"},
        })

    # Final done node
    nodes.append({
        "id": "done",
        "type": "tool",
        "tool": "test.tracker",
        "inputs": {"value": "all_done"},
    })

    # Build edges: trigger -> loop_0
    edges.append({"source": "test_trigger", "target": "loop_0", "type": "trigger"})

    # Chain loops: loop_i -> loop_{i+1} (loop_body)
    for level in range(n - 1):
        edges.append({
            "source": f"loop_{level}",
            "target": f"loop_{level + 1}",
            "type": "loop_body"
        })

    # Innermost loop -> process
    edges.append({
        "source": f"loop_{n - 1}",
        "target": "process",
        "type": "loop_body"
    })

    # Inner loops exit to after nodes
    for level in range(n - 1, 0, -1):
        edges.append({
            "source": f"loop_{level}",
            "target": f"after_{level}",
            "type": "loop_exit"
        })

    # Outermost loop exits to done
    edges.append({
        "source": "loop_0",
        "target": "done",
        "type": "loop_exit"
    })

    return {
        "version": "2",
        "triggers": [trigger],
        "nodes": nodes,
        "edges": edges,
    }, items_arrays


def _compute_expected_combinations(n: int, items_per_level: int = 2) -> list[str]:
    """
    Compute all expected output strings for n nested loops.

    For n=2 with 2 items each, returns:
        ["L0_0.L1_0", "L0_0.L1_1", "L0_1.L1_0", "L0_1.L1_1"]

    This is the cartesian product of all item arrays.
    """
    from itertools import product

    # Generate item labels for each level
    level_items = [
        [f"L{level}_{i}" for i in range(items_per_level)]
        for level in range(n)
    ]

    # Cartesian product gives all combinations
    combinations = list(product(*level_items))

    # Join with dots to match the process node output format
    return [".".join(combo) for combo in combinations]


@pytest.mark.asyncio
async def test_single_level_loop() -> None:
    """
    Baseline test: single-level loop (n=1) works correctly.

    This confirms the base case works, helping isolate that the bug
    only manifests when nesting_depth >= 2.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_nested_loop_spec(1, items_per_level=3)
    compiled = await _compile_workflow(spec, [tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # All 3 items should be processed
    assert "L0_0" in call_tracker
    assert "L0_1" in call_tracker
    assert "L0_2" in call_tracker
    assert "all_done" in call_tracker


@pytest.mark.asyncio
@pytest.mark.parametrize("nesting_depth", [2, 3, 4, 5])
async def test_n_level_nested_loops(nesting_depth: int) -> None:
    """
    Parameterized test for n levels of nested for_each loops (n >= 2).

    This test dynamically generates workflows with 2-5 levels of nesting
    and verifies that all expected item combinations are processed.

    KNOWN BUG: For nesting_depth >= 2, inner loops don't complete all
    iterations due to state isolation issues between loop levels.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_nested_loop_spec(nesting_depth, items_per_level=2)

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Build trigger envelope with all item arrays
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Compute expected combinations
    expected = _compute_expected_combinations(nesting_depth, items_per_level=2)

    # Verify all expected combinations were processed
    for combo in expected:
        assert combo in call_tracker, f"Missing combination: {combo} (nesting_depth={nesting_depth})"

    # Verify total count matches (no duplicates, no missing)
    combo_count = sum(1 for c in call_tracker if "." in c or c.startswith("L"))
    expected_count = len(expected)
    assert combo_count == expected_count, (
        f"Expected {expected_count} combinations but got {combo_count} "
        f"(nesting_depth={nesting_depth})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("nesting_depth", [2, 3, 4])
async def test_n_level_trace_key_uniqueness(nesting_depth: int) -> None:
    """
    Verify trace keys are unique for each iteration path in n-level nesting.

    For n=2 with 2 items each, we expect trace keys like:
        _trace_process_iter_0_iter_0  (L0_0, L1_0)
        _trace_process_iter_0_iter_1  (L0_0, L1_1)
        _trace_process_iter_1_iter_0  (L0_1, L1_0)
        _trace_process_iter_1_iter_1  (L0_1, L1_1)

    KNOWN BUG: Trace keys are not correctly generated for nested loops,
    causing collisions and data overwrites.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_nested_loop_spec(nesting_depth, items_per_level=2)

    compiled = await _compile_workflow(spec, [tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Count unique trace keys for the 'process' node
    process_trace_keys = [k for k in result.keys() if k.startswith("_trace_process")]

    expected_count = 2 ** nesting_depth  # 2 items per level
    assert len(process_trace_keys) == expected_count, (
        f"Expected {expected_count} unique trace keys for 'process' node "
        f"but got {len(process_trace_keys)}: {process_trace_keys}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nesting_depth,items_per_level",
    [
        (2, 3),   # 2 levels, 3 items each = 9 combinations
        (3, 2),   # 3 levels, 2 items each = 8 combinations
        (4, 2),   # 4 levels, 2 items each = 16 combinations
        (2, 4),   # 2 levels, 4 items each = 16 combinations
    ]
)
async def test_n_level_with_varying_item_counts(nesting_depth: int, items_per_level: int) -> None:
    """
    Test nested loops with varying numbers of items per level.

    This catches bugs that might only manifest with specific item counts,
    such as off-by-one errors or modulo arithmetic issues.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_nested_loop_spec(nesting_depth, items_per_level)

    compiled = await _compile_workflow(spec, [tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    expected = _compute_expected_combinations(nesting_depth, items_per_level)
    expected_count = len(expected)  # items_per_level ^ nesting_depth

    # Count actual item combinations processed
    combo_count = sum(1 for c in call_tracker if c.startswith("L"))

    assert combo_count == expected_count, (
        f"Expected {expected_count} combinations "
        f"({items_per_level}^{nesting_depth}) but got {combo_count}"
    )


# =============================================================================
# N-LEVEL NESTING WITH CONDITIONALS
# =============================================================================


def _generate_n_level_loop_with_alternating_if(n: int) -> tuple[dict, dict]:
    """
    Generate workflow with n nested loops where even-indexed iterations
    at the outermost level skip the inner processing.

    Structure:
        loop_0 -> if_check (idx_0 % 2 == 0)
                    ├─ true: loop_1 -> ... -> process
                    └─ false: skip_node

    This tests conditional branches within nested loops.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    items_arrays = {
        f"level_{i}_items": [f"L{i}_{j}" for j in range(3)]  # 3 items per level
        for i in range(n)
    }

    trigger_schema_props = {
        f"level_{i}_items": {"type": "array", "items": {"type": "string"}}
        for i in range(n)
    }

    trigger = {
        "id": "test_trigger",
        "key": "test.trigger",
        "mode": "webhook",
        "event_schema": {"type": "object", "properties": trigger_schema_props}
    }

    nodes = []
    edges = []

    # Outermost loop
    nodes.append({
        "id": "loop_0",
        "type": "for_each",
        "items": "${test_trigger.level_0_items}",
        "item_var": "item_0",
        "index_var": "idx_0",
    })

    # Conditional check after outermost loop
    nodes.append({
        "id": "if_check",
        "type": "if",
        "condition": "${idx_0} % 2 == 0",  # Even indices run inner loops
    })

    # Skip node for odd indices
    nodes.append({
        "id": "skip_node",
        "type": "tool",
        "tool": "test.tracker",
        "inputs": {"value": "skipped_${item_0}"},
    })

    # Inner loops (levels 1 to n-1)
    for level in range(1, n):
        nodes.append({
            "id": f"loop_{level}",
            "type": "for_each",
            "items": f"${{test_trigger.level_{level}_items}}",
            "item_var": f"item_{level}",
            "index_var": f"idx_{level}",
        })

    # Innermost processing node
    item_refs = ".".join([f"${{item_{i}}}" for i in range(n)])
    nodes.append({
        "id": "process",
        "type": "tool",
        "tool": "test.tracker",
        "inputs": {"value": item_refs},
    })

    # Done node
    nodes.append({
        "id": "done",
        "type": "tool",
        "tool": "test.tracker",
        "inputs": {"value": "all_done"},
    })

    # Edges
    edges.append({"source": "test_trigger", "target": "loop_0", "type": "trigger"})
    edges.append({"source": "loop_0", "target": "if_check", "type": "loop_body"})
    edges.append({"source": "if_check", "target": "skip_node", "type": "conditional_false"})

    if n > 1:
        edges.append({"source": "if_check", "target": "loop_1", "type": "conditional_true"})
        for level in range(1, n - 1):
            edges.append({
                "source": f"loop_{level}",
                "target": f"loop_{level + 1}",
                "type": "loop_body"
            })
        edges.append({
            "source": f"loop_{n - 1}",
            "target": "process",
            "type": "loop_body"
        })
        # Add "after_level_{i}" nodes for each inner loop level to properly return control
        # Each inner loop needs its own "after" node to track when it completes
        for level in range(1, n):
            nodes.append({
                "id": f"after_level_{level}",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": f"after_level_{level}"},
            })
            edges.append({
                "source": f"loop_{level}",
                "target": f"after_level_{level}",
                "type": "loop_exit"
            })
    else:
        edges.append({"source": "if_check", "target": "process", "type": "conditional_true"})

    edges.append({"source": "loop_0", "target": "done", "type": "loop_exit"})

    return {
        "version": "2",
        "triggers": [trigger],
        "nodes": nodes,
        "edges": edges,
    }, items_arrays


@pytest.mark.asyncio
async def test_single_level_loop_with_conditional_filter() -> None:
    """
    Baseline test: single-level loop with conditional filter (n=1) works correctly.

    This confirms that conditionals work correctly within a single loop level.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_loop_with_alternating_if(1)
    compiled = await _compile_workflow(spec, [tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify conditional filtering works
    assert "skipped_L0_1" in call_tracker, "Odd index (1) should be skipped"
    assert "L0_0" in call_tracker, "Even index (0) should process"
    assert "L0_2" in call_tracker, "Even index (2) should process"
    assert "L0_1" not in call_tracker, "Index 1 should NOT be in process results"


@pytest.mark.asyncio
@pytest.mark.parametrize("nesting_depth", [2, 3])
async def test_n_level_loop_with_conditional_filter(nesting_depth: int) -> None:
    """
    Test n-level nested loops (n >= 2) with conditional filtering at the outermost level.

    For 3 items at level 0 (indices 0, 1, 2):
    - Index 0 (even): processes inner loops
    - Index 1 (odd): skipped
    - Index 2 (even): processes inner loops

    KNOWN BUG: For nesting_depth >= 2, the combination of conditionals and
    nested loops causes state isolation issues.
    """
    call_tracker: list = []
    tracking_tool = _create_tracking_tool(call_tracker)

    spec, items_arrays = _generate_n_level_loop_with_alternating_if(nesting_depth)

    compiled = await _compile_workflow(spec, [tracking_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        **items_arrays
    }

    await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Index 1 should be skipped
    assert "skipped_L0_1" in call_tracker, "Odd index should be skipped"

    # Check that L0_0 combinations exist (should have inner loop products)
    l0_0_combos = [c for c in call_tracker if c.startswith("L0_0.")]
    expected_combos = 3 ** (nesting_depth - 1)  # 3 items per inner level
    assert len(l0_0_combos) == expected_combos, (
        f"Index 0 should have {expected_combos} inner loop combinations, got {len(l0_0_combos)}"
    )

    # Check that L0_1 combinations don't exist
    l0_1_combos = [c for c in call_tracker if c.startswith("L0_1.")]
    assert len(l0_1_combos) == 0, "Index 1 should be skipped entirely"

    # Check that L0_2 combinations exist
    l0_2_combos = [c for c in call_tracker if c.startswith("L0_2.")]
    assert len(l0_2_combos) == expected_combos, (
        f"Index 2 should have {expected_combos} inner loop combinations, got {len(l0_2_combos)}"
    )
