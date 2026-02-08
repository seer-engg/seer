# pylint: disable=unused-argument
# Reason: Mock functions have required signatures but don't use all parameters
"""
Tests for error handling in nested control flow scenarios.

These tests verify that:
- Tool failures in loops are properly captured in traces
- Error propagation works correctly in nested loops
- Partial success (some iterations succeed, others fail) is handled
- Conditional branches with failures are traced correctly
"""

from __future__ import annotations

import pytest

from seer.core.errors import ExecutionError
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


def _create_failing_tool(fail_on_values: list[str]) -> ToolDefinition:
    """Create a tool that fails when input value matches any in fail_on_values."""

    def handler(inputs, config, context):
        value = inputs.get("value", "")
        if value in fail_on_values:
            raise ValueError(f"Intentional test failure on: {value}")
        return f"processed_{value}"

    async def async_handler(inputs, config, context):
        value = inputs.get("value", "")
        if value in fail_on_values:
            raise ValueError(f"Intentional test failure on: {value}")
        return f"processed_{value}"

    return ToolDefinition(
        name="test.failing_tool",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={"type": "string"},
        handler=handler,
        async_handler=async_handler,
    )


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
# SINGLE LOOP ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_tool_failure_in_single_loop_captures_error_trace() -> None:
    """Test that tool failures in loops are captured with 'failed' status in trace.

    When a tool fails during one iteration:
    - The trace for that iteration should have status='failed'
    - The error message should be captured in the trace
    - The workflow should stop (default LangGraph behavior)
    """
    failing_tool = _create_failing_tool(fail_on_values=["fail"])

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                },
            }
        ],
        "nodes": [
            {"id": "loop", "type": "for_each", "items": "${test_trigger.items}"},
            {
                "id": "process",
                "type": "tool",
                "tool": "test.failing_tool",
                "inputs": {"value": "${item}"},
            },
            {"id": "done", "type": "tool", "tool": "test.failing_tool", "inputs": {"value": "complete"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["ok", "fail", "never_reached"],  # "fail" should cause error
    }

    # The workflow should raise an error when the tool fails
    with pytest.raises(ExecutionError, match="Intentional test failure on: fail"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)


@pytest.mark.asyncio
async def test_first_iteration_succeeds_before_failure() -> None:
    """Test that iterations before the failure are processed successfully.

    When iteration 1 (index 0) succeeds but iteration 2 (index 1) fails:
    - First iteration trace should have status='succeeded'
    - Workflow stops at the failing iteration
    """
    failing_tool = _create_failing_tool(fail_on_values=["b"])
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
                    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                },
            }
        ],
        "nodes": [
            {"id": "loop", "type": "for_each", "items": "${test_trigger.items}"},
            {"id": "track", "type": "tool", "tool": "test.tracker", "inputs": {"value": "${item}"}},
            {"id": "process", "type": "tool", "tool": "test.failing_tool", "inputs": {"value": "${item}"}},
            {"id": "done", "type": "tool", "tool": "test.tracker", "inputs": {"value": "complete"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "track", "type": "loop_body"},
            {"source": "track", "target": "process", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool, tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"],  # "b" will fail
    }

    with pytest.raises(ExecutionError, match="Intentional test failure on: b"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # "a" was tracked before failure, "b" was tracked but then process failed
    assert "a" in call_tracker
    assert "b" in call_tracker
    # "c" was never reached, "complete" was never reached
    assert "c" not in call_tracker
    assert "complete" not in call_tracker


# =============================================================================
# NESTED LOOP ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_inner_loop_failure_stops_workflow() -> None:
    """Test that a failure in an inner loop stops the entire workflow.

    Structure: outer_loop -> inner_loop -> process (fails)

    When the inner loop's process node fails:
    - The workflow should stop
    - Outer loop does not continue to next iteration
    """
    failing_tool = _create_failing_tool(fail_on_values=["A_fail"])
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
                        "inner_items": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.outer_items}",
                "item_var": "outer_item",
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner_items}",
                "item_var": "inner_item",
            },
            {
                "id": "track",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.failing_tool",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {"id": "after_inner", "type": "tool", "tool": "test.tracker", "inputs": {"value": "after_inner"}},
            {"id": "done", "type": "tool", "tool": "test.tracker", "inputs": {"value": "done"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_loop", "type": "trigger"},
            {"source": "outer_loop", "target": "inner_loop", "type": "loop_body"},
            {"source": "inner_loop", "target": "track", "type": "loop_body"},
            {"source": "track", "target": "process", "type": "default"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool, tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_items": ["A", "B"],
        "inner_items": ["ok", "fail"],  # A_fail will cause failure
    }

    with pytest.raises(ExecutionError, match="Intentional test failure on: A_fail"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # A_ok was tracked and processed before A_fail
    assert "A_ok" in call_tracker
    # A_fail was tracked but process failed
    assert "A_fail" in call_tracker
    # Nothing after the failure should be reached
    assert "after_inner" not in call_tracker
    assert "B_ok" not in call_tracker
    assert "done" not in call_tracker


@pytest.mark.asyncio
async def test_outer_loop_failure_skips_remaining_iterations() -> None:
    """Test that a failure in outer loop node stops before inner loop runs.

    Structure: outer_loop -> outer_process (fails on B) -> inner_loop -> ...

    When outer_process fails:
    - Inner loop for that iteration never executes
    - Subsequent outer iterations never execute
    """
    failing_tool = _create_failing_tool(fail_on_values=["B"])
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
                        "inner_items": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_loop",
                "type": "for_each",
                "items": "${test_trigger.outer_items}",
                "item_var": "outer_item",
            },
            {
                "id": "outer_process",
                "type": "tool",
                "tool": "test.failing_tool",
                "inputs": {"value": "${outer_item}"},  # Fails on "B"
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner_items}",
                "item_var": "inner_item",
            },
            {
                "id": "inner_track",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {"id": "after_inner", "type": "tool", "tool": "test.tracker", "inputs": {"value": "after_inner"}},
            {"id": "done", "type": "tool", "tool": "test.tracker", "inputs": {"value": "done"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_loop", "type": "trigger"},
            {"source": "outer_loop", "target": "outer_process", "type": "loop_body"},
            {"source": "outer_process", "target": "inner_loop", "type": "default"},
            {"source": "inner_loop", "target": "inner_track", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool, tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_items": ["A", "B", "C"],
        "inner_items": ["x", "y"],
    }

    with pytest.raises(ExecutionError, match="Intentional test failure on: B"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # A's inner loop should have completed
    assert "A_x" in call_tracker
    assert "A_y" in call_tracker
    # B failed at outer_process, so B's inner loop never ran
    assert "B_x" not in call_tracker
    # C was never reached
    assert "C_x" not in call_tracker


# =============================================================================
# CONDITIONAL BRANCH ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_conditional_branch_failure_in_loop() -> None:
    """Test that a failure in one branch of if-in-loop stops the workflow.

    Structure: loop -> if -> true_branch (fails on even) / false_branch

    When true_branch fails:
    - The workflow stops
    - False branch iterations before the failure completed successfully
    """
    failing_tool = _create_failing_tool(fail_on_values=["c_true"])
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
                    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                },
            }
        ],
        "nodes": [
            {"id": "loop", "type": "for_each", "items": "${test_trigger.items}"},
            {"id": "check", "type": "if", "condition": "${index} % 2 == 0"},  # Even -> true
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.failing_tool",
                "inputs": {"value": "${item}_true"},  # Fails on "c_true"
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_false"},
            },
            {"id": "done", "type": "tool", "tool": "test.tracker", "inputs": {"value": "done"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "check", "type": "loop_body"},
            {"source": "check", "target": "true_branch", "type": "conditional_true"},
            {"source": "check", "target": "false_branch", "type": "conditional_false"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool, tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c", "d"],  # indices 0,1,2,3 -> a=true, b=false, c=true(fail), d=false
    }

    with pytest.raises(ExecutionError, match="Intentional test failure on: c_true"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # b went to false_branch (succeeded)
    assert "b_false" in call_tracker
    # a went to true_branch (index 0, even) - should have succeeded (failing_tool returns processed_a_true)
    # Note: "a_true" doesn't fail because fail_on_values only has "c_true"
    # c went to true_branch (index 2, even) and failed
    # d was never reached
    assert "d_false" not in call_tracker


# =============================================================================
# MULTIPLE FAILURES TEST
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_stops_on_first_failure() -> None:
    """Test that workflow stops immediately on first failure, not continuing.

    This confirms LangGraph's default error handling behavior.
    """
    # Fail on both "b" and "c", but we should only see the "b" failure
    failing_tool = _create_failing_tool(fail_on_values=["b", "c"])
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
                    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                },
            }
        ],
        "nodes": [
            {"id": "loop", "type": "for_each", "items": "${test_trigger.items}"},
            {"id": "track", "type": "tool", "tool": "test.tracker", "inputs": {"value": "${item}"}},
            {"id": "process", "type": "tool", "tool": "test.failing_tool", "inputs": {"value": "${item}"}},
            {"id": "done", "type": "tool", "tool": "test.tracker", "inputs": {"value": "done"}},
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "track", "type": "loop_body"},
            {"source": "track", "target": "process", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [failing_tool, tracking_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c", "d"],  # b fails first, c would fail but never reached
    }

    # Should fail on "b", not "c"
    with pytest.raises(ExecutionError, match="Intentional test failure on: b"):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # "a" and "b" were tracked
    assert "a" in call_tracker
    assert "b" in call_tracker
    # "c" and "d" were never reached
    assert "c" not in call_tracker
    assert "d" not in call_tracker
