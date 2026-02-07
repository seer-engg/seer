# pylint: disable=too-many-lines,unused-argument
# Reason: Comprehensive test coverage for trace key generation requires many test cases; mock functions have required signatures
"""
Tests for trace key generation in control flow scenarios.

These tests verify that:
- Trace keys are correctly generated for if/else branches
- Loop iterations have unique trace keys (_iter_N suffix)
- Nested control flow produces correct trace hierarchy
- The bug fix for loop body detection results in correct trace keys

This addresses the Test Gap Analysis recommendation:
"Add trace verification for all control flow paths"
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


def _create_mock_tool() -> ToolDefinition:
    """Create a mock test.tool that simply returns its input value."""
    def handler(inputs, config, context):
        return inputs.get("value", "")

    async def async_handler(inputs, config, context):
        return inputs.get("value", "")

    return ToolDefinition(
        name="test.tool",
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
# BASIC TRACE KEY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_simple_node_trace_key() -> None:
    """Test that a simple tool node generates correct trace key."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "my_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "test"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "my_node", "type": "trigger"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trace key exists with correct format
    assert "_trace_my_node" in result
    trace = result["_trace_my_node"]
    assert trace["node_id"] == "my_node"
    assert trace["node_type"] == "tool"
    assert trace["status"] == "succeeded"
    assert "inputs" in trace
    assert "output" in trace
    assert "timestamp" in trace


# =============================================================================
# IF/ELSE BRANCH TRACE KEY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_if_true_branch_creates_correct_trace_key() -> None:
    """Test that trace keys are correctly generated for true branch execution."""
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
                        "flag": {"type": "boolean"}
                    }
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
                "id": "true_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "true_result"},
            },
            {
                "id": "false_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "false_result"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check", "type": "trigger"},
            {"source": "check", "target": "true_node", "type": "conditional_true"},
            {"source": "check", "target": "false_node", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": True
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify true branch trace exists
    assert "_trace_true_node" in result
    assert result["_trace_true_node"]["node_id"] == "true_node"
    assert result["_trace_true_node"]["status"] == "succeeded"

    # Verify false branch trace does NOT exist (node wasn't executed)
    assert "_trace_false_node" not in result


@pytest.mark.asyncio
async def test_if_false_branch_creates_correct_trace_key() -> None:
    """Test that trace keys are correctly generated for false branch execution."""
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
                        "flag": {"type": "boolean"}
                    }
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
                "id": "true_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "true_result"},
            },
            {
                "id": "false_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "false_result"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check", "type": "trigger"},
            {"source": "check", "target": "true_node", "type": "conditional_true"},
            {"source": "check", "target": "false_node", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": False
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify false branch trace exists
    assert "_trace_false_node" in result
    assert result["_trace_false_node"]["node_id"] == "false_node"
    assert result["_trace_false_node"]["status"] == "succeeded"

    # Verify true branch trace does NOT exist (node wasn't executed)
    assert "_trace_true_node" not in result


# =============================================================================
# FOR-EACH LOOP TRACE KEY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_for_each_iteration_trace_keys() -> None:
    """Test that each loop iteration has a unique trace key with _iter_N suffix."""
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
                        "items": {"type": "array"}
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
                "id": "process",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${item}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "finished"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "process", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify each iteration has its own trace key
    assert "_trace_process_iter_0" in result
    assert "_trace_process_iter_1" in result
    assert "_trace_process_iter_2" in result

    # Verify trace content is correct for each iteration
    assert result["_trace_process_iter_0"]["inputs"]["value"] == "a"
    assert result["_trace_process_iter_1"]["inputs"]["value"] == "b"
    assert result["_trace_process_iter_2"]["inputs"]["value"] == "c"

    # Verify "done" node (outside loop) has standard trace key (no _iter suffix)
    assert "_trace_done" in result
    assert result["_trace_done"]["node_id"] == "done"


# =============================================================================
# NESTED CONTROL FLOW TRACE KEY TESTS (THE BUG FIX VALIDATION)
# =============================================================================


@pytest.mark.asyncio
async def test_for_each_with_if_iteration_traces() -> None:
    """Test that nodes after if within loop have iteration-specific trace keys.

    This is the CRITICAL test that validates the bug fix:
    Previously, log_sent_status (after if) was NOT detected as part of the loop body,
    so all iterations wrote to the same _trace_log_sent_status key (collision).

    After the fix, each iteration should have _trace_log_sent_status_iter_N.
    """
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
                "condition": "${index} % 2 == 0",  # Even indices go to true branch
            },
            {
                "id": "transform",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${item}_transformed"},
            },
            {
                "id": "log_status",  # The node that was problematic in the bug
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${item}_logged"},
            },
            {
                "id": "skip",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${item}_skipped"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "complete"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "check_item", "type": "loop_body"},
            {"source": "check_item", "target": "transform", "type": "conditional_true"},
            {"source": "check_item", "target": "skip", "type": "conditional_false"},
            {"source": "transform", "target": "log_status", "type": "default"},
            # NOTE: No explicit back-edges - implicit edges are added by the compiler
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"]  # indices 0, 1, 2
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # CRITICAL ASSERTIONS - validates the bug fix:
    # Nodes after if (log_status) should have iteration-specific trace keys

    # Iteration 0 (even): true branch -> transform -> log_status
    assert "_trace_transform_iter_0" in result
    assert "_trace_log_status_iter_0" in result
    assert result["_trace_log_status_iter_0"]["inputs"]["value"] == "a_logged"

    # Iteration 1 (odd): false branch -> skip
    assert "_trace_skip_iter_1" in result

    # Iteration 2 (even): true branch -> transform -> log_status
    assert "_trace_transform_iter_2" in result
    assert "_trace_log_status_iter_2" in result
    assert result["_trace_log_status_iter_2"]["inputs"]["value"] == "c_logged"

    # Verify no collision: _trace_log_status (without _iter) should NOT exist
    assert "_trace_log_status" not in result

    # Verify done node is outside loop (no _iter suffix)
    assert "_trace_done" in result


@pytest.mark.asyncio
async def test_nested_loops_trace_keys() -> None:
    """Test trace keys for nested loop scenarios.

    KNOWN BUG: The outer loop iteration indices in trace keys appear to be off.
    Expected _trace_after_inner_iter_0 and _trace_after_inner_iter_1, but
    actual keys start from _iter_1 and _iter_2. Related to state isolation bug.
    """
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
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner_items}",
                "item_var": "inner_item",
            },
            {
                "id": "process_inner",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${inner_item}"},
            },
            {
                "id": "after_inner",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "inner_done"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "all_done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_loop", "type": "trigger"},
            {"source": "outer_loop", "target": "inner_loop", "type": "loop_body"},
            {"source": "inner_loop", "target": "process_inner", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_items": ["A", "B"],
        "inner_items": ["x", "y"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Inner loop iterations should have iteration-specific traces
    # With nested loops, trace keys include both outer and inner iteration indices
    # Format: _trace_{node_id}_iter_{outer_idx}_iter_{inner_idx}
    assert "_trace_process_inner_iter_0_iter_0" in result  # outer=0 (A), inner=0 (x)
    assert "_trace_process_inner_iter_0_iter_1" in result  # outer=0 (A), inner=1 (y)
    assert "_trace_process_inner_iter_1_iter_0" in result  # outer=1 (B), inner=0 (x)
    assert "_trace_process_inner_iter_1_iter_1" in result  # outer=1 (B), inner=1 (y)

    # After-inner node is in outer loop body only, should have outer loop iteration
    assert "_trace_after_inner_iter_0" in result  # after A's inner loop
    assert "_trace_after_inner_iter_1" in result  # after B's inner loop

    # Done node is outside both loops
    assert "_trace_done" in result


@pytest.mark.asyncio
async def test_trace_structure_completeness() -> None:
    """Test that trace data contains all required fields."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "test_node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "test_value"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "test_node", "type": "trigger"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    result = await compiled.ainvoke(
        config=None, context=None,
        trigger={"trigger_id": "test_trigger", "trigger_key": "test.trigger"}
    )

    trace = result["_trace_test_node"]

    # Verify all required trace fields are present
    assert "node_id" in trace
    assert "node_type" in trace
    assert "inputs" in trace
    assert "output" in trace
    assert "timestamp" in trace
    assert "status" in trace

    # Verify field values
    assert trace["node_id"] == "test_node"
    assert trace["node_type"] == "tool"
    assert trace["inputs"] == {"value": "test_value"}
    assert trace["output"] == "test_value"
    assert trace["status"] == "succeeded"


# =============================================================================
# MIXED BRANCH TRACE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_for_each_mixed_branches_all_have_traces() -> None:
    """Test that both true and false branches in a loop generate correct traces."""
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
                "id": "check",
                "type": "if",
                "condition": "${index} % 2 == 0",  # Even indices -> path_a, odd -> path_b
            },
            {
                "id": "path_a",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "a"},
            },
            {
                "id": "path_b",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "b"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "end"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "check", "type": "loop_body"},
            {"source": "check", "target": "path_a", "type": "conditional_true"},
            {"source": "check", "target": "path_b", "type": "conditional_false"},
            # NOTE: No explicit back-edges - implicit edges are added by the compiler
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["w", "x", "y", "z"]  # indices 0, 1, 2, 3
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # path_a executed in iterations 0 and 2 (even indices)
    assert "_trace_path_a_iter_0" in result
    assert "_trace_path_a_iter_2" in result

    # path_b executed in iterations 1 and 3 (odd indices)
    assert "_trace_path_b_iter_1" in result
    assert "_trace_path_b_iter_3" in result

    # Verify no trace collision (path_a/path_b without iteration suffix shouldn't exist)
    assert "_trace_path_a" not in result
    assert "_trace_path_b" not in result

    # Each trace should have correct output
    assert result["_trace_path_a_iter_0"]["output"] == "a"
    assert result["_trace_path_b_iter_1"]["output"] == "b"
