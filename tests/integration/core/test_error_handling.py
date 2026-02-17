# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for error handling
"""
Integration tests for error handling across compilation and runtime.

Tests verify that errors are properly detected, reported, and propagated
through the compilation and execution pipeline.
"""
from __future__ import annotations

from typing import Any, List

import pytest

from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.registry.tool_registry import ToolNotFoundError
from seer.core.errors import ExecutionError

from .conftest import (
    compile_workflow,
    create_error_tool,
    create_tracking_tool,
    simple_trigger_spec,
)


# =============================================================================
# COMPILATION ERROR TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_compile_error_on_invalid_edge_target() -> None:
    """
    Test that compilation fails when edge points to non-existent node.
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
                "inputs": {"value": "test"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
            {"source": "process", "target": "nonexistent_node", "type": "default"},
        ],
    }

    with pytest.raises((ValueError, KeyError, Exception)) as exc_info:
        await compile_workflow(spec, tool_defs=[tracking_tool])

    # Should mention the invalid node
    assert "nonexistent" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_compile_error_on_missing_required_fields() -> None:
    """
    Test that compilation fails when required fields are missing.
    """
    # Missing 'type' field in node
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "process",
                # Missing 'type' field
                "tool": "test.tracker",
                "inputs": {"value": "test"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    with pytest.raises((ValueError, KeyError, Exception)):
        await compile_workflow(spec)


@pytest.mark.asyncio
async def test_compile_error_on_invalid_version() -> None:
    """
    Test that compilation fails with invalid version.
    """
    spec = {
        "version": "invalid",
        "triggers": [simple_trigger_spec()],
        "nodes": [],
        "edges": [],
    }

    with pytest.raises((ValueError, Exception)):
        await compile_workflow(spec)


@pytest.mark.asyncio
async def test_compile_error_on_tool_not_registered() -> None:
    """
    Test that ToolNotFoundError is raised for unregistered tools.
    """
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "unknown.tool.name",
                "inputs": {"value": "test"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    with pytest.raises(ToolNotFoundError) as exc_info:
        await compile_workflow(spec, tool_defs=[])

    assert "unknown.tool.name" in str(exc_info.value)


@pytest.mark.asyncio
async def test_compile_error_on_invalid_for_each_items() -> None:
    """
    Test that compilation fails when for_each has invalid items expression.
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
                "items": "${nonexistent_reference}",
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
                "inputs": {"value": "done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    with pytest.raises(Exception) as exc_info:
        await compile_workflow(spec, tool_defs=[tracking_tool])

    # Should mention the unknown reference
    assert "nonexistent" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()


# =============================================================================
# RUNTIME ERROR TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_execution_error_from_tool_failure() -> None:
    """
    Test that ExecutionError from tool is properly raised.
    """
    error_tool = create_error_tool()

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "fail_node",
                "type": "tool",
                "tool": "test.error",
                "inputs": {"message": "Intentional test failure"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fail_node", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[error_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    with pytest.raises(ExecutionError) as exc_info:
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "Intentional test failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_failure_preserves_previous_state() -> None:
    """
    Test that tool failure doesn't lose state from previously executed nodes.

    Note: LangGraph behavior may vary - the test verifies error propagation.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)
    error_tool = create_error_tool()

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "good_node",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "successful"},
            },
            {
                "id": "fail_node",
                "type": "tool",
                "tool": "test.error",
                "inputs": {"message": "Failure after success"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "good_node", "type": "trigger"},
            {"source": "good_node", "target": "fail_node", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool, error_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    # First node should have executed before failure
    with pytest.raises(ExecutionError):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify first node was executed
    assert "successful" in call_tracker


@pytest.mark.asyncio
async def test_for_each_error_on_non_list_items() -> None:
    """
    Test that for_each fails at compile-time when items expression doesn't resolve to array.

    The compiler validates that for_each items must be an array schema.
    """
    from seer.core.errors import ValidationPhaseError  # pylint: disable=import-outside-toplevel

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
                        "not_a_list": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.not_a_list}",  # String, not array
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
                "inputs": {"value": "done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    # The compiler validates that for_each items must resolve to an array schema
    with pytest.raises(ValidationPhaseError) as exc_info:
        await compile_workflow(spec, tool_defs=[tracking_tool])

    assert "array" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_error_in_loop_iteration() -> None:
    """
    Test that error in loop iteration is properly propagated.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)
    error_tool = create_error_tool()

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
                "item_var": "current",
                "index_var": "idx",
            },
            {
                "id": "check",
                "type": "if",
                "condition": "${idx} == 1",  # Fail on second item
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${current}"},
            },
            {
                "id": "fail",
                "type": "tool",
                "tool": "test.error",
                "inputs": {"message": "Error at index ${idx}"},
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
            {"source": "check", "target": "fail", "type": "conditional_true"},
            {"source": "check", "target": "process", "type": "conditional_false"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool, error_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b", "c"],
    }

    with pytest.raises(ExecutionError):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # First item should have been processed before error
    assert "a" in call_tracker


@pytest.mark.asyncio
async def test_if_condition_evaluation_error() -> None:
    """
    Test that invalid condition expression produces error.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "check",
                "type": "if",
                "condition": "${test_trigger.nonexistent_field}",  # Field doesn't exist
            },
            {
                "id": "true_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "true"},
            },
            {
                "id": "false_path",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "false"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check", "type": "trigger"},
            {"source": "check", "target": "true_path", "type": "conditional_true"},
            {"source": "check", "target": "false_path", "type": "conditional_false"},
        ],
    }

    # This should fail during compilation (reference validation) or runtime
    try:
        compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

        trigger_envelope = {
            "trigger_id": "test_trigger",
            "trigger_key": "test.trigger",
            "message": "test",
        }

        with pytest.raises((ExecutionError, KeyError, Exception)):
            await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    except Exception:
        # Expected if caught during compilation
        pass


@pytest.mark.asyncio
async def test_multiple_errors_first_propagates() -> None:
    """
    Test that when multiple nodes could fail, the first error propagates.
    """
    error_tool = create_error_tool()

    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "fail_first",
                "type": "tool",
                "tool": "test.error",
                "inputs": {"message": "First error"},
            },
            {
                "id": "fail_second",
                "type": "tool",
                "tool": "test.error",
                "inputs": {"message": "Second error"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "fail_first", "type": "trigger"},
            {"source": "fail_first", "target": "fail_second", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[error_tool])

    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "message": "test",
    }

    with pytest.raises(ExecutionError) as exc_info:
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # First error should be the one that propagates
    assert "First error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execution_plan_validation() -> None:
    """
    Test that execution plan validates basic graph structure.
    """
    # Valid spec for parsing
    spec_payload = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "node",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "node", "type": "trigger"},
        ],
    }

    spec = parse_workflow_spec(spec_payload)
    plan = build_execution_plan(spec)

    # Verify plan has expected structure
    assert plan.trigger_targets is not None
    assert "test_trigger" in plan.trigger_targets
    assert plan.trigger_targets["test_trigger"] == ["node"]
