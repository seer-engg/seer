# pylint: disable=too-many-lines,unused-argument
# Reason: Comprehensive test coverage for if-node conditionals requires many test cases; mock functions have required signatures
"""
Comprehensive tests for if-node conditional execution in the workflow compiler.

Tests cover:
- True branch execution
- False branch execution (CRITICAL: previously untested)
- Both branches converging to same node
- Nested conditions
- Edge cases (falsy values, null conditions)
- Branch-to-END scenarios

This test file addresses the gap identified in the Code Review & Test Gap Analysis
where if/else execution, especially the false branch, had ~40% coverage.
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

pytestmark = pytest.mark.unit


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
# BASIC IF BRANCH EXECUTION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_if_true_branch_execution() -> None:
    """Test that the true branch executes when condition is true."""
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
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "true_executed"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "false_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
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

    # Verify true branch executed
    assert result["true_branch"] == "true_executed"
    # Verify condition result was stored
    assert result["_if_result_check_condition"] is True


@pytest.mark.asyncio
async def test_if_false_branch_execution() -> None:
    """CRITICAL TEST: Verify false branch executes when condition is false.

    This was previously untested and identified as a critical gap in coverage.
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
                        "flag": {"type": "boolean"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "true_executed"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "false_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": False  # CRITICAL: Testing false branch
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify false branch executed
    assert result["false_branch"] == "false_executed"
    # Verify condition result was stored
    assert result["_if_result_check_condition"] is False


@pytest.mark.asyncio
async def test_if_both_branches_converge_to_same_node() -> None:
    """Test that both branches can converge to a common continuation node."""
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
                        "flag": {"type": "boolean"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "true_path"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "false_path"},
            },
            {
                "id": "merge_point",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "merged"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
            {"source": "true_branch", "target": "merge_point", "type": "default"},
            {"source": "false_branch", "target": "merge_point", "type": "default"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test true path convergence
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": True
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["merge_point"] == "merged"
    assert "true_path" in call_tracker
    assert "merged" in call_tracker
    assert "false_path" not in call_tracker

    # Test false path convergence
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": False
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["merge_point"] == "merged"
    assert "false_path" in call_tracker
    assert "merged" in call_tracker
    assert "true_path" not in call_tracker


@pytest.mark.asyncio
async def test_if_true_only_branch() -> None:
    """Test if node with only true branch defined (false goes to END)."""
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
                        "flag": {"type": "boolean"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "true_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            # No false branch edge - should go to END
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test true path
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": True
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["true_branch"] == "true_executed"
    assert "true_executed" in call_tracker

    # Test false path (should go to END without executing true_branch)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": False
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Workflow should complete without executing true_branch
    assert "true_executed" not in call_tracker


@pytest.mark.asyncio
async def test_if_false_only_branch() -> None:
    """Test if node with only false branch defined (true goes to END)."""
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
                        "flag": {"type": "boolean"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.flag}",
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "false_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            # No true branch edge - should go to END
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test false path
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": False
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["false_branch"] == "false_executed"
    assert "false_executed" in call_tracker

    # Test true path (should go to END without executing false_branch)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "flag": True
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Workflow should complete without executing false_branch
    assert "false_executed" not in call_tracker


# =============================================================================
# NESTED CONDITIONAL TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_if_nested_conditions() -> None:
    """Test nested if nodes (if within if)."""
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
                        "outer_flag": {"type": "boolean"},
                        "inner_flag": {"type": "boolean"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "outer_check",
                "type": "if",
                "condition": "${test_trigger.outer_flag}",
            },
            {
                "id": "inner_check",
                "type": "if",
                "condition": "${test_trigger.inner_flag}",
            },
            {
                "id": "inner_true",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "inner_true_executed"},
            },
            {
                "id": "inner_false",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "inner_false_executed"},
            },
            {
                "id": "outer_false",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "outer_false_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "outer_check", "type": "trigger"},
            {"source": "outer_check", "target": "inner_check", "type": "conditional_true"},
            {"source": "outer_check", "target": "outer_false", "type": "conditional_false"},
            {"source": "inner_check", "target": "inner_true", "type": "conditional_true"},
            {"source": "inner_check", "target": "inner_false", "type": "conditional_false"},
        ],
    }

    compiled = await _compile_workflow(spec, [tracking_tool])

    # Test outer=true, inner=true
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_flag": True,
        "inner_flag": True
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["inner_true"] == "inner_true_executed"
    assert "inner_true_executed" in call_tracker
    assert "inner_false_executed" not in call_tracker
    assert "outer_false_executed" not in call_tracker

    # Test outer=true, inner=false
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_flag": True,
        "inner_flag": False
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["inner_false"] == "inner_false_executed"
    assert "inner_false_executed" in call_tracker
    assert "inner_true_executed" not in call_tracker
    assert "outer_false_executed" not in call_tracker

    # Test outer=false (should skip inner check entirely)
    call_tracker.clear()
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "outer_flag": False,
        "inner_flag": True  # This shouldn't matter
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["outer_false"] == "outer_false_executed"
    assert "outer_false_executed" in call_tracker
    assert "inner_true_executed" not in call_tracker
    assert "inner_false_executed" not in call_tracker


# =============================================================================
# EDGE CASE TESTS: FALSY VALUES
# =============================================================================


@pytest.mark.asyncio
async def test_if_condition_with_falsy_zero() -> None:
    """Test that numeric 0 is treated as falsy."""
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
                        "count": {"type": "number"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.count}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "has_count"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "no_count"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test count = 0 (falsy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "count": 0
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["false_branch"] == "no_count"

    # Test count = 5 (truthy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "count": 5
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["true_branch"] == "has_count"


@pytest.mark.asyncio
async def test_if_condition_with_empty_string() -> None:
    """Test that empty string is treated as falsy."""
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
                        "text": {"type": "string"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.text}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "has_text"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "no_text"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test empty string (falsy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "text": ""
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["false_branch"] == "no_text"

    # Test non-empty string (truthy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "text": "hello"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["true_branch"] == "has_text"


@pytest.mark.asyncio
async def test_if_condition_with_empty_array() -> None:
    """Test that empty array is treated as falsy."""
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
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.items}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "has_items"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "no_items"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test empty array (falsy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": []
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["false_branch"] == "no_items"

    # Test non-empty array (truthy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "items": ["a", "b"]
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["true_branch"] == "has_items"


@pytest.mark.asyncio
async def test_if_condition_with_null() -> None:
    """Test that null value is treated as falsy."""
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
                        "data": {"type": ["object", "null"]}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": "${test_trigger.data}",
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "has_data"},
            },
            {
                "id": "false_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "no_data"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
            {"source": "check_condition", "target": "false_branch", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test null (falsy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "data": None
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["false_branch"] == "no_data"

    # Test non-null object (truthy)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "data": {"key": "value"}
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["true_branch"] == "has_data"


# =============================================================================
# CONDITIONAL EXPRESSION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_if_condition_with_comparison_expression() -> None:
    """Test if node with comparison expression in condition."""
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
                        "score": {"type": "number"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_score",
                "type": "if",
                "condition": "${test_trigger.score} >= 50",
            },
            {
                "id": "pass",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "passed"},
            },
            {
                "id": "fail",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "failed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_score", "type": "trigger"},
            {"source": "check_score", "target": "pass", "type": "conditional_true"},
            {"source": "check_score", "target": "fail", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test passing score
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "score": 75
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["pass"] == "passed"

    # Test failing score
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "score": 25
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["fail"] == "failed"

    # Test edge case (exactly 50)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "score": 50
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["pass"] == "passed"


@pytest.mark.asyncio
async def test_if_condition_with_equality_expression() -> None:
    """Test if node with equality expression in condition."""
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
                        "status": {"type": "string"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_status",
                "type": "if",
                "condition": "${test_trigger.status} == 'active'",
            },
            {
                "id": "handle_active",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "is_active"},
            },
            {
                "id": "handle_inactive",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "is_inactive"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_status", "type": "trigger"},
            {"source": "check_status", "target": "handle_active", "type": "conditional_true"},
            {"source": "check_status", "target": "handle_inactive", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test active status
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "status": "active"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["handle_active"] == "is_active"

    # Test inactive status
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "status": "inactive"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["handle_inactive"] == "is_inactive"


@pytest.mark.asyncio
async def test_if_condition_referencing_previous_node_output() -> None:
    """Test if node that references output from a previous node."""
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
                        "input_value": {"type": "string"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${test_trigger.input_value}"},
            },
            {
                "id": "check_result",
                "type": "if",
                "condition": "${process} == 'success'",
            },
            {
                "id": "success_handler",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "handled_success"},
            },
            {
                "id": "failure_handler",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "handled_failure"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
            {"source": "process", "target": "check_result", "type": "default"},
            {"source": "check_result", "target": "success_handler", "type": "conditional_true"},
            {"source": "check_result", "target": "failure_handler", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test success case
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "input_value": "success"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["success_handler"] == "handled_success"

    # Test failure case
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "input_value": "error"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["failure_handler"] == "handled_failure"


# =============================================================================
# CHAIN OF CONDITIONALS TEST
# =============================================================================


@pytest.mark.asyncio
async def test_if_chain_multiple_conditions() -> None:
    """Test a chain of sequential if nodes (if-else-if pattern)."""
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
                        "priority": {"type": "string"}
                    }
                },
            }
        ],
        "nodes": [
            {
                "id": "check_high",
                "type": "if",
                "condition": "${test_trigger.priority} == 'high'",
            },
            {
                "id": "handle_high",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "high_priority"},
            },
            {
                "id": "check_medium",
                "type": "if",
                "condition": "${test_trigger.priority} == 'medium'",
            },
            {
                "id": "handle_medium",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "medium_priority"},
            },
            {
                "id": "handle_low",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "low_priority"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_high", "type": "trigger"},
            {"source": "check_high", "target": "handle_high", "type": "conditional_true"},
            {"source": "check_high", "target": "check_medium", "type": "conditional_false"},
            {"source": "check_medium", "target": "handle_medium", "type": "conditional_true"},
            {"source": "check_medium", "target": "handle_low", "type": "conditional_false"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow(spec, [mock_tool])

    # Test high priority
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "priority": "high"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["handle_high"] == "high_priority"

    # Test medium priority
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "priority": "medium"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["handle_medium"] == "medium_priority"

    # Test low priority (default case)
    trigger_envelope = {
        "trigger_id": "test_trigger",
        "trigger_key": "test.trigger",
        "priority": "low"
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)
    assert result["handle_low"] == "low_priority"
