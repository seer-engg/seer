# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for compilation pipeline
"""
Integration tests for the workflow compilation pipeline.

Tests verify that all 5 stages of compilation work together correctly:
1. Parse - JSON payload to WorkflowSpec
2. Type Environment - Build type information
3. Reference Validation - Verify ${...} expressions
4. Control Flow Lowering - Build ExecutionPlan
5. LangGraph Emission - Create executable graph
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from seer.core.compiler.context import CompilerContext
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.model_registry import ModelDefinition
from seer.core.registry.tool_registry import ToolDefinition, ToolNotFoundError

from .conftest import (
    compile_workflow,
    create_mock_json_llm_handler,
    create_mock_text_llm_handler,
    create_tracking_tool,
    simple_trigger_spec,
)


# =============================================================================
# END-TO-END COMPILATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_compile_simple_workflow_with_real_registries() -> None:
    """
    Test full compilation pipeline with real registries.

    Verifies that a simple workflow compiles successfully through all 5 stages
    and produces a valid CompiledWorkflow ready for execution.
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
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Verify compilation produced valid structure
    assert compiled.spec is not None
    assert compiled.type_env is not None
    assert compiled.graph is not None
    assert compiled.runtime is not None

    # Verify type environment has expected entries
    assert "test_trigger" in compiled.type_env
    assert "process" in compiled.type_env


@pytest.mark.asyncio
async def test_compile_workflow_with_multiple_node_types() -> None:
    """
    Test compilation of workflow with Tool + If + ForEach nodes.

    Verifies that different node types compile together correctly
    with proper type environment entries for each.
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
                        "items": {"type": "array", "items": {"type": "string"}},
                        "flag": {"type": "boolean"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "check_flag",
                "type": "if",
                "condition": "${test_trigger.flag}",
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
                "inputs": {"value": "${item}"},
            },
            {
                "id": "skip",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "skipped"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "done"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_flag", "type": "trigger"},
            {"source": "check_flag", "target": "loop", "type": "conditional_true"},
            {"source": "check_flag", "target": "skip", "type": "conditional_false"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Verify tool nodes are in type environment
    # Note: Control flow nodes (if, for_each) don't produce outputs, so they're not in type_env
    assert "process_item" in compiled.type_env
    assert "skip" in compiled.type_env
    assert "done" in compiled.type_env
    # Loop node is in type_env as it represents the array being iterated
    assert "loop" in compiled.type_env


@pytest.mark.asyncio
async def test_compile_workflow_generates_correct_type_environment() -> None:
    """
    Test that type environment correctly captures node output types.

    The type environment should have JSON schema for each node's output,
    enabling reference validation and downstream type checking.
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
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Verify trigger type is captured from event_schema
    trigger_type = compiled.type_env.get("test_trigger")
    assert trigger_type is not None
    assert trigger_type.get("type") == "object"
    assert "properties" in trigger_type

    # Verify tool output type matches tool's output_schema
    process_type = compiled.type_env.get("process")
    assert process_type is not None


@pytest.mark.asyncio
async def test_compile_workflow_resolves_tool_schemas() -> None:
    """
    Test that tool output schemas are correctly resolved from registry.

    The compiler should look up tool definitions and use their output_schema
    for type environment building.
    """
    # Create tool with specific output schema
    def handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {"status": "ok", "count": 42}

    async def async_handler(inputs: Dict[str, Any], config: Any, context: Any) -> Dict[str, Any]:
        return {"status": "ok", "count": 42}

    custom_tool = ToolDefinition(
        name="test.custom",
        version="v1",
        input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
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
                "id": "process",
                "type": "tool",
                "tool": "test.custom",
                "inputs": {"input": "${test_trigger.message}"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[custom_tool])

    # Verify type environment has tool's output schema
    process_type = compiled.type_env.get("process")
    assert process_type is not None
    assert process_type.get("type") == "object"
    assert "properties" in process_type
    assert "status" in process_type["properties"]


@pytest.mark.asyncio
async def test_compile_workflow_builds_correct_execution_plan() -> None:
    """
    Test that ExecutionPlan has correct edges, entry_node, and trigger_targets.

    The execution plan should correctly map the workflow graph structure
    for LangGraph emission.
    """
    spec_payload = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_a",
                "key": "test.a",
                "mode": "webhook",
                "event_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
            },
            {
                "id": "trigger_b",
                "key": "test.b",
                "mode": "webhook",
                "event_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
            },
        ],
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
            {"source": "trigger_a", "target": "node_a", "type": "trigger"},
            {"source": "trigger_b", "target": "node_b", "type": "trigger"},
        ],
    }

    spec = parse_workflow_spec(spec_payload)
    plan = build_execution_plan(spec)

    # Verify trigger_targets maps each trigger to its entry node
    assert "trigger_a" in plan.trigger_targets
    assert "trigger_b" in plan.trigger_targets
    assert plan.trigger_targets["trigger_a"] == ["node_a"]
    assert plan.trigger_targets["trigger_b"] == ["node_b"]


@pytest.mark.asyncio
async def test_compile_workflow_detects_loop_body_nodes() -> None:
    """
    Test that loop body detection correctly identifies nested nodes.

    All nodes reachable within a loop body should be marked as loop_body_nodes
    for correct iteration-scoped trace key generation.
    """
    spec_payload = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${test_trigger.items}",
            },
            {
                "id": "step_a",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_a"},
            },
            {
                "id": "step_b",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${item}_b"},
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

    spec = parse_workflow_spec(spec_payload)
    plan = build_execution_plan(spec)

    # Verify loop body nodes are detected
    # loop_body_nodes is a dict mapping loop_id to set of body nodes
    assert "loop" in plan.loop_body_nodes
    assert "step_a" in plan.loop_body_nodes["loop"]
    assert "step_b" in plan.loop_body_nodes["loop"]
    # done is after loop exit, not in body
    assert "done" not in plan.loop_body_nodes.get("loop", set())


# =============================================================================
# COMPILATION ERROR TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_compile_fails_on_unregistered_tool() -> None:
    """
    Test that compilation fails with ToolNotFoundError for unknown tools.

    The compiler should fail early during type environment building
    when a tool is not registered.
    """
    spec = {
        "version": "2",
        "triggers": [simple_trigger_spec()],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "nonexistent.tool",
                "inputs": {"value": "test"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    with pytest.raises(ToolNotFoundError) as exc_info:
        await compile_workflow(spec, tool_defs=[])

    assert "nonexistent.tool" in str(exc_info.value)


@pytest.mark.asyncio
async def test_compile_fails_on_invalid_expression_reference() -> None:
    """
    Test that compilation fails on invalid ${...} references.

    Reference validation should catch references to non-existent nodes
    or properties.
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
                "inputs": {"value": "${nonexistent_node.field}"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    with pytest.raises(Exception) as exc_info:
        await compile_workflow(spec, tool_defs=[tracking_tool])

    # Should fail during reference validation
    assert "nonexistent_node" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_compile_fails_on_duplicate_node_ids() -> None:
    """
    Test that compilation fails when node IDs are duplicated.
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
                "inputs": {"value": "first"},
            },
            {
                "id": "process",  # Duplicate ID
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "second"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "process", "type": "trigger"},
        ],
    }

    with pytest.raises((ValueError, Exception)):
        await compile_workflow(spec, tool_defs=[tracking_tool])


@pytest.mark.asyncio
async def test_compile_with_multiple_triggers() -> None:
    """
    Test compilation with multiple triggers mapping to different entry nodes.

    Each trigger should correctly map to its target node in trigger_targets.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "gmail_trigger",
                "key": "gmail.email_received",
                "mode": "polling",
                "event_schema": {"type": "object", "properties": {"email": {"type": "object"}}},
            },
            {
                "id": "slack_trigger",
                "key": "slack.message_received",
                "mode": "polling",
                "event_schema": {"type": "object", "properties": {"message": {"type": "object"}}},
            },
        ],
        "nodes": [
            {
                "id": "handle_email",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "email"},
            },
            {
                "id": "handle_slack",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "slack"},
            },
        ],
        "edges": [
            {"source": "gmail_trigger", "target": "handle_email", "type": "trigger"},
            {"source": "slack_trigger", "target": "handle_slack", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Verify both triggers are in type environment
    assert "gmail_trigger" in compiled.type_env
    assert "slack_trigger" in compiled.type_env

    # Verify both handlers are compiled
    assert "handle_email" in compiled.type_env
    assert "handle_slack" in compiled.type_env


@pytest.mark.asyncio
async def test_compile_workflow_with_agent_node() -> None:
    """
    Test compilation of workflow with agent node (supersedes LLM node).

    Verifies that agent nodes compile correctly with model registry lookup.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=create_mock_text_llm_handler("response"),
        json_handler=create_mock_json_llm_handler({"result": "ok"}),
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
                    "prompt": "Process: ${test_trigger.message}",
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

    assert "generate" in compiled.type_env
    assert "process" in compiled.type_env


@pytest.mark.asyncio
async def test_compile_workflow_with_nested_control_flow() -> None:
    """
    Test compilation of deeply nested control flow structures.

    Verifies that nested loops and conditionals compile correctly
    with proper loop_body_nodes detection.
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
                "index_var": "outer_idx",
            },
            {
                "id": "check",
                "type": "if",
                "condition": "${outer_idx} % 2 == 0",
            },
            {
                "id": "inner_loop",
                "type": "for_each",
                "items": "${test_trigger.inner_items}",
                "item_var": "inner_item",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_${inner_item}"},
            },
            {
                "id": "skip",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${outer_item}_skipped"},
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
            {"source": "outer_loop", "target": "check", "type": "loop_body"},
            {"source": "check", "target": "inner_loop", "type": "conditional_true"},
            {"source": "check", "target": "skip", "type": "conditional_false"},
            {"source": "inner_loop", "target": "process", "type": "loop_body"},
            {"source": "inner_loop", "target": "after_inner", "type": "loop_exit"},
            {"source": "outer_loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Verify tool nodes are compiled (control flow nodes don't produce outputs)
    for node_id in ["process", "skip", "after_inner", "done"]:
        assert node_id in compiled.type_env
    # Loop nodes are in type_env as they represent the arrays being iterated
    assert "outer_loop" in compiled.type_env
    assert "inner_loop" in compiled.type_env


@pytest.mark.asyncio
async def test_compile_workflow_validates_condition_expressions() -> None:
    """
    Test that if conditions with valid expressions compile correctly.
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
                        "count": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "check_count",
                "type": "if",
                "condition": "${test_trigger.count} > 10",
            },
            {
                "id": "high",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "high"},
            },
            {
                "id": "low",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "low"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_count", "type": "trigger"},
            {"source": "check_count", "target": "high", "type": "conditional_true"},
            {"source": "check_count", "target": "low", "type": "conditional_false"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # If nodes (control flow) don't produce outputs, so not in type_env
    # But the compilation should succeed with valid expressions
    assert compiled.spec is not None
    assert "high" in compiled.type_env
    assert "low" in compiled.type_env
