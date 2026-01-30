# pylint: disable=unused-argument
# Reason: Mock handlers require specific function signatures even if not all params are used
"""
Tests for IfNode to SwitchNode conversion.

Tests cover:
- Automatic conversion of IfNode to SwitchNode at compile time
- Backward compatibility for existing workflows with IfNode
- Deprecation warnings
- Conversion correctness (edges, condition, execution)
"""
from __future__ import annotations

import warnings

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
from seer.core.schema.models import IfNode
from seer.core.schema.schema_registry import SchemaRegistry


async def _compile_workflow(spec_payload: dict, tool_registry: ToolRegistry) -> CompiledWorkflow:
    """Helper to compile a workflow from a spec dictionary."""
    schema_registry = SchemaRegistry()
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


def _create_echo_tool() -> ToolDefinition:
    """Create a mock test.echo tool that returns its input as output."""
    def handler(inputs, config, context):
        return {"result": inputs.get("message", "no message")}

    async def async_handler(inputs, config, context):
        return {"result": inputs.get("message", "no message")}

    return ToolDefinition(
        name="test.echo",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "additionalProperties": False,
        },
        handler=handler,
        async_handler=async_handler,
    )


# =============================================================================
# Deprecation Warning Tests
# =============================================================================

def test_if_node_deprecation_warning():
    """Test that IfNode emits deprecation warning when instantiated."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create IfNode - should emit deprecation warning
        node = IfNode(
            id="if_1",
            type="if",
            condition="${x} > 10"
        )

        # Verify warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "IfNode is deprecated" in str(w[0].message)
        assert "Use SwitchNode instead" in str(w[0].message)
        assert node.id == "if_1"


# =============================================================================
# Conversion Tests
# =============================================================================

@pytest.mark.asyncio
async def test_if_node_converted_to_switch():
    """Test that IfNode is automatically converted to SwitchNode during compilation."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    # Workflow uses deprecated IfNode
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.webhook",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"success": {"type": "boolean"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "if_1",
                "type": "if",
                "condition": "${trigger.event.success}"
            },
            {"id": "task_true", "type": "tool", "tool": "test.echo", "inputs": {"message": "Success"}},
            {"id": "task_false", "type": "tool", "tool": "test.echo", "inputs": {"message": "Failure"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "if_1", "type": "trigger"},
            {"source": "if_1", "target": "task_true", "type": "conditional_true"},
            {"source": "if_1", "target": "task_false", "type": "conditional_false"},
        ]
    }

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    # Test true branch
    result = await compiled.run(trigger_id="trigger_1", event={"success": True})
    assert result["task_true"]["result"] == "Success"
    assert "task_false" not in result

    # Test false branch
    result = await compiled.run(trigger_id="trigger_1", event={"success": False})
    assert result["task_false"]["result"] == "Failure"
    assert "task_true" not in result


@pytest.mark.asyncio
async def test_if_node_backward_compatibility():
    """Test that existing workflows with IfNode continue to work correctly."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    # This is a typical V2 workflow with IfNode
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.webhook",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"count": {"type": "number"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "if_1",
                "type": "if",
                "condition": "${trigger.event.count} > 100"
            },
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High count"}},
            {"id": "task_low", "type": "tool", "tool": "test.echo", "inputs": {"message": "Low count"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "if_1", "type": "trigger"},
            {"source": "if_1", "target": "task_high", "type": "conditional_true"},
            {"source": "if_1", "target": "task_low", "type": "conditional_false"},
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    # High count
    result = await compiled.run(trigger_id="trigger_1", event={"count": 200})
    assert result["task_high"]["result"] == "High count"
    assert "task_low" not in result

    # Low count
    result = await compiled.run(trigger_id="trigger_1", event={"count": 50})
    assert result["task_low"]["result"] == "Low count"
    assert "task_high" not in result


@pytest.mark.asyncio
async def test_if_node_complex_condition():
    """Test that complex conditions in IfNode are preserved during conversion."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.webhook",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "priority": {"type": "number"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "if_1",
                "type": "if",
                "condition": "${trigger.event.status} == 'active' and ${trigger.event.priority} > 5"
            },
            {"id": "task_proceed", "type": "tool", "tool": "test.echo", "inputs": {"message": "Proceed"}},
            {"id": "task_skip", "type": "tool", "tool": "test.echo", "inputs": {"message": "Skip"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "if_1", "type": "trigger"},
            {"source": "if_1", "target": "task_proceed", "type": "conditional_true"},
            {"source": "if_1", "target": "task_skip", "type": "conditional_false"},
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    # Both conditions true
    result = await compiled.run(trigger_id="trigger_1", event={"status": "active", "priority": 10})
    assert result["task_proceed"]["result"] == "Proceed"

    # status false, priority true
    result = await compiled.run(trigger_id="trigger_1", event={"status": "inactive", "priority": 10})
    assert result["task_skip"]["result"] == "Skip"

    # status true, priority false
    result = await compiled.run(trigger_id="trigger_1", event={"status": "active", "priority": 3})
    assert result["task_skip"]["result"] == "Skip"


@pytest.mark.asyncio
async def test_if_node_with_upstream_task():
    """Test IfNode with upstream task output (common pattern)."""
    tool_registry = ToolRegistry()

    def status_tool_handler(inputs, config, context):
        return {"success": inputs.get("succeed", False)}

    tool_registry.register(ToolDefinition(
        name="test.status",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"succeed": {"type": "boolean"}},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"success": {"type": "boolean"}},
            "additionalProperties": False,
        },
        handler=status_tool_handler,
        async_handler=lambda i, c, ctx: status_tool_handler(i, c, ctx),
    ))
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {"id": "trigger_1", "key": "test.webhook", "mode": "webhook", "schemas": {"event": {"type": "object"}}}
        ],
        "nodes": [
            {"id": "task_1", "type": "tool", "tool": "test.status", "inputs": {"succeed": True}},
            {
                "id": "if_1",
                "type": "if",
                "condition": "${task_1.success}"
            },
            {"id": "task_success", "type": "tool", "tool": "test.echo", "inputs": {"message": "Success"}},
            {"id": "task_failure", "type": "tool", "tool": "test.echo", "inputs": {"message": "Failure"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "task_1", "type": "trigger"},
            {"source": "task_1", "target": "if_1", "type": "default"},
            {"source": "if_1", "target": "task_success", "type": "conditional_true"},
            {"source": "if_1", "target": "task_failure", "type": "conditional_false"},
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    result = await compiled.run(trigger_id="trigger_1", event={})
    assert result["task_1"]["success"] is True
    assert result["task_success"]["result"] == "Success"
    assert "task_failure" not in result


@pytest.mark.asyncio
async def test_if_node_only_true_branch():
    """Test IfNode with only true branch (no false branch)."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.webhook",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"proceed": {"type": "boolean"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "if_1",
                "type": "if",
                "condition": "${trigger.event.proceed}"
            },
            {"id": "task_proceed", "type": "tool", "tool": "test.echo", "inputs": {"message": "Proceed"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "if_1", "type": "trigger"},
            {"source": "if_1", "target": "task_proceed", "type": "conditional_true"},
            # No false branch!
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    # True case
    result = await compiled.run(trigger_id="trigger_1", event={"proceed": True})
    assert result["task_proceed"]["result"] == "Proceed"

    # False case - workflow ends (no false branch)
    result = await compiled.run(trigger_id="trigger_1", event={"proceed": False})
    assert "task_proceed" not in result


@pytest.mark.asyncio
async def test_multiple_if_nodes_in_workflow():
    """Test workflow with multiple IfNodes (all get converted)."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.webhook",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "check1": {"type": "boolean"},
                            "check2": {"type": "boolean"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {"id": "if_1", "type": "if", "condition": "${trigger.event.check1}"},
            {"id": "if_2", "type": "if", "condition": "${trigger.event.check2}"},
            {"id": "task_both", "type": "tool", "tool": "test.echo", "inputs": {"message": "Both true"}},
            {"id": "task_first", "type": "tool", "tool": "test.echo", "inputs": {"message": "First only"}},
            {"id": "task_none", "type": "tool", "tool": "test.echo", "inputs": {"message": "None"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "if_1", "type": "trigger"},
            {"source": "if_1", "target": "if_2", "type": "conditional_true"},
            {"source": "if_1", "target": "task_none", "type": "conditional_false"},
            {"source": "if_2", "target": "task_both", "type": "conditional_true"},
            {"source": "if_2", "target": "task_first", "type": "conditional_false"},
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compiled = await _compile_workflow(spec, tool_registry)

    # Both true
    result = await compiled.run(trigger_id="trigger_1", event={"check1": True, "check2": True})
    assert result["task_both"]["result"] == "Both true"

    # First true, second false
    result = await compiled.run(trigger_id="trigger_1", event={"check1": True, "check2": False})
    assert result["task_first"]["result"] == "First only"

    # First false
    result = await compiled.run(trigger_id="trigger_1", event={"check1": False, "check2": True})
    assert result["task_none"]["result"] == "None"


# =============================================================================
# Conversion Correctness Tests
# =============================================================================

def test_conversion_preserves_node_id():
    """Test that conversion preserves the original node ID."""
    from seer.core.compiler.lower_control_flow import _convert_if_to_switch
    from seer.core.schema.models import Edge, EdgeType

    if_node = IfNode(id="my_if_node", type="if", condition="${x} > 10")
    edges = [
        Edge(source="my_if_node", target="task_true", type=EdgeType.conditional_true),
        Edge(source="my_if_node", target="task_false", type=EdgeType.conditional_false),
    ]

    switch_node, new_edges = _convert_if_to_switch(if_node, edges)

    # Node ID should be preserved
    assert switch_node.id == "my_if_node"
    # Should have exactly one case with the original condition
    assert len(switch_node.cases) == 1
    assert switch_node.cases[0].condition == "${x} > 10"
    assert switch_node.cases[0].label == "__if_true"


def test_conversion_edge_types():
    """Test that edge types are correctly converted."""
    from seer.core.compiler.lower_control_flow import _convert_if_to_switch
    from seer.core.schema.models import Edge, EdgeType

    if_node = IfNode(id="if_1", type="if", condition="${x}")
    edges = [
        Edge(source="if_1", target="task_true", type=EdgeType.conditional_true),
        Edge(source="if_1", target="task_false", type=EdgeType.conditional_false),
        Edge(source="other_node", target="if_1", type=EdgeType.default),  # Unrelated edge
    ]

    switch_node, new_edges = _convert_if_to_switch(if_node, edges)

    # Find converted edges
    true_edge = next(e for e in new_edges if e.target == "task_true")
    false_edge = next(e for e in new_edges if e.target == "task_false")
    unrelated_edge = next(e for e in new_edges if e.source == "other_node")

    # True edge should become switch_case with route="__if_true"
    assert true_edge.type == EdgeType.switch_case
    assert true_edge.route == "__if_true"

    # False edge should become switch_default
    assert false_edge.type == EdgeType.switch_default
    assert false_edge.route is None

    # Unrelated edge should be unchanged
    assert unrelated_edge.type == EdgeType.default


def test_conversion_preserves_ui_metadata():
    """Test that UI metadata is preserved during conversion."""
    from seer.core.compiler.lower_control_flow import _convert_if_to_switch

    if_node = IfNode(
        id="if_1",
        type="if",
        condition="${x}",
        ui={"x": 100, "y": 200, "label": "My If Node"}
    )

    switch_node, _ = _convert_if_to_switch(if_node, [])

    # UI metadata should be preserved
    assert switch_node.ui == {"x": 100, "y": 200, "label": "My If Node"}
