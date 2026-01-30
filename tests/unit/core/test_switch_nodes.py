# pylint: disable=unused-argument
# Reason: Mock handlers require specific function signatures even if not all params are used
"""
Comprehensive tests for SwitchNode functionality.

Tests cover:
- Schema validation (duplicate labels, reserved keywords, unique labels)
- Basic multi-way conditional routing
- First-match-wins evaluation semantics
- Default case handling
- Case evaluation error handling
- Integration with other node types
- Edge cases
"""
from __future__ import annotations

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.errors import ValidationPhaseError
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.models import SwitchCase, SwitchNode
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
# Schema Validation Tests
# =============================================================================

def test_switch_node_duplicate_labels():
    """Test that duplicate case labels are rejected during validation."""
    with pytest.raises(ValueError, match="Duplicate case labels"):
        SwitchNode(
            id="switch_1",
            type="switch",
            cases=[
                SwitchCase(condition="${x} > 10", label="high"),
                SwitchCase(condition="${x} > 5", label="high"),  # Duplicate!
            ]
        )


def test_switch_node_reserved_default_label():
    """Test that 'default' is rejected as a case label (reserved)."""
    with pytest.raises(ValueError, match='Case label cannot be "default"'):
        SwitchCase(condition="${x} > 10", label="default")


def test_switch_node_empty_cases_list():
    """Test that switch node requires at least one case."""
    with pytest.raises(ValueError, match="at least 1 item"):
        SwitchNode(
            id="switch_1",
            type="switch",
            cases=[]
        )


def test_switch_node_invalid_label_pattern():
    """Test that invalid label patterns are rejected."""
    with pytest.raises(ValueError, match="does not match"):
        SwitchCase(condition="${x} > 10", label="invalid label!")  # Space and ! not allowed


def test_switch_node_valid_labels():
    """Test that valid label patterns are accepted."""
    # All of these should be valid
    node = SwitchNode(
        id="switch_1",
        type="switch",
        cases=[
            SwitchCase(condition="${x} > 10", label="high"),
            SwitchCase(condition="${x} > 5", label="medium-2"),
            SwitchCase(condition="${x} > 0", label="low_value"),
            SwitchCase(condition="${x} == 0", label="zero123"),
        ]
    )
    assert len(node.cases) == 4


def test_edge_route_field_validation():
    """Test that switch_case edges require route field."""
    from seer.core.schema.models import Edge, EdgeType

    # switch_case without route should fail
    with pytest.raises(ValueError, match="switch_case edges require a route label"):
        Edge(
            source="switch_1",
            target="task_1",
            type=EdgeType.switch_case,
            route=None  # Missing route!
        )

    # Non-switch_case with route should fail
    with pytest.raises(ValueError, match="route field only allowed for switch_case"):
        Edge(
            source="task_1",
            target="task_2",
            type=EdgeType.default,
            route="some_route"  # Should not have route!
        )

    # Valid switch_case with route
    edge = Edge(
        source="switch_1",
        target="task_1",
        type=EdgeType.switch_case,
        route="high"
    )
    assert edge.route == "high"


# =============================================================================
# Basic Routing Tests
# =============================================================================

@pytest.mark.asyncio
async def test_switch_basic_routing():
    """Test basic switch routing with multiple cases."""
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
                            "status": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.status} == 'success'", "label": "success"},
                    {"condition": "${trigger.event.status} == 'error'", "label": "error"},
                    {"condition": "${trigger.event.status} == 'pending'", "label": "pending"},
                ]
            },
            {"id": "task_success", "type": "tool", "tool": "test.echo", "inputs": {"message": "Success path"}},
            {"id": "task_error", "type": "tool", "tool": "test.echo", "inputs": {"message": "Error path"}},
            {"id": "task_pending", "type": "tool", "tool": "test.echo", "inputs": {"message": "Pending path"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_success", "type": "switch_case", "route": "success"},
            {"source": "switch_1", "target": "task_error", "type": "switch_case", "route": "error"},
            {"source": "switch_1", "target": "task_pending", "type": "switch_case", "route": "pending"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # Test success path
    result = await compiled.run(trigger_id="trigger_1", event={"status": "success"})
    assert result["task_success"]["result"] == "Success path"
    assert "task_error" not in result
    assert "task_pending" not in result

    # Test error path
    result = await compiled.run(trigger_id="trigger_1", event={"status": "error"})
    assert result["task_error"]["result"] == "Error path"
    assert "task_success" not in result

    # Test pending path
    result = await compiled.run(trigger_id="trigger_1", event={"status": "pending"})
    assert result["task_pending"]["result"] == "Pending path"
    assert "task_success" not in result


@pytest.mark.asyncio
async def test_switch_first_match_wins():
    """Test that only the first matching case is executed."""
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
                            "count": {"type": "number"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.count} > 50", "label": "very_high"},  # Matches
                    {"condition": "${trigger.event.count} > 10", "label": "high"},  # Would also match, but not evaluated
                    {"condition": "${trigger.event.count} > 0", "label": "low"},   # Would also match, but not evaluated
                ]
            },
            {"id": "task_very_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "Very high"}},
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
            {"id": "task_low", "type": "tool", "tool": "test.echo", "inputs": {"message": "Low"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_very_high", "type": "switch_case", "route": "very_high"},
            {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
            {"source": "switch_1", "target": "task_low", "type": "switch_case", "route": "low"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # Count=100: should match very_high (first), not high or low
    result = await compiled.run(trigger_id="trigger_1", event={"count": 100})
    assert result["task_very_high"]["result"] == "Very high"
    assert "task_high" not in result
    assert "task_low" not in result

    # Count=20: should match high (first), not low
    result = await compiled.run(trigger_id="trigger_1", event={"count": 20})
    assert result["task_high"]["result"] == "High"
    assert "task_very_high" not in result
    assert "task_low" not in result

    # Count=5: should match low (only one that matches)
    result = await compiled.run(trigger_id="trigger_1", event={"count": 5})
    assert result["task_low"]["result"] == "Low"
    assert "task_very_high" not in result
    assert "task_high" not in result


# =============================================================================
# Default Case Tests
# =============================================================================

@pytest.mark.asyncio
async def test_switch_with_default():
    """Test switch with default case when no conditions match."""
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
                            "value": {"type": "number"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.value} > 100", "label": "high"},
                    {"condition": "${trigger.event.value} < 10", "label": "low"},
                ]
            },
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
            {"id": "task_low", "type": "tool", "tool": "test.echo", "inputs": {"message": "Low"}},
            {"id": "task_default", "type": "tool", "tool": "test.echo", "inputs": {"message": "Default"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
            {"source": "switch_1", "target": "task_low", "type": "switch_case", "route": "low"},
            {"source": "switch_1", "target": "task_default", "type": "switch_default"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # value=50 (middle) - should use default
    result = await compiled.run(trigger_id="trigger_1", event={"value": 50})
    assert result["task_default"]["result"] == "Default"
    assert "task_high" not in result
    assert "task_low" not in result


@pytest.mark.asyncio
async def test_switch_without_default_no_match():
    """Test switch without default case - should end workflow when no match."""
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
                            "value": {"type": "number"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.value} > 100", "label": "high"},
                ]
            },
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # value=50 - no match, no default, workflow should end
    result = await compiled.run(trigger_id="trigger_1", event={"value": 50})
    # Only switch result should be in state, task_high should not run
    assert "task_high" not in result
    assert "_switch_result_switch_1" in result
    assert result["_switch_result_switch_1"] is None  # No match


# =============================================================================
# Complex Conditions Tests
# =============================================================================

@pytest.mark.asyncio
async def test_switch_complex_conditions():
    """Test switch with complex boolean expressions."""
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
                            "tier": {"type": "string"},
                            "severity": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {
                        "condition": "${trigger.event.tier} == 'premium' and ${trigger.event.severity} == 'high'",
                        "label": "urgent"
                    },
                    {
                        "condition": "${trigger.event.tier} == 'premium'",
                        "label": "priority"
                    },
                    {
                        "condition": "${trigger.event.severity} == 'high'",
                        "label": "high"
                    },
                ]
            },
            {"id": "task_urgent", "type": "tool", "tool": "test.echo", "inputs": {"message": "Urgent"}},
            {"id": "task_priority", "type": "tool", "tool": "test.echo", "inputs": {"message": "Priority"}},
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_urgent", "type": "switch_case", "route": "urgent"},
            {"source": "switch_1", "target": "task_priority", "type": "switch_case", "route": "priority"},
            {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # Premium + High Severity = Urgent
    result = await compiled.run(trigger_id="trigger_1", event={"tier": "premium", "severity": "high"})
    assert result["task_urgent"]["result"] == "Urgent"

    # Premium + Low Severity = Priority
    result = await compiled.run(trigger_id="trigger_1", event={"tier": "premium", "severity": "low"})
    assert result["task_priority"]["result"] == "Priority"

    # Basic + High Severity = High
    result = await compiled.run(trigger_id="trigger_1", event={"tier": "basic", "severity": "high"})
    assert result["task_high"]["result"] == "High"


# =============================================================================
# Validation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_switch_missing_edges():
    """Test that compilation fails when cases are missing edges."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {"id": "trigger_1", "key": "test.webhook", "mode": "webhook", "schemas": {"event": {"type": "object"}}}
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.x} > 10", "label": "high"},
                    {"condition": "${trigger.event.x} <= 10", "label": "low"},
                ]
            },
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            # Missing edge for "high" case!
            # {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
        ]
    }

    with pytest.raises(ValueError, match="SwitchNode switch_1 has cases without edges: high, low"):
        await _compile_workflow(spec, tool_registry)


@pytest.mark.asyncio
async def test_switch_invalid_references():
    """Test that invalid references in conditions are caught during validation."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {"id": "trigger_1", "key": "test.webhook", "mode": "webhook", "schemas": {"event": {"type": "object"}}}
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${nonexistent.value} > 10", "label": "high"},  # Invalid reference!
                ]
            },
            {"id": "task_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "High"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_high", "type": "switch_case", "route": "high"},
        ]
    }

    with pytest.raises(ValidationPhaseError, match="Unknown symbol.*nonexistent"):
        await _compile_workflow(spec, tool_registry)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_switch_with_upstream_task():
    """Test switch node routing based on upstream task output."""
    tool_registry = ToolRegistry()
    tool_registry.register(_create_echo_tool())

    spec = {
        "version": "2",
        "triggers": [
            {"id": "trigger_1", "key": "test.webhook", "mode": "webhook", "schemas": {"event": {"type": "object"}}}
        ],
        "nodes": [
            {"id": "task_1", "type": "tool", "tool": "test.echo", "inputs": {"message": "success"}},
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${task_1.result} == 'success'", "label": "success"},
                    {"condition": "${task_1.result} == 'error'", "label": "error"},
                ]
            },
            {"id": "task_success", "type": "tool", "tool": "test.echo", "inputs": {"message": "Success path"}},
            {"id": "task_error", "type": "tool", "tool": "test.echo", "inputs": {"message": "Error path"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "task_1", "type": "trigger"},
            {"source": "task_1", "target": "switch_1", "type": "default"},
            {"source": "switch_1", "target": "task_success", "type": "switch_case", "route": "success"},
            {"source": "switch_1", "target": "task_error", "type": "switch_case", "route": "error"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)
    result = await compiled.run(trigger_id="trigger_1", event={})

    assert result["task_1"]["result"] == "success"
    assert result["task_success"]["result"] == "Success path"
    assert "task_error" not in result


@pytest.mark.asyncio
async def test_nested_switches():
    """Test nested switch nodes (switch inside switch branch)."""
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
                            "category": {"type": "string"},
                            "priority": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_category",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.category} == 'A'", "label": "cat_a"},
                    {"condition": "${trigger.event.category} == 'B'", "label": "cat_b"},
                ]
            },
            {
                "id": "switch_priority",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.priority} == 'high'", "label": "high"},
                    {"condition": "${trigger.event.priority} == 'low'", "label": "low"},
                ]
            },
            {"id": "task_a_high", "type": "tool", "tool": "test.echo", "inputs": {"message": "A-High"}},
            {"id": "task_a_low", "type": "tool", "tool": "test.echo", "inputs": {"message": "A-Low"}},
            {"id": "task_b", "type": "tool", "tool": "test.echo", "inputs": {"message": "B"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_category", "type": "trigger"},
            {"source": "switch_category", "target": "switch_priority", "type": "switch_case", "route": "cat_a"},
            {"source": "switch_category", "target": "task_b", "type": "switch_case", "route": "cat_b"},
            {"source": "switch_priority", "target": "task_a_high", "type": "switch_case", "route": "high"},
            {"source": "switch_priority", "target": "task_a_low", "type": "switch_case", "route": "low"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    # Category A + High Priority
    result = await compiled.run(trigger_id="trigger_1", event={"category": "A", "priority": "high"})
    assert result["task_a_high"]["result"] == "A-High"
    assert "task_a_low" not in result
    assert "task_b" not in result

    # Category A + Low Priority
    result = await compiled.run(trigger_id="trigger_1", event={"category": "A", "priority": "low"})
    assert result["task_a_low"]["result"] == "A-Low"

    # Category B (skip second switch)
    result = await compiled.run(trigger_id="trigger_1", event={"category": "B", "priority": "high"})
    assert result["task_b"]["result"] == "B"
    assert "task_a_high" not in result


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.asyncio
async def test_single_case_switch():
    """Test switch with only one case."""
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
                        "properties": {"value": {"type": "boolean"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "switch_1",
                "type": "switch",
                "cases": [
                    {"condition": "${trigger.event.value}", "label": "true_case"},
                ]
            },
            {"id": "task_true", "type": "tool", "tool": "test.echo", "inputs": {"message": "True"}},
        ],
        "edges": [
            {"source": "trigger_1", "target": "switch_1", "type": "trigger"},
            {"source": "switch_1", "target": "task_true", "type": "switch_case", "route": "true_case"},
        ]
    }

    compiled = await _compile_workflow(spec, tool_registry)

    result = await compiled.run(trigger_id="trigger_1", event={"value": True})
    assert result["task_true"]["result"] == "True"

    # False case - no match, no default, ends
    result = await compiled.run(trigger_id="trigger_1", event={"value": False})
    assert "task_true" not in result
