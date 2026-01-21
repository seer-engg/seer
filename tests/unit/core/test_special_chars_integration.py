"""
Integration tests for workflow compilation with special characters in 'out' keys and trigger titles.

Tests the full workflow lifecycle: parsing -> type environment building ->
validation -> execution planning -> runtime execution.
"""
from __future__ import annotations

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.errors import TypeEnvironmentError
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry


async def _compile_workflow(spec_payload: dict, tool_defs: list[ToolDefinition] | None = None) -> CompiledWorkflow:
    """Compile a workflow spec into a CompiledWorkflow object."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    if tool_defs:
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
# Integration Tests for 'out' Keys with Special Characters
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_with_out_keys_containing_spaces():
    """Test full workflow compilation with 'out' keys containing spaces."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "Hello World",
                "out": "my task result"  # 'out' key with spaces
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": "${my task result}",  # Reference with spaces
                "out": "final output"  # Another 'out' key with spaces
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify type environment has the correct symbols
    assert "my task result" in workflow.type_env
    assert "final output" in workflow.type_env
    assert workflow.type_env["my task result"]["type"] == "string"
    assert workflow.type_env["final output"]["type"] == "string"


@pytest.mark.asyncio
async def test_workflow_with_out_keys_containing_hyphens():
    """Test full workflow compilation with 'out' keys containing hyphens."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": 42,
                "out": "user-count"  # 'out' key with hyphen
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": 100,
                "out": "max-limit"
            },
            {
                "id": "if1",
                "type": "if",
                "condition": "${user-count} < ${max-limit}"  # References with hyphens
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"},
            {"id": "edge_task2_if1", "source": "task2", "target": "if1", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify type environment
    assert "user-count" in workflow.type_env
    assert "max-limit" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_out_keys_containing_special_chars():
    """
    Test workflow with various special characters in 'out' keys.

    Note: Characters like '.', '[', ']' have special meaning in references
    (property/array access), so they can be in 'out' keys but cannot be
    directly referenced. Other special characters like '@', '#', '$' work fine.
    """
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "test@example.com",
                "out": "email@field"  # @ character
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": {"status": "ok"},
                "out": "api#response"  # # character
            },
            {
                "id": "task3",
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3],
                "out": "data$items"  # $ character
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all special character out keys are registered
    assert "email@field" in workflow.type_env
    assert "api#response" in workflow.type_env
    assert "data$items" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_unicode_out_keys():
    """Test workflow with Unicode characters in 'out' keys."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "Success",
                "out": "résultat"  # French accent
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": {"message": "完成"},
                "out": "结果"  # Chinese characters
            },
            {
                "id": "task3",
                "type": "task",
                "kind": "set",
                "value": "Combined: ${résultat} - ${结果}",
                "out": "summary"
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"},
            {"id": "edge_task2_task3", "source": "task2", "target": "task3", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify Unicode out keys are registered
    assert "résultat" in workflow.type_env
    assert "结果" in workflow.type_env
    assert "summary" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_nested_property_access_special_chars():
    """Test workflow with nested property access on 'out' keys with special characters."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": {
                    "user": {"name": "John", "age": 30},
                    "status": "active"
                },
                "out": "api-response"  # Hyphen in out key
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": "${api-response.user.name}",  # Nested property access
                "out": "user name"  # Space in out key
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify both out keys are registered
    assert "api-response" in workflow.type_env
    assert "user name" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_foreach_special_out_keys():
    """Test workflow with ForEach node using special characters in 'out' keys."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": [{"id": 1}, {"id": 2}, {"id": 3}],
                "out": "data-items"  # Hyphen in out key
            },
            {
                "id": "loop1",
                "type": "for_each",
                "items": "${data-items}",  # Reference with hyphen
                "item_var": "current_item",
                "index_var": "idx",
                "out": "processed items"  # Space in out key
            }
        ],
        "edges": [
            {"id": "edge_task1_loop1", "source": "task1", "target": "loop1", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify out keys and loop variables are registered
    assert "data-items" in workflow.type_env
    assert "processed items" in workflow.type_env
    assert "current_item" in workflow.type_env
    assert "idx" in workflow.type_env


# =============================================================================
# Integration Tests for Trigger Titles with Special Characters
# =============================================================================


def test_workflow_with_trigger_title_spaces_fails():
    """Test that trigger titles with spaces fail during compilation."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "title": "My Trigger",  # Invalid: spaces
                "provider": "test",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}}
                    }
                }
            }
        ],
        "nodes": [],
        "edges": []
    }

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    # Should fail during type environment building
    parsed_spec = parse_workflow_spec(spec)
    with pytest.raises(TypeEnvironmentError, match="Invalid trigger title 'My Trigger'"):
        build_type_environment(
            parsed_spec,
            schema_registry=schema_registry,
            tool_registry=tool_registry,
        )


def test_workflow_with_trigger_title_hyphen_fails():
    """Test that trigger titles with hyphens fail during compilation."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "title": "my-trigger",  # Invalid: hyphen
                "provider": "test",
                "mode": "polling",
                "schemas": {"event": {}}
            }
        ],
        "nodes": [],
        "edges": []
    }

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    parsed_spec = parse_workflow_spec(spec)
    with pytest.raises(TypeEnvironmentError, match="Invalid trigger title 'my-trigger'"):
        build_type_environment(
            parsed_spec,
            schema_registry=schema_registry,
            tool_registry=tool_registry,
        )


@pytest.mark.parametrize("invalid_title", [
    "trigger@email",
    "trigger.name",
    "trigger:colon",
    "1trigger",
    "trigger!",
])
def test_workflow_with_trigger_title_special_chars_fails(invalid_title):
    """Test that trigger titles with various special characters fail during compilation."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "title": invalid_title,
                "provider": "test",
                "mode": "polling",
                "schemas": {"event": {}}
            }
        ],
        "nodes": [],
        "edges": []
    }

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    parsed_spec = parse_workflow_spec(spec)
    with pytest.raises(TypeEnvironmentError, match=f"Invalid trigger title '{invalid_title}'"):
        build_type_environment(
            parsed_spec,
            schema_registry=schema_registry,
            tool_registry=tool_registry,
        )


# =============================================================================
# Complex Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complex_workflow_with_mixed_special_characters():
    """Test complex workflow with multiple nodes using various special characters."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": {"status": 200, "message": "OK"},
                "out": "api-response"
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": 150,
                "out": "user count"  # Space
            },
            {
                "id": "task3",
                "type": "task",
                "kind": "set",
                "value": "2024-01-15T10:30:00Z",
                "out": "timestamp@data"  # @ character
            },
            {
                "id": "if1",
                "type": "if",
                "condition": "${user count} > 100"  # Reference with space
            },
            {
                "id": "task4",
                "type": "task",
                "kind": "set",
                "value": "Status: ${api-response.status}, Users: ${user count}, Time: ${timestamp@data}",
                "out": "final-summary"  # Hyphen
            },
            {
                "id": "task5",
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3, 4, 5],
                "out": "data$array"  # $ character
            },
            {
                "id": "loop1",
                "type": "for_each",
                "items": "${data$array}",
                "item_var": "item",
                "index_var": "index",
                "out": "processed#results"  # # character
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"},
            {"id": "edge_task2_task3", "source": "task2", "target": "task3", "type": "default"},
            {"id": "edge_task3_if1", "source": "task3", "target": "if1", "type": "default"},
            {"id": "edge_if1_task4", "source": "if1", "target": "task4", "type": "conditional_true"},
            {"id": "edge_task4_task5", "source": "task4", "target": "task5", "type": "default"},
            {"id": "edge_task5_loop1", "source": "task5", "target": "loop1", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all special character out keys are registered
    assert "api-response" in workflow.type_env
    assert "user count" in workflow.type_env
    assert "timestamp@data" in workflow.type_env
    assert "final-summary" in workflow.type_env
    assert "data$array" in workflow.type_env
    assert "processed#results" in workflow.type_env

    # Verify loop variables
    assert "item" in workflow.type_env
    assert "index" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_template_strings_with_special_out_keys():
    """Test workflow using template strings with references to special character 'out' keys."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "John",
                "out": "first-name"
            },
            {
                "id": "task2",
                "type": "task",
                "kind": "set",
                "value": "Doe",
                "out": "last name"  # Space
            },
            {
                "id": "task3",
                "type": "task",
                "kind": "set",
                "value": 30,
                "out": "user@age"  # @ sign
            },
            {
                "id": "task4",
                "type": "task",
                "kind": "set",
                "value": "Name: ${first-name} ${last name}, Age: ${user@age}",  # Template with all special refs
                "out": "user#profile"  # # character
            }
        ],
        "edges": [
            {"id": "edge_task1_task2", "source": "task1", "target": "task2", "type": "default"},
            {"id": "edge_task2_task3", "source": "task2", "target": "task3", "type": "default"},
            {"id": "edge_task3_task4", "source": "task3", "target": "task4", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all out keys are registered
    assert "first-name" in workflow.type_env
    assert "last name" in workflow.type_env
    assert "user@age" in workflow.type_env
    assert "user#profile" in workflow.type_env
