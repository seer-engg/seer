# pylint: disable=too-many-lines,duplicate-code
# Reason: Comprehensive integration tests for special character handling; test helper code shared with test_multiple_triggers
"""
Integration tests for workflow compilation with special characters in node IDs and trigger IDs.

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
# Integration Tests for Node IDs with Special Characters
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_with_node_ids_containing_spaces():
    """Test full workflow compilation with node IDs containing spaces."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "my task result",  # Node ID with spaces
                "type": "task",
                "kind": "set",
                "value": "Hello World"
            },
            {
                "id": "final output",  # Node ID with spaces
                "type": "task",
                "kind": "set",
                "value": "${my task result}"  # Reference with spaces
            }
        ],
        "edges": [
            {"source": "my task result", "target": "final output", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify type environment has the correct symbols (node IDs)
    assert "my task result" in workflow.type_env
    assert "final output" in workflow.type_env
    assert workflow.type_env["my task result"]["type"] == "string"
    assert workflow.type_env["final output"]["type"] == "string"


@pytest.mark.asyncio
async def test_workflow_with_node_ids_containing_hyphens():
    """Test full workflow compilation with node IDs containing hyphens."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "user-count",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": 42
            },
            {
                "id": "max-limit",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": 100
            },
            {
                "id": "if1",
                "type": "if",
                "condition": "${user-count} < ${max-limit}"  # References with hyphens
            }
        ],
        "edges": [
            {"source": "user-count", "target": "max-limit", "type": "default"},
            {"source": "max-limit", "target": "if1", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify type environment (node IDs)
    assert "user-count" in workflow.type_env
    assert "max-limit" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_node_ids_containing_special_chars():
    """
    Test workflow with various special characters in node IDs.

    Note: Characters like '.', '[', ']' have special meaning in references
    (property/array access), so they can be in node IDs but cannot be
    directly referenced. Other special characters like '@', '#', '$' work fine.
    """
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "email@field",  # Node ID with @ character
                "type": "task",
                "kind": "set",
                "value": "test@example.com"
            },
            {
                "id": "api#response",  # Node ID with # character
                "type": "task",
                "kind": "set",
                "value": {"status": "ok"}
            },
            {
                "id": "data$items",  # Node ID with $ character
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3]
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all special character node IDs are registered
    assert "email@field" in workflow.type_env
    assert "api#response" in workflow.type_env
    assert "data$items" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_unicode_node_ids():
    """Test workflow with Unicode characters in node IDs."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "résultat",  # Node ID with French accent
                "type": "task",
                "kind": "set",
                "value": "Success"
            },
            {
                "id": "结果",  # Node ID with Chinese characters
                "type": "task",
                "kind": "set",
                "value": {"message": "完成"}
            },
            {
                "id": "summary",
                "type": "task",
                "kind": "set",
                "value": "Combined: ${résultat} - ${结果}"
            }
        ],
        "edges": [
            {"source": "résultat", "target": "结果", "type": "default"},
            {"source": "结果", "target": "summary", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify Unicode node IDs are registered
    assert "résultat" in workflow.type_env
    assert "结果" in workflow.type_env
    assert "summary" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_nested_property_access_special_chars():
    """Test workflow with nested property access on node IDs with special characters."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "api-response",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": {
                    "user": {"name": "John", "age": 30},
                    "status": "active"
                }
            },
            {
                "id": "user name",  # Node ID with space
                "type": "task",
                "kind": "set",
                "value": "${api-response.user.name}"  # Nested property access
            }
        ],
        "edges": [
            {"source": "api-response", "target": "user name", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify both node IDs are registered
    assert "api-response" in workflow.type_env
    assert "user name" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_foreach_special_node_ids():
    """Test workflow with ForEach node using special characters in node IDs."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "data-items",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": [{"id": 1}, {"id": 2}, {"id": 3}]
            },
            {
                "id": "processed items",  # Node ID with space
                "type": "for_each",
                "items": "${data-items}",  # Reference with hyphen
                "item_var": "current_item",
                "index_var": "idx"
            }
        ],
        "edges": [
            {"source": "data-items", "target": "processed items", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify node IDs and loop variables are registered
    assert "data-items" in workflow.type_env
    assert "processed items" in workflow.type_env
    assert "current_item" in workflow.type_env
    assert "idx" in workflow.type_env


# =============================================================================
# Integration Tests for Trigger IDs with Special Characters
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_with_trigger_id_spaces():
    """Test that trigger IDs with spaces are now accepted and work in full workflow."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "My Trigger",  # Trigger ID with spaces
                "key": "test.trigger",
                "title": "My Trigger",
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
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "${My Trigger.data}"  # Reference with space
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify trigger is registered in type environment (by trigger ID)
    assert "My Trigger" in workflow.type_env
    assert "My Trigger.data" in workflow.type_env
    assert "task1" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_trigger_id_hyphen():
    """Test that trigger IDs with hyphens are now accepted and work in full workflow."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "my-trigger",  # Trigger ID with hyphen
                "key": "test.trigger",
                "title": "my-trigger",
                "provider": "test",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"value": {"type": "number"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": "${my-trigger.value}"  # Reference with hyphen
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify trigger is registered in type environment (by trigger ID)
    assert "my-trigger" in workflow.type_env
    assert "my-trigger.value" in workflow.type_env
    assert "task1" in workflow.type_env


@pytest.mark.parametrize("valid_id", [
    "trigger@email",
    # Note: "trigger.name" is omitted because dots have special meaning in property access
    # and would be ambiguous (trigger.name.message could be parsed as trigger["name"]["message"])
    "trigger:colon",
    "1trigger",
    "trigger!",
    "数据触发器",  # Unicode
    "trigger with spaces",
])
@pytest.mark.asyncio
async def test_workflow_with_trigger_id_special_chars(valid_id):
    """Test that trigger IDs with various special characters are now accepted."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": valid_id,  # Trigger ID with special chars
                "key": "test.trigger",
                "title": valid_id,
                "provider": "test",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "task1",
                "type": "task",
                "kind": "set",
                "value": f"${{{valid_id}.message}}"  # Reference with special char
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify trigger is registered in type environment (by trigger ID)
    assert valid_id in workflow.type_env
    assert f"{valid_id}.message" in workflow.type_env
    assert "task1" in workflow.type_env


# =============================================================================
# Complex Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complex_workflow_with_mixed_special_characters():
    """Test complex workflow with multiple nodes using various special characters in node IDs."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "api-response",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": {"status": 200, "message": "OK"}
            },
            {
                "id": "user count",  # Node ID with space
                "type": "task",
                "kind": "set",
                "value": 150
            },
            {
                "id": "timestamp@data",  # Node ID with @ character
                "type": "task",
                "kind": "set",
                "value": "2024-01-15T10:30:00Z"
            },
            {
                "id": "if1",
                "type": "if",
                "condition": "${user count} > 100"  # Reference with space
            },
            {
                "id": "final-summary",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": "Status: ${api-response.status}, Users: ${user count}, Time: ${timestamp@data}"
            },
            {
                "id": "data$array",  # Node ID with $ character
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3, 4, 5]
            },
            {
                "id": "processed#results",  # Node ID with # character
                "type": "for_each",
                "items": "${data$array}",
                "item_var": "item",
                "index_var": "index"
            }
        ],
        "edges": [
            {"source": "api-response", "target": "user count", "type": "default"},
            {"source": "user count", "target": "timestamp@data", "type": "default"},
            {"source": "timestamp@data", "target": "if1", "type": "default"},
            {"source": "if1", "target": "final-summary", "type": "conditional_true"},
            {"source": "final-summary", "target": "data$array", "type": "default"},
            {"source": "data$array", "target": "processed#results", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all special character node IDs are registered
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
async def test_workflow_template_strings_with_special_node_ids():
    """Test workflow using template strings with references to special character node IDs."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "first-name",  # Node ID with hyphen
                "type": "task",
                "kind": "set",
                "value": "John"
            },
            {
                "id": "last name",  # Node ID with space
                "type": "task",
                "kind": "set",
                "value": "Doe"
            },
            {
                "id": "user@age",  # Node ID with @ sign
                "type": "task",
                "kind": "set",
                "value": 30
            },
            {
                "id": "user#profile",  # Node ID with # character
                "type": "task",
                "kind": "set",
                "value": "Name: ${first-name} ${last name}, Age: ${user@age}"  # Template with all special refs
            }
        ],
        "edges": [
            {"source": "first-name", "target": "last name", "type": "default"},
            {"source": "last name", "target": "user@age", "type": "default"},
            {"source": "user@age", "target": "user#profile", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all node IDs are registered
    assert "first-name" in workflow.type_env
    assert "last name" in workflow.type_env
    assert "user@age" in workflow.type_env
    assert "user#profile" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_multiple_triggers_special_chars():
    """Test workflow with multiple triggers having special characters in IDs."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "Gmail Inbox",  # Trigger ID with space
                "key": "gmail.inbox",
                "title": "Gmail Inbox",
                "provider": "gmail",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"subject": {"type": "string"}}
                    }
                }
            },
            {
                "id": "slack-message",  # Trigger ID with hyphen
                "key": "slack.message",
                "title": "slack-message",
                "provider": "slack",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}}
                    }
                }
            },
            {
                "id": "trigger@webhook",  # Trigger ID with @ character
                "key": "custom.trigger",
                "title": "trigger@webhook",
                "provider": "custom",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"payload": {"type": "object"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "combined message",  # Node ID with space
                "type": "task",
                "kind": "set",
                "value": "Email: ${Gmail Inbox.subject}, Slack: ${slack-message.text}"
            },
            {
                "id": "if1",
                "type": "if",
                "condition": "${trigger@webhook.payload} != null"
            }
        ],
        "edges": [
            {"source": "combined message", "target": "if1", "type": "default"}
        ]
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify all triggers are registered with their special character IDs
    assert "Gmail Inbox" in workflow.type_env
    assert "Gmail Inbox.subject" in workflow.type_env
    assert "slack-message" in workflow.type_env
    assert "slack-message.text" in workflow.type_env
    assert "trigger@webhook" in workflow.type_env
    assert "trigger@webhook.payload" in workflow.type_env
    assert "combined message" in workflow.type_env


@pytest.mark.asyncio
async def test_workflow_with_unicode_trigger_ids():
    """Test workflow with Unicode characters in trigger IDs."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "数据触发器",  # Trigger ID with Chinese characters
                "key": "chinese.trigger",
                "title": "数据触发器",
                "provider": "custom",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"数据": {"type": "string"}}
                    }
                }
            },
            {
                "id": "Déclencheur Français",  # Trigger ID with French accents and space
                "key": "french.trigger",
                "title": "Déclencheur Français",
                "provider": "custom",
                "mode": "polling",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}}
                    }
                }
            }
        ],
        "nodes": [
            {
                "id": "chinese_result",
                "type": "task",
                "kind": "set",
                "value": "${数据触发器.数据}"
            },
            {
                "id": "french_result",
                "type": "task",
                "kind": "set",
                "value": "${Déclencheur Français.message}"
            }
        ],
        "edges": []
    }

    # Should compile successfully
    workflow = await _compile_workflow(spec)

    # Verify Unicode triggers are registered (by trigger ID)
    assert "数据触发器" in workflow.type_env
    assert "数据触发器.数据" in workflow.type_env
    assert "Déclencheur Français" in workflow.type_env
    assert "Déclencheur Français.message" in workflow.type_env
    assert "chinese_result" in workflow.type_env
    assert "french_result" in workflow.type_env
