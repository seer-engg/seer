# pylint: disable=import-outside-toplevel,too-many-locals,unused-import
# Reason: Test file with lazy imports and complex workflow setup
"""
Tests for multiple triggers of the same type in a workflow.

This test suite verifies that workflows can have multiple triggers of the same
trigger type (e.g., two Gmail triggers, two webhook triggers) and that they
route correctly to different nodes based on trigger ID.
"""
from __future__ import annotations

import pytest

from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.validate_refs import validate_references
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.schema.models import WorkflowSpec
from seer.core.errors import ValidationPhaseError
from seer.core.registry.tool_registry import ToolDefinition


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


@pytest.mark.unit
class TestMultipleTriggersSameType:
    """Test suite for multiple triggers of the same type."""

    def test_schema_allows_duplicate_trigger_keys_with_different_ids(self):
        """Verify WorkflowSpec accepts multiple triggers with same key but different IDs."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
                {"id": "node2", "type": "tool", "tool": "test.tool", "inputs": {"value": "result2"}},
            ],
            "edges": [
                {"source": "trigger_1", "target": "node1", "type": "trigger"},
                {"source": "trigger_2", "target": "node2", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "trigger_1",
                    "key": "webhook.generic",
                    "title": "Webhook_1",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
                {
                    "id": "trigger_2",
                    "key": "webhook.generic",
                    "title": "Webhook_2",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        # Should parse without error
        spec = parse_workflow_spec(spec_payload)
        assert len(spec.triggers) == 2
        assert spec.triggers[0].key == spec.triggers[1].key  # Same type
        assert spec.triggers[0].id != spec.triggers[1].id  # Different IDs

    def test_schema_rejects_duplicate_trigger_ids(self):
        """Verify WorkflowSpec rejects multiple triggers with same ID."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [
                {"source": "trigger_1", "target": "node1", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "trigger_1",
                    "key": "webhook.generic",
                    "title": "Webhook_1",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
                {
                    "id": "trigger_1",  # Duplicate ID!
                    "key": "webhook.other",
                    "title": "Webhook_2",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        # Should raise validation error
        with pytest.raises(ValidationPhaseError, match="Duplicate trigger id"):
            parse_workflow_spec(spec_payload)

    def test_edge_validation_uses_trigger_id(self):
        """Verify edge validation checks trigger IDs, not keys."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [
                {"source": "nonexistent_trigger", "target": "node1", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "trigger_1",
                    "key": "webhook.generic",
                    "title": "Webhook_1",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        # Should raise validation error for missing trigger ID
        with pytest.raises(ValidationPhaseError, match="Trigger edge with source 'nonexistent_trigger'.*not found in triggers"):
            parse_workflow_spec(spec_payload)

    def test_compiler_builds_separate_trigger_targets(self):
        """Verify compiler creates separate routing entries for each trigger instance."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
                {"id": "node2", "type": "tool", "tool": "test.tool", "inputs": {"value": "result2"}},
                {"id": "node3", "type": "tool", "tool": "test.tool", "inputs": {"value": "result3"}},
            ],
            "edges": [
                {"source": "gmail_1", "target": "node1", "type": "trigger"},
                {"source": "gmail_2", "target": "node2", "type": "trigger"},
                {"source": "webhook_1", "target": "node3", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "gmail_1",
                    "key": "gmail.new_email",
                    "title": "Gmail_Inbox",
                    "provider": "gmail",
                    "mode": "polling",
                    "enabled": True,
                },
                {
                    "id": "gmail_2",
                    "key": "gmail.new_email",  # Same type as gmail_1
                    "title": "Gmail_Important",
                    "provider": "gmail",
                    "mode": "polling",
                    "enabled": True,
                },
                {
                    "id": "webhook_1",
                    "key": "webhook.generic",
                    "title": "Generic_Webhook",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        plan = build_execution_plan(spec)

        # Verify trigger_targets has 3 entries (one per trigger instance)
        assert len(plan.trigger_targets) == 3
        assert plan.trigger_targets["gmail_1"] == ["node1"]
        assert plan.trigger_targets["gmail_2"] == ["node2"]
        assert plan.trigger_targets["webhook_1"] == ["node3"]

    def test_multiple_edges_from_same_trigger_fan_out(self):
        """
        Verify that if multiple edges exist from the same trigger ID,
        all targets are collected for parallel execution (fan-out).

        This enables workflows where a single trigger fires multiple
        parallel branches simultaneously.
        """
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
                {"id": "node2", "type": "tool", "tool": "test.tool", "inputs": {"value": "result2"}},
            ],
            "edges": [
                {"source": "trigger_1", "target": "node1", "type": "trigger"},
                {"source": "trigger_1", "target": "node2", "type": "trigger"},  # Same source, parallel
            ],
            "triggers": [
                {
                    "id": "trigger_1",
                    "key": "webhook.generic",
                    "title": "Webhook_1",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        plan = build_execution_plan(spec)

        # Both targets are collected for parallel execution
        assert len(plan.trigger_targets) == 1
        assert plan.trigger_targets["trigger_1"] == ["node1", "node2"]

    def test_trigger_id_field_is_required(self):
        """Verify that trigger specs require an id field."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [],
            "triggers": [
                {
                    # Missing "id" field
                    "key": "webhook.generic",
                    "title": "Webhook_1",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                },
            ],
        }

        # Should raise validation error for missing id
        with pytest.raises(ValidationPhaseError, match="Field required"):
            parse_workflow_spec(spec_payload)

    def test_workflow_with_three_gmail_triggers(self):
        """
        Comprehensive test with three Gmail triggers routing to different nodes.

        This simulates a real-world use case where a user wants to monitor:
        - Gmail inbox for new messages → node1
        - Gmail important folder → node2
        - Gmail sent folder → node3
        """
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "process_inbox", "type": "tool", "tool": "test.tool", "inputs": {"value": "inbox processed"}},
                {"id": "process_important", "type": "tool", "tool": "test.tool", "inputs": {"value": "important processed"}},
                {"id": "process_sent", "type": "tool", "tool": "test.tool", "inputs": {"value": "sent processed"}},
            ],
            "edges": [
                {"source": "gmail_inbox", "target": "process_inbox", "type": "trigger"},
                {"source": "gmail_important", "target": "process_important", "type": "trigger"},
                {"source": "gmail_sent", "target": "process_sent", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "gmail_inbox",
                    "key": "gmail.new_email",
                    "title": "Gmail_Inbox",
                    "provider": "gmail",
                    "mode": "polling",
                    "enabled": True,
                    "filters": {"folder": "inbox"},
                },
                {
                    "id": "gmail_important",
                    "key": "gmail.new_email",  # Same key!
                    "title": "Gmail_Important",
                    "provider": "gmail",
                    "mode": "polling",
                    "enabled": True,
                    "filters": {"folder": "important"},
                },
                {
                    "id": "gmail_sent",
                    "key": "gmail.new_email",  # Same key!
                    "title": "Gmail_Sent",
                    "provider": "gmail",
                    "mode": "polling",
                    "enabled": True,
                    "filters": {"folder": "sent"},
                },
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        plan = build_execution_plan(spec)

        # Verify all triggers are distinct and route to correct nodes
        assert len(spec.triggers) == 3
        assert len(plan.trigger_targets) == 3
        assert plan.trigger_targets["gmail_inbox"] == ["process_inbox"]
        assert plan.trigger_targets["gmail_important"] == ["process_important"]
        assert plan.trigger_targets["gmail_sent"] == ["process_sent"]

        # Verify all have same key but different IDs
        trigger_keys = [t.key for t in spec.triggers]
        trigger_ids = [t.id for t in spec.triggers]
        assert len(set(trigger_keys)) == 1  # All same key
        assert len(set(trigger_ids)) == 3  # All different IDs


@pytest.mark.unit
class TestTriggerEventEnvelope:
    """Test trigger event envelope includes both trigger_id and trigger_key."""

    def test_event_envelope_structure(self):
        """Verify event envelopes include both trigger_id and trigger_key."""
        from seer.core.triggers.events import build_event_envelope, TriggerEventEnvelopeInput
        from datetime import datetime, timezone

        event_input = TriggerEventEnvelopeInput(
            trigger_id="gmail_inbox_123",
            trigger_key="gmail.new_email",
            title="GmailInbox",
            provider="gmail",
            provider_connection_id=42,
            payload={"subject": "Test email"},
            raw={"raw_data": "..."},
            occurred_at=datetime.now(timezone.utc),
        )
        envelope = build_event_envelope(event_input)

        assert "trigger_id" in envelope
        assert "trigger_key" in envelope
        assert envelope["trigger_id"] == "gmail_inbox_123"
        assert envelope["trigger_key"] == "gmail.new_email"
        assert envelope["data"]["subject"] == "Test email"


@pytest.mark.unit
class TestTriggerReferenceResolution:
    """Test trigger reference resolution with realistic envelope structure."""

    @pytest.mark.asyncio
    async def test_trigger_reference_with_mismatched_ids(self):
        """Verify ${trigger-id} works when envelope.id != envelope.trigger_id."""
        from seer.core.compiler.emit_langgraph import emit_langgraph
        from seer.core.compiler.type_env import build_type_environment
        from seer.core.compiler.validate_refs import validate_references
        from seer.core.registry.model_registry import ModelRegistry
        from seer.core.registry.tool_registry import ToolRegistry
        from seer.core.runtime.execution import CompiledWorkflow
        from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
        from seer.core.schema.schema_registry import SchemaRegistry

        spec_payload = {
            "version": "2",
            "nodes": [
                {
                    "id": "echo_node",
                    "type": "tool",
                    "tool": "test.tool",
                    "inputs": {"value": "${trigger-1.data.message}"},
                },
            ],
            "edges": [
                {"source": "trigger-1", "target": "echo_node", "type": "trigger"},
            ],
            "triggers": [
                {
                    "id": "trigger-1",
                    "key": "webhook.generic",
                    "title": "Test Trigger",
                    "provider": "webhook",
                    "mode": "webhook",
                    "enabled": True,
                    "schemas": {
                        "event": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        }

        # Compile workflow using test pattern
        schema_registry = SchemaRegistry()
        tool_registry = ToolRegistry()
        model_registry = ModelRegistry()

        # Register mock tool
        mock_tool = _create_mock_tool()
        tool_registry.register(mock_tool)

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
        compiled = CompiledWorkflow(
            spec=spec,
            type_env=type_env.as_dict(),
            graph=graph,
            runtime=runtime,
        )

        # Create envelope mimicking production: id is UUID, trigger_id is spec ID
        trigger_envelope = {
            "id": "evt_c33d2cc513c44f1eb6b584beb5c20e11",  # System-generated UUID
            "trigger_id": "trigger-1",  # Workflow spec ID
            "trigger_key": "webhook.generic",
            "title": "Test Trigger",
            "data": {"message": "hello from trigger"},
            "occurred_at": "2026-01-22T09:00:00Z",
            "received_at": "2026-01-22T09:00:00Z",
        }

        result = await compiled.ainvoke(
            config=None,
            context=None,
            trigger=trigger_envelope,
        )

        # Should successfully resolve ${trigger-1.data.message}
        assert result["echo_node"] == "hello from trigger"


@pytest.mark.unit
class TestOrphanedTriggerValidation:
    """Test suite for orphaned trigger detection (compile-time validation)."""

    def test_compiler_rejects_single_orphaned_trigger(self):
        """Verify validate_references rejects a trigger with no edge connecting it."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [],  # No trigger edge!
            "triggers": [
                {
                    "id": "trigger_1",
                    "key": "webhook.generic",
                    "mode": "webhook",
                },
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        with pytest.raises(ValidationPhaseError, match="Orphaned triggers without edges are not allowed"):
            validate_references(spec, type_env)

    def test_compiler_rejects_multiple_orphaned_triggers(self):
        """Verify error lists all orphaned trigger IDs."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [],
            "triggers": [
                {"id": "trigger_a", "key": "webhook.generic", "mode": "webhook"},
                {"id": "trigger_b", "key": "gmail.new_email", "mode": "polling"},
                {"id": "trigger_c", "key": "schedule.cron", "mode": "schedule"},
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        with pytest.raises(ValidationPhaseError) as exc_info:
            validate_references(spec, type_env)

        error_msg = str(exc_info.value)
        assert "trigger_a" in error_msg
        assert "trigger_b" in error_msg
        assert "trigger_c" in error_msg

    def test_compiler_rejects_partial_orphaned_triggers(self):
        """Verify error when some triggers are connected and others are not."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [
                {"source": "connected_trigger", "target": "node1", "type": "trigger"},
            ],
            "triggers": [
                {"id": "connected_trigger", "key": "webhook.generic", "mode": "webhook"},
                {"id": "orphan_trigger", "key": "gmail.new_email", "mode": "polling"},
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        with pytest.raises(ValidationPhaseError) as exc_info:
            validate_references(spec, type_env)

        error_msg = str(exc_info.value)
        assert "orphan_trigger" in error_msg
        assert "connected_trigger" not in error_msg  # Should not list connected trigger

    def test_compiler_allows_triggers_with_edges(self):
        """Verify validate_references accepts triggers that are properly connected."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
                {"id": "node2", "type": "tool", "tool": "test.tool", "inputs": {"value": "result2"}},
            ],
            "edges": [
                {"source": "trigger_1", "target": "node1", "type": "trigger"},
                {"source": "trigger_2", "target": "node2", "type": "trigger"},
            ],
            "triggers": [
                {"id": "trigger_1", "key": "webhook.generic", "mode": "webhook"},
                {"id": "trigger_2", "key": "gmail.new_email", "mode": "polling"},
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        # Should validate without error
        validate_references(spec, type_env)
        assert len(spec.triggers) == 2

    def test_compiler_allows_workflow_without_triggers(self):
        """Verify validate_references accepts workflows with no triggers (manual execution)."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
            ],
            "edges": [],
            "triggers": [],  # No triggers at all is valid
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        # Should validate without error
        validate_references(spec, type_env)
        assert len(spec.triggers) == 0

    def test_compiler_allows_trigger_with_multiple_edges(self):
        """Verify a single trigger can have multiple edges (fan-out pattern)."""
        spec_payload = {
            "version": "2",
            "nodes": [
                {"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {"value": "result1"}},
                {"id": "node2", "type": "tool", "tool": "test.tool", "inputs": {"value": "result2"}},
            ],
            "edges": [
                {"source": "trigger_1", "target": "node1", "type": "trigger"},
                {"source": "trigger_1", "target": "node2", "type": "trigger"},  # Same trigger, multiple targets
            ],
            "triggers": [
                {"id": "trigger_1", "key": "webhook.generic", "mode": "webhook"},
            ],
        }

        spec = parse_workflow_spec(spec_payload)
        type_env = TypeEnvironment()

        # Should validate without error - trigger has at least one edge
        validate_references(spec, type_env)
        assert len(spec.triggers) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
