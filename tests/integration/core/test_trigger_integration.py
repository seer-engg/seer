# pylint: disable=too-many-lines
# Reason: Comprehensive integration tests for trigger integration
"""
Integration tests for trigger data integration into workflows.

Tests verify that trigger data flows correctly into workflow execution
and is accessible via ${trigger_id.field} expressions.
"""
from __future__ import annotations

from typing import Any, List

import pytest

from .conftest import (
    compile_workflow,
    create_echo_tool,
    create_tracking_tool,
)


# =============================================================================
# TRIGGER DATA ACCESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_trigger_data_accessible_via_trigger_id() -> None:
    """
    Test that ${trigger_id.field} resolves to trigger envelope data.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "email_trigger",
                "key": "gmail.email_received",
                "mode": "polling",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "from": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "log_email",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "Subject: ${email_trigger.subject}"},
            },
            {
                "id": "log_from",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "From: ${email_trigger.from}"},
            },
        ],
        "edges": [
            {"source": "email_trigger", "target": "log_email", "type": "trigger"},
            {"source": "log_email", "target": "log_from", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "email_trigger",
        "trigger_key": "gmail.email_received",
        "subject": "Test Email Subject",
        "from": "sender@example.com",
        "body": "Email body content",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trigger data was accessible
    assert "Subject: Test Email Subject" in call_tracker
    assert "From: sender@example.com" in call_tracker
    assert result["log_email"] == "Subject: Test Email Subject"


@pytest.mark.asyncio
async def test_trigger_data_includes_all_envelope_fields() -> None:
    """
    Test that trigger_id, trigger_key, and custom data are all accessible.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)
    echo_tool = create_echo_tool()

    # Define schema with properly nested properties
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "webhook_trigger",
                "key": "custom.webhook",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "user_id": {"type": "string"},
                            },
                        },
                        "headers": {
                            "type": "object",
                            "properties": {
                                "content-type": {"type": "string"},
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "capture",
                "type": "tool",
                "tool": "test.echo",
                "inputs": {
                    "message": "${webhook_trigger.payload.action}",
                    "data": "${webhook_trigger.payload}",
                },
            },
            {
                "id": "log",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${capture.message}"},
            },
        ],
        "edges": [
            {"source": "webhook_trigger", "target": "capture", "type": "trigger"},
            {"source": "capture", "target": "log", "type": "default"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[echo_tool, tracking_tool])

    trigger_envelope = {
        "trigger_id": "webhook_trigger",
        "trigger_key": "custom.webhook",
        "payload": {
            "action": "user_signup",
            "user_id": "123",
        },
        "headers": {
            "content-type": "application/json",
        },
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify nested trigger data was accessible
    assert "user_signup" in call_tracker
    assert result["capture"]["message"] == "user_signup"
    assert result["capture"]["data"]["user_id"] == "123"


@pytest.mark.asyncio
async def test_multiple_triggers_route_to_correct_nodes() -> None:
    """
    Test that different trigger IDs invoke different entry points.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "email_trigger",
                "key": "gmail.email_received",
                "mode": "polling",
                "event_schema": {
                    "type": "object",
                    "properties": {"email_data": {"type": "string"}},
                },
            },
            {
                "id": "slack_trigger",
                "key": "slack.message_received",
                "mode": "polling",
                "event_schema": {
                    "type": "object",
                    "properties": {"slack_data": {"type": "string"}},
                },
            },
        ],
        "nodes": [
            {
                "id": "handle_email",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "email_handler_${email_trigger.email_data}"},
            },
            {
                "id": "handle_slack",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "slack_handler_${slack_trigger.slack_data}"},
            },
        ],
        "edges": [
            {"source": "email_trigger", "target": "handle_email", "type": "trigger"},
            {"source": "slack_trigger", "target": "handle_slack", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Test email trigger
    call_tracker.clear()
    email_envelope = {
        "trigger_id": "email_trigger",
        "trigger_key": "gmail.email_received",
        "email_data": "test_email",
    }
    await compiled.ainvoke(config=None, context=None, trigger=email_envelope)

    assert "email_handler_test_email" in call_tracker
    # Slack handler should not have been called
    assert not any("slack_handler" in str(c) for c in call_tracker)

    # Test slack trigger
    call_tracker.clear()
    slack_envelope = {
        "trigger_id": "slack_trigger",
        "trigger_key": "slack.message_received",
        "slack_data": "test_slack",
    }
    await compiled.ainvoke(config=None, context=None, trigger=slack_envelope)

    assert "slack_handler_test_slack" in call_tracker


@pytest.mark.asyncio
async def test_trigger_data_propagates_to_nested_loops() -> None:
    """
    Test that trigger data is accessible inside loop body via expression.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "batch_trigger",
                "key": "batch.process",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "loop",
                "type": "for_each",
                "items": "${batch_trigger.items}",
            },
            {
                "id": "process",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "${batch_trigger.batch_id}_${item}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "batch_${batch_trigger.batch_id}_complete"},
            },
        ],
        "edges": [
            {"source": "batch_trigger", "target": "loop", "type": "trigger"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "batch_trigger",
        "trigger_key": "batch.process",
        "batch_id": "BATCH001",
        "items": ["item_a", "item_b", "item_c"],
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify trigger data was accessible inside loop
    assert "BATCH001_item_a" in call_tracker
    assert "BATCH001_item_b" in call_tracker
    assert "BATCH001_item_c" in call_tracker
    assert "batch_BATCH001_complete" in call_tracker


@pytest.mark.asyncio
async def test_trigger_with_complex_event_schema() -> None:
    """
    Test trigger with complex nested event data structure.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "complex_trigger",
                "key": "test.complex",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "timestamp": {"type": "string"},
                            },
                        },
                        "data": {
                            "type": "object",
                            "properties": {
                                "users": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "log_source",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "Source: ${complex_trigger.metadata.source}"},
            },
            {
                "id": "loop_users",
                "type": "for_each",
                "items": "${complex_trigger.data.users}",
                "item_var": "user",
            },
            {
                "id": "log_user",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "User: ${user.name}"},
            },
            {
                "id": "done",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "done"},
            },
        ],
        "edges": [
            {"source": "complex_trigger", "target": "log_source", "type": "trigger"},
            {"source": "log_source", "target": "loop_users", "type": "default"},
            {"source": "loop_users", "target": "log_user", "type": "loop_body"},
            {"source": "loop_users", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    trigger_envelope = {
        "trigger_id": "complex_trigger",
        "trigger_key": "test.complex",
        "metadata": {
            "source": "api_gateway",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        "data": {
            "users": [
                {"name": "Alice", "email": "alice@example.com"},
                {"name": "Bob", "email": "bob@example.com"},
            ],
        },
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify complex nested access worked
    assert "Source: api_gateway" in call_tracker
    assert "User: Alice" in call_tracker
    assert "User: Bob" in call_tracker


@pytest.mark.asyncio
async def test_multiple_triggers_same_type() -> None:
    """
    Test multiple Gmail triggers with different IDs.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "work_email",
                "key": "gmail.email_received",
                "mode": "polling",
                "event_schema": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"}},
                },
            },
            {
                "id": "personal_email",
                "key": "gmail.email_received",
                "mode": "polling",
                "event_schema": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"}},
                },
            },
        ],
        "nodes": [
            {
                "id": "handle_work",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "work_${work_email.subject}"},
            },
            {
                "id": "handle_personal",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "personal_${personal_email.subject}"},
            },
        ],
        "edges": [
            {"source": "work_email", "target": "handle_work", "type": "trigger"},
            {"source": "personal_email", "target": "handle_personal", "type": "trigger"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Test work email trigger
    call_tracker.clear()
    work_envelope = {
        "trigger_id": "work_email",
        "trigger_key": "gmail.email_received",
        "subject": "Meeting Request",
    }
    await compiled.ainvoke(config=None, context=None, trigger=work_envelope)

    assert "work_Meeting Request" in call_tracker

    # Test personal email trigger
    call_tracker.clear()
    personal_envelope = {
        "trigger_id": "personal_email",
        "trigger_key": "gmail.email_received",
        "subject": "Family Photo",
    }
    await compiled.ainvoke(config=None, context=None, trigger=personal_envelope)

    assert "personal_Family Photo" in call_tracker


@pytest.mark.asyncio
async def test_trigger_data_in_conditional_check() -> None:
    """
    Test that trigger data can be used in if condition.
    """
    call_tracker: List[Any] = []
    tracking_tool = create_tracking_tool(call_tracker)

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "priority_trigger",
                "key": "test.priority",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "priority": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "check_priority",
                "type": "if",
                "condition": "${priority_trigger.priority} == 'urgent'",
            },
            {
                "id": "urgent_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "URGENT: ${priority_trigger.message}"},
            },
            {
                "id": "normal_handler",
                "type": "tool",
                "tool": "test.tracker",
                "inputs": {"value": "Normal: ${priority_trigger.message}"},
            },
        ],
        "edges": [
            {"source": "priority_trigger", "target": "check_priority", "type": "trigger"},
            {"source": "check_priority", "target": "urgent_handler", "type": "conditional_true"},
            {"source": "check_priority", "target": "normal_handler", "type": "conditional_false"},
        ],
    }

    compiled = await compile_workflow(spec, tool_defs=[tracking_tool])

    # Test urgent message
    call_tracker.clear()
    urgent_envelope = {
        "trigger_id": "priority_trigger",
        "trigger_key": "test.priority",
        "priority": "urgent",
        "message": "Server down!",
    }
    await compiled.ainvoke(config=None, context=None, trigger=urgent_envelope)

    assert "URGENT: Server down!" in call_tracker
    assert not any("Normal:" in str(c) for c in call_tracker)

    # Test normal message
    call_tracker.clear()
    normal_envelope = {
        "trigger_id": "priority_trigger",
        "trigger_key": "test.priority",
        "priority": "normal",
        "message": "Daily report",
    }
    await compiled.ainvoke(config=None, context=None, trigger=normal_envelope)

    assert "Normal: Daily report" in call_tracker
    assert not any("URGENT:" in str(c) for c in call_tracker)
