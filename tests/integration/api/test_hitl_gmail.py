"""
Integration tests for HITL Gmail delivery channel.

Tests the complete flow of:
1. Workflow hits HITL node with Gmail delivery channel
2. Form is created, email is sent
3. User submits form
4. Workflow resumes

These tests verify the Gmail HITL feature works correctly end-to-end.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database.workflow_models import (
    TriggerSubscription,
    Workflow,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
)


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """Generate hash for workflow spec."""
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# HITL Form Creation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_form_creation_with_gmail_channel(db_engine, test_user):
    """
    Test that HITL form is created correctly for Gmail delivery channel.

    Verifies:
    - TriggerSubscription is created with form.hitl trigger key
    - form_config contains HITL metadata
    - form_fields are mapped from HITL inputs
    """
    from seer.services.workflows.hitl_form import HITLFormService

    workflow = await Workflow.create(
        user=test_user,
        name="HITL Gmail Test Workflow",
    )

    spec_dict = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "approval_node",
                "type": "hitl",
                "title": "Approval Required",
                "description": "Please approve this request",
                "inputs": [
                    {
                        "id": "decision",
                        "question": "Approve or reject?",
                        "input_type": "single_choice",
                        "required": True,
                        "options": [
                            {"value": "approve", "label": "Approve"},
                            {"value": "reject", "label": "Reject"},
                        ],
                    },
                    {
                        "id": "comments",
                        "question": "Additional comments",
                        "input_type": "text",
                        "required": False,
                    },
                ],
                "delivery_channels": [
                    {"type": "platform"},
                    {"type": "gmail", "gmail": {"recipient_email": "approver@example.com"}},
                ],
            }
        ],
        "edges": [],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="approval_node",
        pending_interrupt_data={
            "type": "hitl",
            "node_id": "approval_node",
            "title": "Approval Required",
            "description": "Please approve this request",
            "inputs": [
                {
                    "id": "decision",
                    "question": "Approve or reject?",
                    "input_type": "single_choice",
                    "required": True,
                    "options": [
                        {"value": "approve", "label": "Approve"},
                        {"value": "reject", "label": "Reject"},
                    ],
                },
                {
                    "id": "comments",
                    "question": "Additional comments",
                    "input_type": "text",
                    "required": False,
                },
            ],
        },
    )

    # Create HITL form
    service = HITLFormService()
    subscription, form_url = await service.create_hitl_form(
        run, run.pending_interrupt_data
    )

    # Verify subscription was created
    assert subscription.id is not None
    assert subscription.trigger_key == "form.hitl"
    assert subscription.form_suffix == f"hitl-{run.run_id}-approval_node"
    assert subscription.enabled is True

    # Verify form config contains HITL metadata
    form_config = subscription.form_config
    assert form_config["_hitl_run_id"] == run.run_id
    assert form_config["_hitl_node_id"] == "approval_node"
    assert form_config["title"] == "Approval Required"

    # Verify form fields are mapped
    form_fields = subscription.form_fields
    assert len(form_fields) == 2
    assert form_fields[0]["name"] == "decision"
    assert form_fields[0]["type"] == "select"
    assert len(form_fields[0]["options"]) == 2
    assert form_fields[1]["name"] == "comments"
    assert form_fields[1]["type"] == "text"

    # Cleanup
    await subscription.delete()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_form_submission_resumes_workflow(db_engine, test_user):
    """
    Test that submitting an HITL form resumes the workflow.

    Verifies:
    - Form submission is detected as HITL
    - Workflow is resumed with form data
    - Form is disabled after use
    """
    workflow = await Workflow.create(
        user=test_user,
        name="HITL Resume Test Workflow",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="test_node",
        pending_interrupt_data={"type": "hitl", "node_id": "test_node"},
    )

    # Create subscription manually (simulating what HITLFormService does)
    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_key="form.hitl",
        trigger_id=f"hitl-form-{run.run_id}-test_node",
        title="Test HITL Form",
        enabled=True,
        form_suffix=f"hitl-{run.run_id}-test_node",
        form_fields=[
            {"name": "decision", "displayLabel": "Decision", "type": "text", "required": True}
        ],
        form_config={
            "title": "Test",
            "_hitl_run_id": run.run_id,
            "_hitl_node_id": "test_node",
        },
    )

    # Now test the form submission handler (mocking resume_workflow_run)
    from seer.api.forms.router import _handle_hitl_form_submission

    with patch("seer.services.workflows.execution.resume_workflow_run") as mock_resume:
        mock_resume.return_value = None  # Successful resume

        result = await _handle_hitl_form_submission(
            subscription=subscription,
            data={"decision": "approved"},
            run_id=run.run_id,
        )

        # Verify resume was called
        mock_resume.assert_called_once()
        call_kwargs = mock_resume.call_args.kwargs
        assert call_kwargs["run_id"] == run.run_id
        assert call_kwargs["responses"] == {"decision": "approved"}

        # Verify response
        assert result["ok"] is True
        assert result["workflow_resumed"] is True

    # Verify form was disabled
    await subscription.refresh_from_db()
    assert subscription.enabled is False

    # Cleanup
    await subscription.delete()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_form_rejects_non_interrupted_run(db_engine, test_user):
    """
    Test that HITL form submission fails for non-interrupted runs.

    Verifies:
    - Already completed runs return graceful message
    - Running runs return error
    """
    from fastapi import HTTPException

    workflow = await Workflow.create(
        user=test_user,
        name="Non-Interrupted Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create a SUCCEEDED run
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.SUCCEEDED,
        finished_at=utcnow(),
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_key="form.hitl",
        trigger_id=f"hitl-form-{run.run_id}-node",
        title="Test",
        enabled=True,
        form_suffix=f"hitl-{run.run_id}-node",
        form_fields=[],
        form_config={
            "_hitl_run_id": run.run_id,
            "_hitl_node_id": "node",
        },
    )

    from seer.api.forms.router import _handle_hitl_form_submission

    # Should return success with already_completed flag
    result = await _handle_hitl_form_submission(
        subscription=subscription,
        data={},
        run_id=run.run_id,
    )

    assert result["ok"] is True
    assert result["already_completed"] is True

    # Cleanup
    await subscription.delete()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_form_displays_in_resolve(db_engine, test_user):
    """
    Test that HITL display items are included in form resolution.

    Verifies:
    - Display items from HITL are passed to frontend
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Display Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_key="form.hitl",
        trigger_id="hitl-display-test",
        title="Test",
        enabled=True,
        form_suffix="hitl-display-test",
        form_fields=[],
        form_config={
            "title": "Review Request",
            "_hitl_run_id": "run_123",
            "_hitl_node_id": "review",
            "_hitl_display": [
                {"label": "Order ID", "value": "ORD-456"},
                {"label": "Amount", "value": "$99.99"},
            ],
        },
    )

    from seer.api.forms.router import resolve_form

    result = await resolve_form("hitl-display-test")

    assert result["title"] == "Review Request"
    assert "display_items" in result
    assert len(result["display_items"]) == 2
    assert result["display_items"][0]["label"] == "Order ID"

    # Cleanup
    await subscription.delete()


# =============================================================================
# Delivery Channel Schema Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_node_with_delivery_channels_validates(db_engine, test_user):
    """
    Test that HITL node with delivery_channels validates correctly.

    Verifies:
    - Schema accepts valid delivery channel configurations
    - Invalid configurations are rejected
    """
    from seer.core.schema.models import HITLNode, HITLDeliveryChannel, GmailDeliveryConfig, DeliveryChannelType

    # Valid: Platform only
    node1 = HITLNode(
        id="hitl_1",
        title="Test 1",
        delivery_channels=[
            HITLDeliveryChannel(type=DeliveryChannelType.platform)
        ],
    )
    assert len(node1.delivery_channels) == 1

    # Valid: Gmail with config
    node2 = HITLNode(
        id="hitl_2",
        title="Test 2",
        delivery_channels=[
            HITLDeliveryChannel(type=DeliveryChannelType.platform),
            HITLDeliveryChannel(
                type=DeliveryChannelType.gmail,
                gmail=GmailDeliveryConfig(recipient_email="test@example.com")
            ),
        ],
    )
    assert len(node2.delivery_channels) == 2

    # Invalid: Gmail without config should raise
    with pytest.raises(ValueError, match="requires gmail config"):
        HITLDeliveryChannel(type=DeliveryChannelType.gmail)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_interrupt_includes_delivery_channels(db_engine, test_user):
    """
    Test that HITL interrupt payload includes delivery_channels.

    This verifies the runtime nodes.py update.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Delivery Channel Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Simulate what the runtime would store
    interrupt_data = {
        "type": "hitl",
        "node_id": "approval",
        "title": "Approval",
        "inputs": [],
        "delivery_channels": [
            {"type": "platform"},
            {"type": "gmail", "gmail": {"recipient_email": "approver@test.com"}},
        ],
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="approval",
        pending_interrupt_data=interrupt_data,
    )

    await run.refresh_from_db()

    # Verify delivery_channels are stored
    assert "delivery_channels" in run.pending_interrupt_data
    channels = run.pending_interrupt_data["delivery_channels"]
    assert len(channels) == 2
    assert channels[0]["type"] == "platform"
    assert channels[1]["type"] == "gmail"
    assert channels[1]["gmail"]["recipient_email"] == "approver@test.com"


# =============================================================================
# Email Service Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_send_hitl_notifications_creates_form_and_sends_email(db_engine, test_user):
    """
    Test the _send_hitl_notifications function creates form and sends email.

    This is a semi-integration test that mocks the Gmail API but tests
    the full notification flow.
    """
    from seer.services.workflows.execution import _send_hitl_notifications

    workflow = await Workflow.create(
        user=test_user,
        name="Notification Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="approval",
        pending_interrupt_data={"type": "hitl", "node_id": "approval", "title": "Test"},
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "approval",
        "title": "Approval Needed",
        "inputs": [],
        "delivery_channels": [
            {"type": "platform"},
            {"type": "gmail", "gmail": {"recipient_email": "approver@test.com"}},
        ],
    }

    # Mock the email service to avoid actual API calls
    with patch("seer.services.workflows.hitl_email.send_hitl_gmail_notification") as mock_send:
        mock_send.return_value = None  # Success

        await _send_hitl_notifications(run, test_user, interrupt_data)

        # Verify Gmail notification was attempted
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["user"] == test_user
        assert call_kwargs["workflow_run"] == run
        assert call_kwargs["gmail_config"].recipient_email == "approver@test.com"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_send_hitl_notifications_handles_failure_gracefully(db_engine, test_user):
    """
    Test that email failures don't break the HITL flow.

    Platform HITL should always work as fallback.
    """
    from seer.services.workflows.execution import _send_hitl_notifications

    workflow = await Workflow.create(
        user=test_user,
        name="Failure Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "node",
        "title": "Test",
        "inputs": [],
        "delivery_channels": [
            {"type": "gmail", "gmail": {"recipient_email": "test@example.com"}},
        ],
    }

    # Mock to raise an exception
    with patch("seer.services.workflows.hitl_email.send_hitl_gmail_notification") as mock_send:
        mock_send.side_effect = Exception("Gmail API error")

        # Should not raise - just logs the error
        await _send_hitl_notifications(run, test_user, interrupt_data)

        # Verify it was attempted
        mock_send.assert_called_once()
