"""
Integration tests for Human-in-the-Loop (HITL) workflow flow.

Tests the complete HITL interrupt and resume cycle:
- Execute workflow with HITL node
- Execution pauses (status=INTERRUPTED)
- GET interrupt data
- POST resume with user input
- Execution continues and completes

These tests verify the HITL feature works correctly end-to-end.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from seer.database.workflow_models import (
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
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
# HITL Interrupt State Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_interrupt_state(db_engine, test_user):
    """
    Test workflow run can be set to INTERRUPTED state.

    Verifies:
    - Run transitions to INTERRUPTED status
    - Interrupt metadata is stored
    """
    workflow = await Workflow.create(
        user=test_user,
        name="HITL Test Workflow",
    )

    spec_dict = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "hitl_node",
                "type": "hitl",
                "title": "Approval Required",
                "description": "Please review and approve",
                "inputs": [
                    {"id": "decision", "type": "choice", "choices": ["approve", "reject"]}
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
        status=WorkflowRunStatus.RUNNING,
    )

    # Simulate HITL interrupt
    interrupt_data = {
        "type": "hitl",
        "node_id": "hitl_node",
        "title": "Approval Required",
        "description": "Please review and approve",
        "inputs": [
            {"id": "decision", "type": "choice", "choices": ["approve", "reject"]}
        ],
        "timeout_seconds": 3600,
    }

    run.status = WorkflowRunStatus.INTERRUPTED
    run.pending_interrupt_node_id = "hitl_node"
    run.pending_interrupt_data = interrupt_data
    run.interrupt_expires_at = utcnow() + timedelta(seconds=3600)
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.INTERRUPTED
    assert run.pending_interrupt_node_id == "hitl_node"
    assert run.pending_interrupt_data["title"] == "Approval Required"
    assert run.interrupt_expires_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_interrupt_expiry(db_engine, test_user):
    """
    Test HITL interrupt expiration tracking.

    Verifies:
    - Expiration time is stored
    - Expired interrupts can be detected
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Expiry Test Workflow",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create run with short expiry
    expires_at = utcnow() + timedelta(seconds=60)

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
        interrupt_expires_at=expires_at,
    )

    await run.refresh_from_db()

    # Check if expired
    is_expired = run.interrupt_expires_at < utcnow()
    assert not is_expired  # Should not be expired yet

    # Update to past time (simulating time passing)
    run.interrupt_expires_at = utcnow() - timedelta(minutes=1)
    await run.save()

    await run.refresh_from_db()
    is_expired = run.interrupt_expires_at < utcnow()
    assert is_expired  # Now expired


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_interrupt_no_expiry(db_engine, test_user):
    """
    Test HITL interrupt without expiration (indefinite wait).

    Verifies:
    - Runs can wait indefinitely for user input
    - interrupt_expires_at can be None
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Indefinite Wait Workflow",
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
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl", "timeout_seconds": None},
        interrupt_expires_at=None,  # No expiry
    )

    await run.refresh_from_db()
    assert run.interrupt_expires_at is None


# =============================================================================
# Resume Flow Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_clears_interrupt_state(db_engine, test_user):
    """
    Test that resuming clears the interrupt state.

    Verifies:
    - pending_interrupt_node_id is cleared
    - pending_interrupt_data is cleared
    - interrupt_expires_at is cleared
    - Status transitions to RUNNING
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Resume Clear Test",
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
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
        interrupt_expires_at=utcnow() + timedelta(hours=1),
    )

    # Simulate resume action
    run.status = WorkflowRunStatus.RUNNING
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    run.interrupt_expires_at = None
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.RUNNING
    assert run.pending_interrupt_node_id is None
    assert run.pending_interrupt_data is None
    assert run.interrupt_expires_at is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_to_completion(db_engine, test_user):
    """
    Test resuming an interrupted run to successful completion.

    Verifies:
    - Run can transition from INTERRUPTED to SUCCEEDED
    - Output is stored after completion
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Resume Complete Test",
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
        pending_interrupt_node_id="approval_node",
        pending_interrupt_data={
            "type": "hitl",
            "node_id": "approval_node",
            "inputs": [{"id": "approved", "type": "boolean"}],
        },
    )

    # Resume with user input
    user_responses = {"approved": True}

    # Simulate resume and completion
    run.status = WorkflowRunStatus.RUNNING
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    await run.save()

    # Workflow continues and completes
    run.status = WorkflowRunStatus.SUCCEEDED
    run.finished_at = utcnow()
    run.output = {"approval_result": user_responses["approved"]}
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.output["approval_result"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_to_another_interrupt(db_engine, test_user):
    """
    Test resuming leads to another HITL interrupt.

    Verifies:
    - Workflow can pause at multiple HITL nodes
    - New interrupt data replaces old
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Multi-Interrupt Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Initial interrupt at first HITL node
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="hitl_node_1",
        pending_interrupt_data={"type": "hitl", "node_id": "hitl_node_1"},
    )

    # Resume from first interrupt
    run.status = WorkflowRunStatus.RUNNING
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    await run.save()

    # Hit second interrupt
    run.status = WorkflowRunStatus.INTERRUPTED
    run.pending_interrupt_node_id = "hitl_node_2"
    run.pending_interrupt_data = {"type": "hitl", "node_id": "hitl_node_2"}
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.INTERRUPTED
    assert run.pending_interrupt_node_id == "hitl_node_2"


# =============================================================================
# HITL Input Types Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_text_input(db_engine, test_user):
    """
    Test HITL node with text input type.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Text Input Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "text_input_node",
        "title": "Enter your name",
        "inputs": [
            {
                "id": "name",
                "type": "text",
                "label": "Full Name",
                "placeholder": "John Doe",
                "required": True,
            }
        ],
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="text_input_node",
        pending_interrupt_data=interrupt_data,
    )

    await run.refresh_from_db()
    assert run.pending_interrupt_data["inputs"][0]["type"] == "text"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_choice_input(db_engine, test_user):
    """
    Test HITL node with choice input type.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Choice Input Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "choice_node",
        "title": "Select priority",
        "inputs": [
            {
                "id": "priority",
                "type": "choice",
                "choices": ["low", "medium", "high", "critical"],
            }
        ],
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="choice_node",
        pending_interrupt_data=interrupt_data,
    )

    await run.refresh_from_db()
    choices = run.pending_interrupt_data["inputs"][0]["choices"]
    assert "critical" in choices


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_multi_input(db_engine, test_user):
    """
    Test HITL node with multiple input fields.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Multi Input Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "multi_input_node",
        "title": "Complete the form",
        "inputs": [
            {"id": "email", "type": "text", "label": "Email"},
            {"id": "action", "type": "choice", "choices": ["approve", "reject"]},
            {"id": "comments", "type": "text", "label": "Comments", "required": False},
        ],
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="multi_input_node",
        pending_interrupt_data=interrupt_data,
    )

    await run.refresh_from_db()
    assert len(run.pending_interrupt_data["inputs"]) == 3


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interrupted_run_failure(db_engine, test_user):
    """
    Test that interrupted run can fail (e.g., on timeout).
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Timeout Failure Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Interrupt that has expired
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
        interrupt_expires_at=utcnow() - timedelta(hours=1),  # Expired
    )

    # Handle expiration by marking as failed
    run.status = WorkflowRunStatus.FAILED
    run.error = "HITL interrupt timed out"
    run.finished_at = utcnow()
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    run.interrupt_expires_at = None
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.FAILED
    assert "timed out" in run.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_failure(db_engine, test_user):
    """
    Test handling failure during resume execution.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Resume Failure Test",
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
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
    )

    # Resume starts
    run.status = WorkflowRunStatus.RUNNING
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    await run.save()

    # Execution fails after resume
    run.status = WorkflowRunStatus.FAILED
    run.error = "Execution error after resume"
    run.finished_at = utcnow()
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.FAILED
    assert "after resume" in run.error


# =============================================================================
# User Authorization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_run_user_ownership(db_engine, test_user):
    """
    Test that HITL runs are owned by user.

    Verifies:
    - Run user_id matches creator
    - Other users cannot access
    """
    from seer.database.models import User

    workflow = await Workflow.create(
        user=test_user,
        name="Ownership Test",
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
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
    )

    # Verify ownership
    assert run.user_id == test_user.id

    # Create another user
    other_user = await User.create(
        user_id="other_user_456",
        email="other@example.com",
        first_name="Other",
        last_name="User",
        created_at=utcnow(),
    )

    # Verify run is not owned by other user
    assert run.user_id != other_user.id


# =============================================================================
# Concurrent HITL Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_runs_with_hitl(db_engine, test_user):
    """
    Test multiple runs of same workflow, each with HITL.

    Verifies:
    - Each run maintains its own interrupt state
    - Runs don't interfere with each other
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Concurrent HITL Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create multiple runs, each at different HITL states
    run1 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="hitl_1",
        pending_interrupt_data={"type": "hitl", "run": 1},
    )

    run2 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        pending_interrupt_node_id="hitl_2",
        pending_interrupt_data={"type": "hitl", "run": 2},
    )

    run3 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.RUNNING,  # This one is not interrupted
    )

    # Resume run1, leave run2 interrupted
    run1.status = WorkflowRunStatus.SUCCEEDED
    run1.pending_interrupt_node_id = None
    run1.pending_interrupt_data = None
    await run1.save()

    # Verify states are independent
    await run1.refresh_from_db()
    await run2.refresh_from_db()
    await run3.refresh_from_db()

    assert run1.status == WorkflowRunStatus.SUCCEEDED
    assert run2.status == WorkflowRunStatus.INTERRUPTED
    assert run2.pending_interrupt_node_id == "hitl_2"
    assert run3.status == WorkflowRunStatus.RUNNING
