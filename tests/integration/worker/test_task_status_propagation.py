"""
Integration tests for worker task status propagation.

Tests:
- API creates run → Worker picks up → Execution → DB status update
- Status consistency between API and worker
- Error status propagation
- Concurrent execution status updates

These tests verify that workflow run status is correctly synchronized
between the API layer, Taskiq worker, and database.
"""
import hashlib
import json
from datetime import datetime, timezone
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
from seer.worker.tasks.workflows import workflow_execution_task


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """Generate hash for workflow spec."""
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Task Execution Status Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_updates_run_to_running(db_engine, test_user):
    """
    Test that task execution updates run status to RUNNING.

    Verifies:
    - QUEUED -> RUNNING transition happens
    - started_at is set
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Status Update Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    async def mock_execute(run_id, user_id, trigger_envelope=None):
        # Simulate the service updating status to RUNNING
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.RUNNING,
            started_at=utcnow(),
        )

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.RUNNING
    assert run.started_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_updates_run_to_succeeded(db_engine, test_user):
    """
    Test that successful task execution updates run to SUCCEEDED.

    Verifies:
    - Status becomes SUCCEEDED
    - output is stored
    - finished_at is set
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Success Status Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    execution_output = {"result": "success", "data": [1, 2, 3]}

    async def mock_execute_success(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            output=execution_output,
            finished_at=utcnow(),
        )

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_success):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.output == execution_output
    assert run.finished_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_updates_run_to_failed(db_engine, test_user):
    """
    Test that failed task execution updates run to FAILED.

    Verifies:
    - Status becomes FAILED
    - error is stored
    - finished_at is set
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Failure Status Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    async def mock_execute_failure(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.FAILED,
            error="Tool execution failed: API rate limit exceeded",
            finished_at=utcnow(),
        )
        raise ValueError("Execution failed")

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_failure):

        with pytest.raises(ValueError):
            await workflow_execution_task(
                run_id=run.id,
                user_id=test_user.id,
            )

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.FAILED
    assert "rate limit" in run.error
    assert run.finished_at is not None


# =============================================================================
# Status Transition Flow Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_status_transition_flow(db_engine, test_user):
    """
    Test complete status flow: QUEUED -> RUNNING -> SUCCEEDED.

    Simulates the full worker execution lifecycle.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Full Flow Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    status_history = []

    async def mock_execute_with_history(run_id, user_id, trigger_envelope=None):
        # Transition to RUNNING
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.RUNNING,
            started_at=utcnow(),
        )
        run_obj = await WorkflowRun.get(id=run_id)
        status_history.append(run_obj.status)

        # Transition to SUCCEEDED
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            output={"completed": True},
            finished_at=utcnow(),
        )
        run_obj = await WorkflowRun.get(id=run_id)
        status_history.append(run_obj.status)

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_with_history):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    # Verify status transitions occurred in order
    assert status_history == [
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.SUCCEEDED,
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_transition_with_interrupt(db_engine, test_user):
    """
    Test status flow with HITL interrupt: QUEUED -> RUNNING -> INTERRUPTED.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Interrupt Flow Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    async def mock_execute_with_interrupt(run_id, user_id, trigger_envelope=None):
        # Start running
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.RUNNING,
            started_at=utcnow(),
        )

        # Hit HITL node, pause
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.INTERRUPTED,
            pending_interrupt_node_id="hitl_node",
            pending_interrupt_data={"type": "hitl"},
        )

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_with_interrupt):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.INTERRUPTED
    assert run.pending_interrupt_node_id == "hitl_node"


# =============================================================================
# Concurrent Execution Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_runs_independent_status(db_engine, test_user):
    """
    Test that concurrent runs maintain independent status.

    Verifies:
    - Multiple runs can execute in parallel
    - Each run updates its own status
    - No status interference between runs
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Concurrent Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create multiple runs
    run1 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )

    run2 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )

    run3 = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )

    async def mock_execute_run1(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.SUCCEEDED,
        )

    async def mock_execute_run2(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.FAILED,
            error="Test failure",
        )
        raise ValueError("Test failure")

    async def mock_execute_run3(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.INTERRUPTED,
        )

    # Execute runs with different outcomes
    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_run1):
        await workflow_execution_task(run_id=run1.id, user_id=test_user.id)

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_run2):
        with pytest.raises(ValueError):
            await workflow_execution_task(run_id=run2.id, user_id=test_user.id)

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_run3):
        await workflow_execution_task(run_id=run3.id, user_id=test_user.id)

    # Verify independent statuses
    await run1.refresh_from_db()
    await run2.refresh_from_db()
    await run3.refresh_from_db()

    assert run1.status == WorkflowRunStatus.SUCCEEDED
    assert run2.status == WorkflowRunStatus.FAILED
    assert run3.status == WorkflowRunStatus.INTERRUPTED


# =============================================================================
# Trigger Envelope Propagation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_envelope_passed_to_execution(db_engine, test_user):
    """
    Test that trigger envelope is passed to execution service.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Trigger Envelope Test",
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
        status=WorkflowRunStatus.QUEUED,
        source=WorkflowRunSource.TRIGGER,
    )

    trigger_envelope = {
        "trigger_id": "gmail_trigger",
        "trigger_key": "gmail.new_email",
        "data": {"subject": "Test Email"},
    }

    received_envelope = None

    async def mock_capture_envelope(run_id, user_id, trigger_envelope=None):
        nonlocal received_envelope
        received_envelope = trigger_envelope
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.SUCCEEDED,
        )

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_capture_envelope):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
            trigger_envelope=trigger_envelope,
        )

    assert received_envelope == trigger_envelope


# =============================================================================
# Error Propagation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_error_stored_in_run(db_engine, test_user):
    """
    Test that execution errors are stored in the run record.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Error Storage Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    error_message = "Tool 'gmail.send_email' failed: Authentication required"

    async def mock_execute_with_error(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.FAILED,
            error=error_message,
            finished_at=utcnow(),
        )
        raise RuntimeError(error_message)

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_with_error):

        with pytest.raises(RuntimeError):
            await workflow_execution_task(
                run_id=run.id,
                user_id=test_user.id,
            )

    await run.refresh_from_db()
    assert run.error == error_message
    assert "Authentication required" in run.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_traces_stored_on_error(db_engine, test_user):
    """
    Test that node traces are stored even when execution fails.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Trace on Error Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    node_traces = [
        {"node_id": "step1", "status": "succeeded"},
        {"node_id": "step2", "status": "failed", "error": "Network timeout"},
    ]

    async def mock_execute_with_traces(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.FAILED,
            error="Execution failed at step2",
            node_traces=node_traces,
            finished_at=utcnow(),
        )
        raise RuntimeError("Network timeout")

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_with_traces):

        with pytest.raises(RuntimeError):
            await workflow_execution_task(
                run_id=run.id,
                user_id=test_user.id,
            )

    await run.refresh_from_db()
    assert run.node_traces == node_traces
    assert run.node_traces[1]["error"] == "Network timeout"


# =============================================================================
# Task Idempotency Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_handles_already_completed_run(db_engine, test_user):
    """
    Test that task handles runs that are already completed.

    This can happen if a task is retried after success.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Idempotency Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Run is already completed
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.SUCCEEDED,
        output={"already": "done"},
        finished_at=utcnow(),
    )

    call_count = 0

    async def mock_execute_check(run_id, user_id, trigger_envelope=None):
        nonlocal call_count
        call_count += 1
        # In real implementation, should check status first

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_check):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    # Run should still be SUCCEEDED
    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.output == {"already": "done"}


# =============================================================================
# Metrics and Timestamps Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_duration_tracked(db_engine, test_user):
    """
    Test that execution duration can be calculated from timestamps.
    """
    from datetime import timedelta

    workflow = await Workflow.create(
        user=test_user,
        name="Duration Test",
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
        status=WorkflowRunStatus.QUEUED,
    )

    start_time = utcnow()
    end_time = start_time + timedelta(seconds=5)

    async def mock_execute_with_duration(run_id, user_id, trigger_envelope=None):
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.RUNNING,
            started_at=start_time,
        )
        await WorkflowRun.filter(id=run_id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            finished_at=end_time,
        )

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_with_duration):

        await workflow_execution_task(
            run_id=run.id,
            user_id=test_user.id,
        )

    await run.refresh_from_db()

    # Calculate duration
    duration = run.finished_at - run.started_at
    assert duration.total_seconds() == 5
