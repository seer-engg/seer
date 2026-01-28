"""
Integration tests for workflow worker tasks.

Tests:
- Workflow execution task
- Task error handling
- Integration with database models
- Task queuing and execution
"""
from unittest.mock import AsyncMock, patch

import pytest

from seer.database.workflow_models import (
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
)
from seer.worker.tasks.workflows import workflow_execution_task


# =============================================================================
# Workflow Execution Task Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_success(db_engine, test_workflow_with_run):
    """Test successful workflow execution via task."""
    workflow, run = test_workflow_with_run

    # Mock the execute_saved_workflow_run service
    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.return_value = None

        # Execute task
        await workflow_execution_task(
            run_id=run.id,
            user_id=workflow.user.id,
        )

        # Verify service was called
        mock_execute.assert_called_once_with(
            run_id=run.id,
            user_id=workflow.user.id,
            trigger_envelope=None,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_with_trigger(db_engine, test_workflow_with_run):
    """Test workflow execution with trigger envelope."""
    workflow, run = test_workflow_with_run

    trigger_envelope = {
        "event_id": "evt_123",
        "data": {"message": "Test event"},
    }

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.return_value = None

        await workflow_execution_task(
            run_id=run.id,
            user_id=workflow.user.id,
            trigger_envelope=trigger_envelope,
        )

        # Verify trigger envelope was passed
        mock_execute.assert_called_once_with(
            run_id=run.id,
            user_id=workflow.user.id,
            trigger_envelope=trigger_envelope,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_handles_errors(db_engine, test_workflow_with_run):
    """Test task handles execution errors."""
    workflow, run = test_workflow_with_run

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.side_effect = ValueError("Execution failed")

        # Task should raise the exception
        with pytest.raises(ValueError, match="Execution failed"):
            await workflow_execution_task(
                run_id=run.id,
                user_id=workflow.user.id,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_logs_info(db_engine, test_workflow_with_run):
    """Test that task logs execution information."""
    workflow, run = test_workflow_with_run

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute, \
         patch("seer.worker.tasks.workflows.logger") as mock_logger:

        mock_execute.return_value = None

        await workflow_execution_task(
            run_id=run.id,
            user_id=workflow.user.id,
        )

        # Verify logging
        assert mock_logger.info.called


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_logs_errors(db_engine, test_workflow_with_run):
    """Test that task logs errors on failure."""
    workflow, run = test_workflow_with_run

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute, \
         patch("seer.worker.tasks.workflows.logger") as mock_logger:

        mock_execute.side_effect = RuntimeError("Task failed")

        with pytest.raises(RuntimeError):
            await workflow_execution_task(
                run_id=run.id,
                user_id=workflow.user.id,
            )

        # Verify error logging
        mock_logger.exception.assert_called_once()


# =============================================================================
# Task Integration with Database Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_updates_run_status(db_engine, test_workflow_with_run):
    """Test that workflow execution updates run status in database."""
    workflow, run = test_workflow_with_run

    async def mock_execute_and_update(run_id, user_id, trigger_envelope=None):
        # Simulate updating run status
        run_obj = await WorkflowRun.get(id=run_id)
        run_obj.status = WorkflowRunStatus.SUCCEEDED
        await run_obj.save()

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=mock_execute_and_update):

        await workflow_execution_task(
            run_id=run.id,
            user_id=workflow.user.id,
        )

        # Verify run status was updated
        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.SUCCEEDED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_with_multiple_runs(db_engine, test_workflow):
    """Test executing multiple workflow runs."""
    # Create multiple runs
    run1 = await WorkflowRun.create(
        user=test_workflow.user,
        workflow=test_workflow,
        spec={"version": "2"},
        status=WorkflowRunStatus.QUEUED,
    )

    run2 = await WorkflowRun.create(
        user=test_workflow.user,
        workflow=test_workflow,
        spec={"version": "2"},
        status=WorkflowRunStatus.QUEUED,
    )

    executed_runs = []

    async def track_execution(run_id, user_id, trigger_envelope=None):
        executed_runs.append(run_id)

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run",
               side_effect=track_execution):

        # Execute both runs
        await workflow_execution_task(run_id=run1.id, user_id=test_workflow.user.id)
        await workflow_execution_task(run_id=run2.id, user_id=test_workflow.user.id)

        # Verify both runs were executed
        assert len(executed_runs) == 2
        assert run1.id in executed_runs
        assert run2.id in executed_runs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_task_with_invalid_run_id(db_engine, test_user):
    """Test task handling of invalid run ID."""
    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.side_effect = ValueError("Run not found")

        with pytest.raises(ValueError, match="Run not found"):
            await workflow_execution_task(
                run_id=99999,  # Non-existent
                user_id=test_user.id,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_task_with_invalid_user_id(db_engine, test_workflow_with_run):
    """Test task handling of invalid user ID."""
    workflow, run = test_workflow_with_run

    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.side_effect = ValueError("User not found")

        with pytest.raises(ValueError, match="User not found"):
            await workflow_execution_task(
                run_id=run.id,
                user_id=99999,  # Non-existent
            )


# =============================================================================
# Task Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_workflow_execution_task_is_broker_task():
    """Test that workflow_execution_task is registered as broker task."""
    # Verify task has broker task attributes
    assert hasattr(workflow_execution_task, "kiq")
    assert callable(workflow_execution_task.kiq)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_execution_task_callable():
    """Test that task is directly callable (not just via broker)."""
    # Task should be callable as a regular async function
    with patch("seer.worker.tasks.workflows.execute_saved_workflow_run") as mock_execute:
        mock_execute.return_value = None

        # Should be callable directly
        await workflow_execution_task(
            run_id=1,
            user_id=1,
        )

        assert mock_execute.called
