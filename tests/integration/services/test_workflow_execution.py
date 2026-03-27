"""
Integration tests for workflow execution service.

Tests execute_saved_workflow_run with real DB and mocked compiler/checkpointer.
Covers: status transitions, HITL interrupt persistence, error handling, resume validation.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.database import User, WorkflowRun, WorkflowRunStatus
from seer.database.workflow_models import (
    Workflow,
    WorkflowRunSource,
    WorkflowVersion,
    WorkflowVersionStatus,
)
from seer.services.workflows.execution import (
    _build_run_config,
    _calculate_interrupt_expiry,
    _extract_hitl_interrupt,
    _validate_resume_request,
    execute_saved_workflow_run,
)


# =============================================================================
# Pure Logic Helpers
# =============================================================================


@pytest.mark.integration
class TestExtractHitlInterrupt:
    """Tests for _extract_hitl_interrupt."""

    def test_no_interrupt(self):
        result = {"output": "done"}
        assert _extract_hitl_interrupt(result) is None

    def test_empty_interrupt_list(self):
        result = {"__interrupt__": []}
        assert _extract_hitl_interrupt(result) is None

    def test_hitl_interrupt_extracted(self):
        interrupt_obj = MagicMock()
        interrupt_obj.value = {"type": "hitl", "node_id": "approval", "title": "Approve"}
        result = {"__interrupt__": [interrupt_obj]}

        data = _extract_hitl_interrupt(result)
        assert data is not None
        assert data["type"] == "hitl"
        assert data["node_id"] == "approval"

    def test_non_hitl_interrupt_ignored(self):
        interrupt_obj = MagicMock()
        interrupt_obj.value = {"type": "other"}
        result = {"__interrupt__": [interrupt_obj]}

        assert _extract_hitl_interrupt(result) is None

    def test_interrupt_without_value_attribute(self):
        result = {"__interrupt__": ["raw_string"]}
        assert _extract_hitl_interrupt(result) is None


@pytest.mark.integration
class TestCalculateInterruptExpiry:
    """Tests for _calculate_interrupt_expiry."""

    def test_no_timeout_returns_none(self):
        assert _calculate_interrupt_expiry(None) is None
        assert _calculate_interrupt_expiry(0) is None

    def test_negative_timeout_returns_none(self):
        assert _calculate_interrupt_expiry(-1) is None

    def test_positive_timeout_returns_future_datetime(self):
        before = datetime.now(timezone.utc)
        result = _calculate_interrupt_expiry(3600)
        after = datetime.now(timezone.utc)

        assert result is not None
        assert result > before
        assert result <= after + timedelta(seconds=3601)


@pytest.mark.integration
class TestBuildRunConfig:
    """Tests for _build_run_config."""

    def test_sets_thread_id_from_run(self):
        run = MagicMock()
        run.thread_id = "thread-123"
        run.run_id = "run_456"

        config = _build_run_config(run)
        assert config["configurable"]["thread_id"] == "thread-123"

    def test_falls_back_to_run_id(self):
        run = MagicMock()
        run.thread_id = None
        run.run_id = "run_789"

        config = _build_run_config(run)
        assert config["configurable"]["thread_id"] == "run_789"

    def test_preserves_existing_config(self):
        run = MagicMock()
        run.thread_id = "t1"
        run.run_id = "r1"

        config = _build_run_config(run, {"recursion_limit": 50})
        assert config["recursion_limit"] == 50
        assert config["configurable"]["thread_id"] == "t1"


# =============================================================================
# Workflow Execution (Real DB)
# =============================================================================


@pytest.mark.integration
class TestExecuteSavedWorkflowRun:
    """Tests for execute_saved_workflow_run with real database."""

    @pytest.mark.asyncio
    async def test_successful_execution_marks_succeeded(self, db_engine, test_user):
        """Successful execution should set status to SUCCEEDED and store output."""
        workflow = await Workflow.create(user=test_user, name="Exec Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec=spec,
            status=WorkflowRunStatus.QUEUED,
            source=WorkflowRunSource.MANUAL,
        )

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"result": "success"}

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock, return_value=MagicMock()),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(return_value=mock_compiled)
            await execute_saved_workflow_run(run_id=run.id, user_id=test_user.id)

        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.SUCCEEDED
        assert run.output == {"result": "success"}
        assert run.finished_at is not None

    @pytest.mark.asyncio
    async def test_compilation_failure_marks_failed(self, db_engine, test_user):
        """Compilation error should set status to FAILED."""
        from seer.core.errors import WorkflowCompilerError

        workflow = await Workflow.create(user=test_user, name="Compile Fail Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec=spec,
            status=WorkflowRunStatus.QUEUED,
            source=WorkflowRunSource.MANUAL,
        )

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock, return_value=MagicMock()),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(
                side_effect=WorkflowCompilerError("Invalid node type"),
            )
            with pytest.raises(WorkflowCompilerError):
                await execute_saved_workflow_run(run_id=run.id, user_id=test_user.id)

        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.FAILED
        assert "Invalid node type" in run.error

    @pytest.mark.asyncio
    async def test_execution_sets_running_status(self, db_engine, test_user):
        """Execution should transition through RUNNING status."""
        workflow = await Workflow.create(user=test_user, name="Running Status Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec=spec,
            status=WorkflowRunStatus.QUEUED,
            source=WorkflowRunSource.MANUAL,
        )

        statuses_seen = []

        original_update = WorkflowRun.filter

        async def track_status_update(*args, **kwargs):
            qs = original_update(*args, **kwargs)
            return qs

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"done": True}

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock, return_value=MagicMock()),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(return_value=mock_compiled)
            await execute_saved_workflow_run(run_id=run.id, user_id=test_user.id)

        await run.refresh_from_db()
        # After execution, should be SUCCEEDED (passed through RUNNING)
        assert run.status == WorkflowRunStatus.SUCCEEDED
        assert run.started_at is not None

    @pytest.mark.asyncio
    async def test_hitl_interrupt_marks_interrupted(self, db_engine, test_user):
        """HITL interrupt should set status to INTERRUPTED with interrupt data."""
        workflow = await Workflow.create(user=test_user, name="HITL Interrupt Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec=spec,
            status=WorkflowRunStatus.QUEUED,
            source=WorkflowRunSource.MANUAL,
        )

        # Mock compiled workflow that returns HITL interrupt
        interrupt_value = MagicMock()
        interrupt_value.value = {
            "type": "hitl",
            "node_id": "approval",
            "title": "Approve Action",
            "timeout_seconds": 3600,
            "delivery_channels": [],
        }
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"__interrupt__": [interrupt_value]}

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock, return_value=MagicMock()),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(return_value=mock_compiled)
            await execute_saved_workflow_run(run_id=run.id, user_id=test_user.id)

        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.INTERRUPTED
        assert run.pending_interrupt_node_id == "approval"
        assert run.pending_interrupt_data is not None
        assert run.pending_interrupt_data["type"] == "hitl"
        assert run.interrupt_expires_at is not None

    @pytest.mark.asyncio
    async def test_runtime_exception_marks_failed(self, db_engine, test_user):
        """Runtime exception during execution should mark run as FAILED."""
        workflow = await Workflow.create(user=test_user, name="Runtime Error Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec=spec,
            status=WorkflowRunStatus.QUEUED,
            source=WorkflowRunSource.MANUAL,
        )

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.side_effect = RuntimeError("Tool API timeout")

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock, return_value=MagicMock()),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(return_value=mock_compiled)
            with pytest.raises(RuntimeError):
                await execute_saved_workflow_run(run_id=run.id, user_id=test_user.id)

        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.FAILED
        assert "Tool API timeout" in run.error


# =============================================================================
# Resume Validation (Real DB)
# =============================================================================


@pytest.mark.integration
class TestValidateResumeRequest:
    """Tests for _validate_resume_request with real database."""

    @pytest.mark.asyncio
    async def test_rejects_wrong_user(self, db_engine, test_user):
        """Resume by wrong user should be rejected."""
        workflow = await Workflow.create(user=test_user, name="Resume Auth Test")
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec={"version": "2"},
            status=WorkflowRunStatus.INTERRUPTED,
        )

        other_user = await User.create(
            user_id="other_user",
            email="other@example.com",
            first_name="Other",
            last_name="User",
        )

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(other_user, run)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_non_interrupted_run(self, db_engine, test_user):
        """Resume of non-interrupted run should fail."""
        workflow = await Workflow.create(user=test_user, name="Resume State Test")
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec={"version": "2"},
            status=WorkflowRunStatus.RUNNING,
        )

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(test_user, run)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_expired_interrupt(self, db_engine, test_user):
        """Resume of expired interrupt should fail with 408."""
        workflow = await Workflow.create(user=test_user, name="Expired Test")
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec={"version": "2"},
            status=WorkflowRunStatus.INTERRUPTED,
            interrupt_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(test_user, run)
        assert exc_info.value.status_code == 408

        # Should also update run to FAILED
        await run.refresh_from_db()
        assert run.status == WorkflowRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_accepts_valid_resume(self, db_engine, test_user):
        """Valid resume request should pass validation."""
        workflow = await Workflow.create(user=test_user, name="Valid Resume Test")
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec={"version": "2"},
            status=WorkflowRunStatus.INTERRUPTED,
            interrupt_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        # Should not raise
        await _validate_resume_request(test_user, run)
