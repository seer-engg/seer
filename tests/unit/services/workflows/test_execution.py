"""
Unit tests for workflow execution service.

Tests the core workflow execution logic including:
- HITL interrupt extraction and handling
- State transitions (QUEUED → RUNNING → SUCCEEDED/FAILED/INTERRUPTED)
- Error handling paths (compilation, runtime, cost cap)
- Timeout scenarios for HITL interrupts
- Resume workflow functionality
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_workflow_run():
    """
    Factory fixture for creating configurable WorkflowRun mocks.

    Returns a factory function that creates runs with specified state.
    """
    from typing import Optional
    from seer.database import WorkflowRunStatus

    def _create_run(
        id: int = 1,
        run_id: str = "run_1",
        status: str = "queued",
        user_id: int = 1,
        spec: Optional[dict] = None,
        inputs: Optional[dict] = None,
        config: Optional[dict] = None,
        thread_id: Optional[str] = None,
        pending_interrupt_node_id: Optional[str] = None,
        pending_interrupt_data: Optional[dict] = None,
        interrupt_expires_at: Optional[datetime] = None,
    ):
        run = MagicMock()
        run.id = id
        run.run_id = run_id
        run.status = WorkflowRunStatus(status)
        run.user_id = user_id
        run.spec = spec or {"version": "2", "nodes": [], "edges": []}
        run.inputs = inputs or {}
        run.config = config or {}
        run.thread_id = thread_id
        run.pending_interrupt_node_id = pending_interrupt_node_id
        run.pending_interrupt_data = pending_interrupt_data
        run.interrupt_expires_at = interrupt_expires_at
        run.workflow = MagicMock()
        run.workflow.workflow_id = "wf_1"
        run.user = MagicMock()
        run.user.id = user_id
        run.fetch_related = AsyncMock()
        run.refresh_from_db = AsyncMock()
        return run

    return _create_run


@pytest.fixture
def mock_compiled_workflow():
    """
    Factory fixture for creating CompiledWorkflow mocks.

    Returns a factory that creates compiled workflows with configurable ainvoke result.
    """
    from typing import Optional

    def _create(result: Optional[dict] = None, side_effect=None):
        compiled = MagicMock()
        if side_effect:
            compiled.ainvoke = AsyncMock(side_effect=side_effect)
        else:
            compiled.ainvoke = AsyncMock(return_value=result or {"output": "success"})
        return compiled

    return _create


@pytest.fixture
def frozen_time():
    """
    Fixture to freeze _now() for deterministic time testing.

    Yields the fixed datetime that _now() will return.
    """
    fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch("seer.services.workflows.execution._now", return_value=fixed_time):
        yield fixed_time


@pytest.fixture
def mock_hitl_interrupt():
    """Standard HITL interrupt data structure for testing."""
    return {
        "type": "hitl",
        "node_id": "approval_node",
        "title": "Approval Required",
        "description": "Please approve this action",
        "inputs": [{"id": "decision", "question": "Approve?", "input_type": "boolean"}],
        "timeout_seconds": 3600,
        "delivery_channels": [{"type": "platform"}],
    }


@pytest.fixture
def mock_interrupt_object():
    """Factory for creating LangGraph Interrupt object mocks."""
    def _create(value: dict):
        interrupt = MagicMock()
        interrupt.value = value
        return interrupt

    return _create


@pytest.fixture
def mock_user_settings():
    """Factory for creating UserSettings mocks with configurable preferences."""
    def _create(cost_cap: float = 5.0):
        settings = MagicMock()
        settings.preferences = {"per_run_cost_cap_usd": cost_cap}
        return settings

    return _create


# =============================================================================
# TestExtractHitlInterrupt - HITL extraction edge cases
# =============================================================================


@pytest.mark.unit
class TestExtractHitlInterrupt:
    """Tests for HITL interrupt extraction from workflow results."""

    def test_returns_none_when_no_interrupt_key(self):
        """Result without __interrupt__ key returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "data": {"value": 42}}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_none(self):
        """Result with __interrupt__ = None returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": None}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_empty_tuple(self):
        """Result with empty interrupt tuple returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": ()}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_empty_list(self):
        """Result with empty interrupt list returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": []}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_has_non_hitl_type(self, mock_interrupt_object):
        """Interrupt with type != 'hitl' returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object({"type": "tool_call", "data": "something"})
        result = {"__interrupt__": (interrupt,)}
        assert _extract_hitl_interrupt(result) is None

    def test_extracts_hitl_interrupt_from_single_interrupt(
        self, mock_interrupt_object, mock_hitl_interrupt
    ):
        """Single HITL interrupt in tuple is extracted correctly."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object(mock_hitl_interrupt)
        result = {"__interrupt__": (interrupt,)}

        extracted = _extract_hitl_interrupt(result)

        assert extracted is not None
        assert extracted["type"] == "hitl"
        assert extracted["node_id"] == "approval_node"
        assert extracted["title"] == "Approval Required"

    def test_extracts_first_hitl_from_multiple_interrupts(self, mock_interrupt_object):
        """First HITL interrupt found in multiple interrupts."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        non_hitl = mock_interrupt_object({"type": "tool_call"})
        hitl_1 = mock_interrupt_object({"type": "hitl", "node_id": "first_hitl"})
        hitl_2 = mock_interrupt_object({"type": "hitl", "node_id": "second_hitl"})

        result = {"__interrupt__": (non_hitl, hitl_1, hitl_2)}

        extracted = _extract_hitl_interrupt(result)

        assert extracted is not None
        assert extracted["node_id"] == "first_hitl"

    def test_handles_interrupt_object_without_value_attr(self):
        """Gracefully handles Interrupt objects missing value attribute."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = MagicMock(spec=[])  # No value attribute
        del interrupt.value  # Ensure it doesn't exist
        result = {"__interrupt__": (interrupt,)}

        assert _extract_hitl_interrupt(result) is None

    def test_handles_non_dict_interrupt_value(self, mock_interrupt_object):
        """Interrupt with non-dict value returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object("just a string")
        result = {"__interrupt__": (interrupt,)}

        assert _extract_hitl_interrupt(result) is None


# =============================================================================
# TestCalculateInterruptExpiry - Timeout calculation
# =============================================================================


@pytest.mark.unit
class TestCalculateInterruptExpiry:
    """Tests for interrupt timeout calculation."""

    def test_returns_none_when_timeout_is_none(self):
        """None timeout returns None (indefinite wait)."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(None) is None

    def test_returns_none_when_timeout_is_zero(self):
        """Zero timeout returns None (indefinite wait)."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(0) is None

    def test_returns_none_when_timeout_is_negative(self):
        """Negative timeout returns None."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(-10) is None
        assert _calculate_interrupt_expiry(-3600) is None

    def test_calculates_future_datetime_for_positive_timeout(self, frozen_time):
        """Positive timeout returns datetime in the future."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        expiry = _calculate_interrupt_expiry(3600)  # 1 hour

        assert expiry is not None
        assert expiry == frozen_time + timedelta(seconds=3600)

    def test_expiry_uses_utc_timezone(self, frozen_time):
        """Returned datetime is UTC timezone-aware."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        expiry = _calculate_interrupt_expiry(60)

        assert expiry.tzinfo == timezone.utc

    @pytest.mark.parametrize(
        "timeout_seconds,expected_none",
        [
            (None, True),
            (0, True),
            (-1, True),
            (-100, True),
            (1, False),
            (60, False),
            (3600, False),
            (86400, False),
        ],
    )
    def test_parametrized_timeout_values(self, timeout_seconds, expected_none, frozen_time):
        """Parametrized test for various timeout values."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        result = _calculate_interrupt_expiry(timeout_seconds)

        if expected_none:
            assert result is None
        else:
            assert result is not None
            assert result == frozen_time + timedelta(seconds=timeout_seconds)


# =============================================================================
# TestBuildRunConfig - LangGraph config construction
# =============================================================================


@pytest.mark.unit
class TestBuildRunConfig:
    """Tests for run configuration building."""

    def test_creates_config_with_thread_id_from_run_id(self, mock_workflow_run):
        """Config uses run.run_id as thread_id when thread_id is None."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123", thread_id=None)

        config = _build_run_config(run)

        assert config["configurable"]["thread_id"] == "run_123"

    def test_uses_run_thread_id_when_present(self, mock_workflow_run):
        """Config uses run.thread_id when it's set."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123", thread_id="custom_thread_456")

        config = _build_run_config(run)

        assert config["configurable"]["thread_id"] == "custom_thread_456"

    def test_overrides_existing_thread_id_in_payload(self, mock_workflow_run):
        """Always overrides thread_id from config_payload."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")
        config_payload = {"configurable": {"thread_id": "old_thread"}}

        config = _build_run_config(run, config_payload)

        # Should override with run.run_id
        assert config["configurable"]["thread_id"] == "run_123"

    def test_preserves_other_configurable_keys(self, mock_workflow_run):
        """Other keys in configurable section are preserved."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")
        config_payload = {
            "configurable": {"other_key": "other_value", "recursion_limit": 50}
        }

        config = _build_run_config(run, config_payload)

        assert config["configurable"]["thread_id"] == "run_123"
        assert config["configurable"]["other_key"] == "other_value"
        assert config["configurable"]["recursion_limit"] == 50

    def test_handles_none_config_payload(self, mock_workflow_run):
        """Handles None config_payload gracefully."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")

        config = _build_run_config(run, None)

        assert config["configurable"]["thread_id"] == "run_123"

    def test_handles_empty_config_payload(self, mock_workflow_run):
        """Handles empty dict config_payload."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")

        config = _build_run_config(run, {})

        assert config["configurable"]["thread_id"] == "run_123"


# =============================================================================
# TestCompileWorkflow - Compiler singleton interaction
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCompileWorkflow:
    """Tests for workflow compilation via singleton."""

    async def test_calls_compiler_singleton_with_correct_args(self, mock_user):
        """Verifies compiler.compile is called with user, spec, checkpointer."""
        from seer.services.workflows.execution import _compile_workflow

        mock_checkpointer = MagicMock()
        spec = {"version": "2", "nodes": [], "edges": []}

        with patch(
            "seer.services.workflows.execution.WorkflowCompilerSingleton"
        ) as mock_singleton:
            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(return_value=MagicMock())
            mock_singleton.instance.return_value = mock_compiler

            await _compile_workflow(mock_user, spec, checkpointer=mock_checkpointer)

            mock_compiler.compile.assert_called_once_with(
                mock_user, spec, checkpointer=mock_checkpointer
            )

    async def test_returns_compiled_workflow(self, mock_user, mock_compiled_workflow):
        """Returns the compiled workflow from singleton."""
        from seer.services.workflows.execution import _compile_workflow

        spec = {"version": "2", "nodes": [], "edges": []}
        expected_compiled = mock_compiled_workflow(result={"test": "output"})

        with patch(
            "seer.services.workflows.execution.WorkflowCompilerSingleton"
        ) as mock_singleton:
            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(return_value=expected_compiled)
            mock_singleton.instance.return_value = mock_compiler

            result = await _compile_workflow(mock_user, spec)

            assert result == expected_compiled

    async def test_passes_none_checkpointer_when_not_provided(self, mock_user):
        """Checkpointer defaults to None when not provided."""
        from seer.services.workflows.execution import _compile_workflow

        spec = {"version": "2", "nodes": [], "edges": []}

        with patch(
            "seer.services.workflows.execution.WorkflowCompilerSingleton"
        ) as mock_singleton:
            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(return_value=MagicMock())
            mock_singleton.instance.return_value = mock_compiler

            await _compile_workflow(mock_user, spec)

            mock_compiler.compile.assert_called_once_with(
                mock_user, spec, checkpointer=None
            )


# =============================================================================
# TestMarkRunSucceeded - Success state persistence
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestMarkRunSucceeded:
    """Tests for marking run as succeeded."""

    async def test_updates_run_status_to_succeeded(self, mock_workflow_run, frozen_time):
        """Run status is updated to SUCCEEDED."""
        from seer.services.workflows.execution import _mark_run_succeeded
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()
        output = {"result": "success"}

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            await _mark_run_succeeded(run, output)

            mock_run_model.filter.assert_called_once_with(id=run.id)
            mock_filter.update.assert_called_once()
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == WorkflowRunStatus.SUCCEEDED

    async def test_sets_finished_at_timestamp(self, mock_workflow_run, frozen_time):
        """Sets finished_at to current UTC time."""
        from seer.services.workflows.execution import _mark_run_succeeded

        run = mock_workflow_run()
        output = {"result": "success"}

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            await _mark_run_succeeded(run, output)

            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["finished_at"] == frozen_time

    async def test_stores_output_json(self, mock_workflow_run, frozen_time):
        """Output dict is stored in run.output."""
        from seer.services.workflows.execution import _mark_run_succeeded

        run = mock_workflow_run()
        output = {"data": {"items": [1, 2, 3]}, "meta": {"count": 3}}

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            await _mark_run_succeeded(run, output)

            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["output"] == output

    async def test_refreshes_run_from_db(self, mock_workflow_run, frozen_time):
        """Calls refresh_from_db on run instance."""
        from seer.services.workflows.execution import _mark_run_succeeded

        run = mock_workflow_run()
        output = {"result": "success"}

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            await _mark_run_succeeded(run, output)

            run.refresh_from_db.assert_called_once()


# =============================================================================
# TestSendHitlNotifications - Fire-and-forget notifications
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSendHitlNotifications:
    """Tests for HITL notification delivery."""

    async def test_skips_when_no_delivery_channels(self, mock_workflow_run, mock_user):
        """No action when delivery_channels is empty."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {"type": "hitl", "delivery_channels": []}

        # Should not raise, should do nothing
        await _send_hitl_notifications(run, mock_user, interrupt_data)

    async def test_skips_when_delivery_channels_missing(self, mock_workflow_run, mock_user):
        """No action when delivery_channels key is missing."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {"type": "hitl"}  # No delivery_channels key

        await _send_hitl_notifications(run, mock_user, interrupt_data)

    async def test_sends_gmail_notification_for_gmail_channel(
        self, mock_workflow_run, mock_user
    ):
        """Gmail notification sent for gmail channel type."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {
            "type": "hitl",
            "node_id": "approval",
            "title": "Test",
            "delivery_channels": [
                {"type": "gmail", "gmail": {"recipient_email": "user@example.com"}}
            ],
        }

        with patch(
            "seer.services.workflows.hitl_email.send_hitl_gmail_notification",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = None  # Success

            await _send_hitl_notifications(run, mock_user, interrupt_data)

            mock_send.assert_called_once()

    async def test_logs_warning_on_gmail_error(self, mock_workflow_run, mock_user):
        """Logs warning when Gmail notification returns error."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {
            "type": "hitl",
            "delivery_channels": [
                {"type": "gmail", "gmail": {"recipient_email": "user@example.com"}}
            ],
        }

        with patch(
            "seer.services.workflows.hitl_email.send_hitl_gmail_notification",
            new_callable=AsyncMock,
        ) as mock_send, patch(
            "seer.services.workflows.execution.logger"
        ) as mock_logger:
            mock_send.return_value = "Gmail API rate limit exceeded"

            await _send_hitl_notifications(run, mock_user, interrupt_data)

            mock_logger.warning.assert_called_once()

    async def test_catches_exception_without_failing(self, mock_workflow_run, mock_user):
        """Exception in notification is caught and logged, not raised."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {
            "type": "hitl",
            "delivery_channels": [
                {"type": "gmail", "gmail": {"recipient_email": "user@example.com"}}
            ],
        }

        with patch(
            "seer.services.workflows.hitl_email.send_hitl_gmail_notification",
            new_callable=AsyncMock,
        ) as mock_send, patch(
            "seer.services.workflows.execution.logger"
        ) as mock_logger:
            mock_send.side_effect = Exception("Connection error")

            # Should not raise
            await _send_hitl_notifications(run, mock_user, interrupt_data)

            mock_logger.exception.assert_called_once()

    async def test_ignores_platform_channel_type(self, mock_workflow_run, mock_user):
        """Platform channel type requires no action (fallback)."""
        from seer.services.workflows.execution import _send_hitl_notifications

        run = mock_workflow_run()
        interrupt_data = {
            "type": "hitl",
            "delivery_channels": [{"type": "platform"}],
        }

        # Should complete without error and without calling any notification service
        await _send_hitl_notifications(run, mock_user, interrupt_data)


# =============================================================================
# TestValidateResumeRequest - Resume authorization & expiry
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValidateResumeRequest:
    """Tests for resume request validation."""

    async def test_raises_403_when_user_not_owner(self, mock_workflow_run, mock_user):
        """HTTPException 403 when user.id != run.user_id."""
        from seer.services.workflows.execution import _validate_resume_request

        run = mock_workflow_run(user_id=999, status="interrupted")  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(mock_user, run)

        assert exc_info.value.status_code == 403
        assert "Not authorized" in exc_info.value.detail

    async def test_raises_400_when_not_interrupted_status(
        self, mock_workflow_run, mock_user
    ):
        """HTTPException 400 when run.status != INTERRUPTED."""
        from seer.services.workflows.execution import _validate_resume_request

        run = mock_workflow_run(user_id=mock_user.id, status="running")

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(mock_user, run)

        assert exc_info.value.status_code == 400
        assert "not in INTERRUPTED state" in exc_info.value.detail

    @pytest.mark.parametrize(
        "status",
        ["queued", "running", "succeeded", "failed", "cancelled"],
    )
    async def test_raises_400_for_all_non_interrupted_statuses(
        self, status, mock_workflow_run, mock_user
    ):
        """Parametrized: All non-INTERRUPTED statuses raise 400."""
        from seer.services.workflows.execution import _validate_resume_request

        run = mock_workflow_run(user_id=mock_user.id, status=status)

        with pytest.raises(HTTPException) as exc_info:
            await _validate_resume_request(mock_user, run)

        assert exc_info.value.status_code == 400

    async def test_raises_408_when_interrupt_expired(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """HTTPException 408 when interrupt_expires_at < now()."""
        from seer.services.workflows.execution import _validate_resume_request

        expired_time = frozen_time - timedelta(hours=1)
        run = mock_workflow_run(
            user_id=mock_user.id, status="interrupted", interrupt_expires_at=expired_time
        )

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            with pytest.raises(HTTPException) as exc_info:
                await _validate_resume_request(mock_user, run)

            assert exc_info.value.status_code == 408
            assert "timed out" in exc_info.value.detail

    async def test_marks_run_failed_on_expiry(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """Expired interrupt updates run to FAILED status."""
        from seer.services.workflows.execution import _validate_resume_request
        from seer.database import WorkflowRunStatus

        expired_time = frozen_time - timedelta(hours=1)
        run = mock_workflow_run(
            user_id=mock_user.id,
            status="interrupted",
            interrupt_expires_at=expired_time,
            pending_interrupt_node_id="node_1",
            pending_interrupt_data={"type": "hitl"},
        )

        with patch("seer.services.workflows.execution.WorkflowRun") as mock_run_model:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            with pytest.raises(HTTPException):
                await _validate_resume_request(mock_user, run)

            # Verify database update
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == WorkflowRunStatus.FAILED
            assert call_kwargs["error"] == "HITL interrupt timed out"
            assert call_kwargs["pending_interrupt_node_id"] is None
            assert call_kwargs["pending_interrupt_data"] is None
            assert call_kwargs["interrupt_expires_at"] is None

    async def test_passes_validation_for_valid_request(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """No exception for valid owner, INTERRUPTED status, non-expired."""
        from seer.services.workflows.execution import _validate_resume_request

        future_expiry = frozen_time + timedelta(hours=1)
        run = mock_workflow_run(
            user_id=mock_user.id, status="interrupted", interrupt_expires_at=future_expiry
        )

        # Should not raise
        await _validate_resume_request(mock_user, run)

    async def test_passes_when_interrupt_expires_at_is_none(
        self, mock_workflow_run, mock_user
    ):
        """None interrupt_expires_at means indefinite (valid)."""
        from seer.services.workflows.execution import _validate_resume_request

        run = mock_workflow_run(
            user_id=mock_user.id, status="interrupted", interrupt_expires_at=None
        )

        # Should not raise
        await _validate_resume_request(mock_user, run)


# =============================================================================
# TestExecuteRun - Core execution logic
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestExecuteRun:
    """Tests for core workflow execution logic."""

    async def test_sets_status_running_at_start(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Run status set to RUNNING at start with started_at timestamp."""
        from seer.services.workflows.execution import _execute_run
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()
        compiled = mock_compiled_workflow(result={"output": "success"})

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            await _execute_run(run, mock_user, inputs={}, config_payload={})

            # First call should set RUNNING status
            first_call = mock_filter.update.call_args_list[0]
            assert first_call.kwargs["status"] == WorkflowRunStatus.RUNNING
            assert first_call.kwargs["started_at"] == frozen_time

    async def test_handles_workflow_compiler_error(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """WorkflowCompilerError sets FAILED status and re-raises."""
        from seer.services.workflows.execution import _execute_run
        from seer.core.errors import WorkflowCompilerError
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.side_effect = WorkflowCompilerError("Tool 'foo' not found")

            with pytest.raises(WorkflowCompilerError):
                await _execute_run(run, mock_user, inputs={}, config_payload={})

            # Verify FAILED status was set
            calls = mock_filter.update.call_args_list
            # Second call (after RUNNING) should be FAILED
            assert len(calls) >= 2
            failed_call = calls[1]
            assert failed_call.kwargs["status"] == WorkflowRunStatus.FAILED
            assert "Tool 'foo' not found" in failed_call.kwargs["error"]

    async def test_handles_run_cost_cap_exceeded(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """RunCostCapExceeded raises HTTPException 402."""
        from seer.services.workflows.execution import _execute_run
        from seer.observability.exceptions import RunCostCapExceeded

        run = mock_workflow_run()
        cost_exc = RunCostCapExceeded(
            run_identifier="run_1",
            accumulated_cost=10.0,
            cost_cap=5.0,
            run_type="workflow",
        )
        compiled = mock_compiled_workflow(side_effect=cost_exc)

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            with pytest.raises(HTTPException) as exc_info:
                await _execute_run(run, mock_user, inputs={}, config_payload={})

            assert exc_info.value.status_code == 402

    async def test_handles_execution_error_with_trace(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """ExecutionError extracts trace_data and stores in node_traces."""
        from seer.services.workflows.execution import _execute_run
        from seer.core.errors import ExecutionError
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()
        trace_data = {"node_id": "node_1", "error": "Failed", "nodes": []}
        exec_error = ExecutionError("Node execution failed", trace_data=trace_data)
        compiled = mock_compiled_workflow(side_effect=exec_error)

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            with pytest.raises(ExecutionError):
                await _execute_run(run, mock_user, inputs={}, config_payload={})

            # Verify node_traces was stored
            calls = mock_filter.update.call_args_list
            failed_call = calls[-1]
            assert failed_call.kwargs["status"] == WorkflowRunStatus.FAILED
            assert failed_call.kwargs["node_traces"] == trace_data

    async def test_handles_generic_exception(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Generic exceptions set FAILED status with error message."""
        from seer.services.workflows.execution import _execute_run
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()
        compiled = mock_compiled_workflow(side_effect=ValueError("Unexpected error"))

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            with pytest.raises(ValueError):
                await _execute_run(run, mock_user, inputs={}, config_payload={})

            # Verify FAILED status
            calls = mock_filter.update.call_args_list
            failed_call = calls[-1]
            assert failed_call.kwargs["status"] == WorkflowRunStatus.FAILED
            assert "Unexpected error" in failed_call.kwargs["error"]

    async def test_sets_status_interrupted_on_hitl(
        self,
        mock_workflow_run,
        mock_user,
        mock_compiled_workflow,
        mock_interrupt_object,
        mock_hitl_interrupt,
        frozen_time,
    ):
        """Run status set to INTERRUPTED when HITL interrupt detected."""
        from seer.services.workflows.execution import _execute_run
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run()
        interrupt = mock_interrupt_object(mock_hitl_interrupt)
        result = {"output": "partial", "__interrupt__": (interrupt,)}
        compiled = mock_compiled_workflow(result=result)

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings, patch(
            "seer.services.workflows.execution._send_hitl_notifications",
            new_callable=AsyncMock,
        ):
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            result = await _execute_run(run, mock_user, inputs={}, config_payload={})

            # Verify INTERRUPTED status was set
            calls = mock_filter.update.call_args_list
            interrupted_call = calls[-1]
            assert interrupted_call.kwargs["status"] == WorkflowRunStatus.INTERRUPTED
            assert interrupted_call.kwargs["pending_interrupt_node_id"] == "approval_node"
            assert "__interrupted__" in result
            assert result["__interrupted__"] is True

    async def test_creates_runtime_context_with_cost_cap(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """WorkflowRuntimeContext created with per_run_cost_cap_usd."""
        from seer.services.workflows.execution import _execute_run

        run = mock_workflow_run()
        compiled = mock_compiled_workflow(result={"output": "success"})

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings, patch(
            "seer.services.workflows.execution.WorkflowRuntimeContext"
        ) as mock_context_cls:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 10.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            mock_context = MagicMock()
            mock_context_cls.return_value = mock_context

            await _execute_run(run, mock_user, inputs={}, config_payload={})

            mock_context_cls.assert_called_once()
            call_kwargs = mock_context_cls.call_args.kwargs
            assert call_kwargs["per_run_cost_cap_usd"] == 10.0
            assert call_kwargs["accumulated_cost_usd"] == 0.0

    async def test_uses_default_cost_cap_when_not_set(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Default 5.0 cost cap used when preference not set."""
        from seer.services.workflows.execution import _execute_run

        run = mock_workflow_run()
        compiled = mock_compiled_workflow(result={"output": "success"})

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ) as mock_cp, patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings, patch(
            "seer.services.workflows.execution.WorkflowRuntimeContext"
        ) as mock_context_cls:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_cp.return_value = None
            mock_compile.return_value = compiled

            # No per_run_cost_cap_usd in preferences
            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            mock_context = MagicMock()
            mock_context_cls.return_value = mock_context

            await _execute_run(run, mock_user, inputs={}, config_payload={})

            call_kwargs = mock_context_cls.call_args.kwargs
            assert call_kwargs["per_run_cost_cap_usd"] == 5.0  # Default


# =============================================================================
# TestExecuteResume - Resume execution
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestExecuteResume:
    """Tests for resume execution logic."""

    async def test_creates_command_with_responses(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Command(resume=responses) passed to ainvoke."""
        from seer.services.workflows.execution import _execute_resume

        run = mock_workflow_run(status="interrupted")
        compiled = mock_compiled_workflow(result={"output": "resumed"})
        responses = {"decision": True, "comment": "Approved"}

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            await _execute_resume(run, mock_user, compiled, responses)

            # Verify ainvoke was called with Command
            compiled.ainvoke.assert_called_once()
            call_args = compiled.ainvoke.call_args
            command = call_args.args[0]
            assert hasattr(command, "resume")
            assert command.resume == responses

    async def test_handles_chained_hitl_interrupt(
        self,
        mock_workflow_run,
        mock_user,
        mock_compiled_workflow,
        mock_interrupt_object,
        frozen_time,
    ):
        """Another HITL interrupt updates run to INTERRUPTED."""
        from seer.services.workflows.execution import _execute_resume
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run(status="interrupted")
        new_interrupt = {
            "type": "hitl",
            "node_id": "second_approval",
            "title": "Second Approval",
            "timeout_seconds": 1800,
        }
        interrupt = mock_interrupt_object(new_interrupt)
        result = {"output": "partial", "__interrupt__": (interrupt,)}
        compiled = mock_compiled_workflow(result=result)

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            result = await _execute_resume(run, mock_user, compiled, {"decision": True})

            # Verify chained HITL
            assert result["__interrupted__"] is True
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == WorkflowRunStatus.INTERRUPTED
            assert call_kwargs["pending_interrupt_node_id"] == "second_approval"

    async def test_marks_succeeded_when_complete(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Run marked SUCCEEDED when no further interrupt."""
        from seer.services.workflows.execution import _execute_resume
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run(status="interrupted")
        compiled = mock_compiled_workflow(result={"output": "final_result"})

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            result = await _execute_resume(run, mock_user, compiled, {"decision": True})

            assert "__interrupted__" not in result
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == WorkflowRunStatus.SUCCEEDED
            assert call_kwargs["output"] == {"output": "final_result"}


# =============================================================================
# TestExecuteSavedWorkflowRun - Taskiq entry point
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestExecuteSavedWorkflowRun:
    """Tests for Taskiq worker entry point."""

    async def test_fetches_run_by_id(self, mock_workflow_run, mock_user, frozen_time):
        """WorkflowRun.get called with run_id."""
        from seer.services.workflows.execution import execute_saved_workflow_run

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.User"
        ) as mock_user_model, patch(
            "seer.services.workflows.execution._execute_run", new_callable=AsyncMock
        ) as mock_exec, patch(
            "seer.services.workflows.execution._mark_run_succeeded", new_callable=AsyncMock
        ):
            mock_run_model.get = AsyncMock(return_value=run)

            mock_exec.return_value = {"output": "success"}

            await execute_saved_workflow_run(run_id=123, user_id=1)

            mock_run_model.get.assert_called_once_with(id=123)

    async def test_fetches_related_workflow_and_user(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """fetch_related called with workflow, user."""
        from seer.services.workflows.execution import execute_saved_workflow_run

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._execute_run", new_callable=AsyncMock
        ) as mock_exec, patch(
            "seer.services.workflows.execution._mark_run_succeeded", new_callable=AsyncMock
        ):
            mock_run_model.get = AsyncMock(return_value=run)
            mock_exec.return_value = {"output": "success"}

            await execute_saved_workflow_run(run_id=123, user_id=1)

            run.fetch_related.assert_called_once_with("workflow", "user")

    async def test_skips_success_marking_on_interrupt(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """Does not mark succeeded when __interrupted__ is True."""
        from seer.services.workflows.execution import execute_saved_workflow_run

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._execute_run", new_callable=AsyncMock
        ) as mock_exec, patch(
            "seer.services.workflows.execution._mark_run_succeeded", new_callable=AsyncMock
        ) as mock_mark:
            mock_run_model.get = AsyncMock(return_value=run)
            mock_exec.return_value = {"__interrupted__": True, "output": "partial"}

            await execute_saved_workflow_run(run_id=123, user_id=1)

            mock_mark.assert_not_called()

    async def test_marks_succeeded_on_completion(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """_mark_run_succeeded called on successful execution."""
        from seer.services.workflows.execution import execute_saved_workflow_run

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._execute_run", new_callable=AsyncMock
        ) as mock_exec, patch(
            "seer.services.workflows.execution._mark_run_succeeded", new_callable=AsyncMock
        ) as mock_mark:
            mock_run_model.get = AsyncMock(return_value=run)
            output = {"output": "success", "data": [1, 2, 3]}
            mock_exec.return_value = output

            await execute_saved_workflow_run(run_id=123, user_id=1)

            mock_mark.assert_called_once_with(run, output)

    async def test_logs_and_reraises_http_exception(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """HTTPException is logged and re-raised."""
        from seer.services.workflows.execution import execute_saved_workflow_run

        run = mock_workflow_run()

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._execute_run", new_callable=AsyncMock
        ) as mock_exec, patch(
            "seer.services.workflows.execution.logger"
        ) as mock_logger:
            mock_run_model.get = AsyncMock(return_value=run)
            mock_exec.side_effect = HTTPException(status_code=402, detail="Cost cap")

            with pytest.raises(HTTPException) as exc_info:
                await execute_saved_workflow_run(run_id=123, user_id=1)

            assert exc_info.value.status_code == 402
            mock_logger.exception.assert_called()


# =============================================================================
# TestResumeWorkflowRun - Resume API logic
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestResumeWorkflowRun:
    """Tests for resume workflow API endpoint logic."""

    async def test_raises_400_for_invalid_run_id_format(self, mock_user):
        """HTTPException 400 for malformed run_id."""
        from seer.services.workflows.execution import resume_workflow_run

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse:
            mock_parse.side_effect = ValueError("Invalid format")

            with pytest.raises(HTTPException) as exc_info:
                await resume_workflow_run(mock_user, "invalid_id", {})

            assert exc_info.value.status_code == 400
            assert "Invalid run_id format" in exc_info.value.detail

    async def test_raises_404_when_run_not_found(self, mock_user):
        """HTTPException 404 when WorkflowRun.get_or_none returns None."""
        from seer.services.workflows.execution import resume_workflow_run

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 123
            mock_run_model.get_or_none = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await resume_workflow_run(mock_user, "run_123", {})

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail

    async def test_calls_validate_resume_request(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """_validate_resume_request called with user and run."""
        from seer.services.workflows.execution import resume_workflow_run

        run = mock_workflow_run(status="interrupted")

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._validate_resume_request",
            new_callable=AsyncMock,
        ) as mock_validate, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._execute_resume",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_exec.return_value = {"output": "success"}

            await resume_workflow_run(mock_user, "run_1", {"decision": True})

            mock_validate.assert_called_once_with(mock_user, run)

    async def test_raises_500_on_compilation_error(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """HTTPException 500 when compilation fails during resume."""
        from seer.services.workflows.execution import resume_workflow_run
        from seer.core.errors import WorkflowCompilerError

        run = mock_workflow_run(status="interrupted")

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._validate_resume_request",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.side_effect = WorkflowCompilerError("Compilation failed")

            with pytest.raises(HTTPException) as exc_info:
                await resume_workflow_run(mock_user, "run_1", {})

            assert exc_info.value.status_code == 500
            assert "Compilation failed" in exc_info.value.detail

    async def test_raises_402_on_cost_cap_exceeded(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """HTTPException 402 when cost cap exceeded during resume."""
        from seer.services.workflows.execution import resume_workflow_run
        from seer.observability.exceptions import RunCostCapExceeded

        run = mock_workflow_run(status="interrupted")

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._validate_resume_request",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution._execute_resume",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.return_value = mock_compiled_workflow()
            mock_exec.side_effect = RunCostCapExceeded(
                run_identifier="run_1",
                accumulated_cost=15.0,
                cost_cap=10.0,
                run_type="workflow",
            )

            with pytest.raises(HTTPException) as exc_info:
                await resume_workflow_run(mock_user, "run_1", {})

            assert exc_info.value.status_code == 402

    async def test_returns_result_on_success(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Returns execution result dict on success."""
        from seer.services.workflows.execution import resume_workflow_run

        run = mock_workflow_run(status="interrupted")
        expected_result = {"output": "final_result", "data": {"items": [1, 2, 3]}}

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution._validate_resume_request",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution._execute_resume",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.return_value = mock_compiled_workflow()
            mock_exec.return_value = expected_result

            result = await resume_workflow_run(mock_user, "run_1", {"decision": True})

            assert result == expected_result


# =============================================================================
# TestGetWorkflowRunInterrupt - Interrupt retrieval
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetWorkflowRunInterrupt:
    """Tests for interrupt data retrieval."""

    async def test_raises_400_for_invalid_run_id_format(self, mock_user):
        """HTTPException 400 for malformed run_id."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse:
            mock_parse.side_effect = ValueError("Invalid format")

            with pytest.raises(HTTPException) as exc_info:
                await get_workflow_run_interrupt(mock_user, "bad_id")

            assert exc_info.value.status_code == 400

    async def test_raises_404_when_run_not_found(self, mock_user):
        """HTTPException 404 when run doesn't exist."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 999
            mock_run_model.get_or_none = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await get_workflow_run_interrupt(mock_user, "run_999")

            assert exc_info.value.status_code == 404

    async def test_raises_403_when_not_owner(self, mock_workflow_run, mock_user):
        """HTTPException 403 when user.id != run.user_id."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        run = mock_workflow_run(user_id=999)  # Different user

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc_info:
                await get_workflow_run_interrupt(mock_user, "run_1")

            assert exc_info.value.status_code == 403

    async def test_returns_none_when_not_interrupted(self, mock_workflow_run, mock_user):
        """Returns None when run.status != INTERRUPTED."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        run = mock_workflow_run(status="running")

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)

            result = await get_workflow_run_interrupt(mock_user, "run_1")

            assert result is None

    async def test_returns_interrupt_data_structure(
        self, mock_workflow_run, mock_user, mock_hitl_interrupt, frozen_time
    ):
        """Returns dict with run_id, status, node_id, interrupt_data, etc."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        future_expiry = frozen_time + timedelta(hours=1)
        run = mock_workflow_run(
            run_id="run_123",
            status="interrupted",
            pending_interrupt_node_id="approval_node",
            pending_interrupt_data=mock_hitl_interrupt,
            interrupt_expires_at=future_expiry,
        )

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 123
            mock_run_model.get_or_none = AsyncMock(return_value=run)

            result = await get_workflow_run_interrupt(mock_user, "run_123")

            assert result is not None
            assert result["run_id"] == "run_123"
            assert result["status"] == "interrupted"
            assert result["node_id"] == "approval_node"
            assert result["interrupt_data"] == mock_hitl_interrupt
            assert result["expires_at"] == future_expiry.isoformat()
            assert result["is_expired"] is False

    async def test_includes_is_expired_flag_when_expired(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """is_expired=True when interrupt_expires_at < now()."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        expired_time = frozen_time - timedelta(hours=1)
        run = mock_workflow_run(
            status="interrupted",
            pending_interrupt_node_id="node_1",
            pending_interrupt_data={"type": "hitl"},
            interrupt_expires_at=expired_time,
        )

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)

            result = await get_workflow_run_interrupt(mock_user, "run_1")

            assert result is not None
            assert result["is_expired"] is True

    async def test_handles_none_expires_at(self, mock_workflow_run, mock_user):
        """expires_at=None means indefinite - is_expired is falsy (None from short-circuit)."""
        from seer.services.workflows.execution import get_workflow_run_interrupt

        run = mock_workflow_run(
            status="interrupted",
            pending_interrupt_node_id="node_1",
            pending_interrupt_data={"type": "hitl"},
            interrupt_expires_at=None,
        )

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)

            result = await get_workflow_run_interrupt(mock_user, "run_1")

            assert result is not None
            assert result["expires_at"] is None
            # When interrupt_expires_at is None, short-circuit evaluation returns None
            # This is semantically correct: no expiry time = not applicable (falsy)
            assert not result["is_expired"]  # Falsy (None or False)


# =============================================================================
# TestStateTransitions - End-to-end state changes
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestStateTransitions:
    """End-to-end state transition tests."""

    async def test_queued_to_running_to_succeeded(
        self, mock_workflow_run, mock_user, mock_compiled_workflow, frozen_time
    ):
        """Happy path: QUEUED -> RUNNING -> SUCCEEDED."""
        from seer.services.workflows.execution import execute_saved_workflow_run
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run(status="queued")
        compiled = mock_compiled_workflow(result={"output": "success"})

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings:
            mock_run_model.get = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            await execute_saved_workflow_run(run_id=1, user_id=1)

            # Verify state transitions
            calls = mock_filter.update.call_args_list
            assert len(calls) >= 2

            # First: RUNNING
            assert calls[0].kwargs["status"] == WorkflowRunStatus.RUNNING

            # Last: SUCCEEDED
            assert calls[-1].kwargs["status"] == WorkflowRunStatus.SUCCEEDED

    async def test_queued_to_running_to_failed_compilation(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """Compilation error: QUEUED -> RUNNING -> FAILED."""
        from seer.services.workflows.execution import execute_saved_workflow_run
        from seer.core.errors import WorkflowCompilerError
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run(status="queued")

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile:
            mock_run_model.get = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.side_effect = WorkflowCompilerError("Invalid spec")

            with pytest.raises(WorkflowCompilerError):
                await execute_saved_workflow_run(run_id=1, user_id=1)

            calls = mock_filter.update.call_args_list
            assert calls[0].kwargs["status"] == WorkflowRunStatus.RUNNING
            assert calls[1].kwargs["status"] == WorkflowRunStatus.FAILED

    async def test_queued_to_running_to_interrupted(
        self,
        mock_workflow_run,
        mock_user,
        mock_compiled_workflow,
        mock_interrupt_object,
        mock_hitl_interrupt,
        frozen_time,
    ):
        """HITL pause: QUEUED -> RUNNING -> INTERRUPTED."""
        from seer.services.workflows.execution import execute_saved_workflow_run
        from seer.database import WorkflowRunStatus

        run = mock_workflow_run(status="queued")
        interrupt = mock_interrupt_object(mock_hitl_interrupt)
        result = {"output": "partial", "__interrupt__": (interrupt,)}
        compiled = mock_compiled_workflow(result=result)

        with patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model, patch(
            "seer.services.workflows.execution.get_checkpointer",
            new_callable=AsyncMock,
        ), patch(
            "seer.services.workflows.execution._compile_workflow",
            new_callable=AsyncMock,
        ) as mock_compile, patch(
            "seer.services.workflows.execution.UserSettings"
        ) as mock_settings, patch(
            "seer.services.workflows.execution._send_hitl_notifications",
            new_callable=AsyncMock,
        ):
            mock_run_model.get = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_compile.return_value = compiled

            mock_settings_instance = MagicMock()
            mock_settings_instance.preferences = {"per_run_cost_cap_usd": 5.0}
            mock_settings.get_or_create = AsyncMock(
                return_value=(mock_settings_instance, False)
            )

            await execute_saved_workflow_run(run_id=1, user_id=1)

            calls = mock_filter.update.call_args_list
            assert calls[0].kwargs["status"] == WorkflowRunStatus.RUNNING
            assert calls[1].kwargs["status"] == WorkflowRunStatus.INTERRUPTED

    async def test_interrupted_to_failed_on_timeout(
        self, mock_workflow_run, mock_user, frozen_time
    ):
        """Timeout expired: INTERRUPTED -> FAILED (via validation)."""
        from seer.services.workflows.execution import resume_workflow_run
        from seer.database import WorkflowRunStatus

        expired_time = frozen_time - timedelta(hours=1)
        run = mock_workflow_run(
            status="interrupted",
            interrupt_expires_at=expired_time,
        )

        with patch(
            "seer.database.workflow_models.parse_run_public_id"
        ) as mock_parse, patch(
            "seer.services.workflows.execution.WorkflowRun"
        ) as mock_run_model:
            mock_parse.return_value = 1
            mock_run_model.get_or_none = AsyncMock(return_value=run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter

            with pytest.raises(HTTPException) as exc_info:
                await resume_workflow_run(mock_user, "run_1", {})

            assert exc_info.value.status_code == 408

            # Verify FAILED status was set
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == WorkflowRunStatus.FAILED
            assert "timed out" in call_kwargs["error"]
