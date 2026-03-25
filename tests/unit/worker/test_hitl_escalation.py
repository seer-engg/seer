"""Unit tests for HITL escalation task."""

import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.worker.tasks.hitl_escalation import (
    CALL_AFTER_SECONDS,
    REMINDER_AFTER_SECONDS,
    _extract_slack_config,
    _extract_twilio_config,
    _get_escalation_state,
    _set_escalation_state,
)


def _make_run(
    run_id: int = 1,
    interrupt_started_minutes_ago: float = 20,
    timeout_seconds: int = CALL_AFTER_SECONDS,
    workflow_name: str = "PULIS",
    node_traces: dict | None = None,
    spec: dict | None = None,
):
    """Create a mock WorkflowRun in INTERRUPTED state."""
    now = dt.datetime.now(dt.timezone.utc)
    interrupt_started_at = now - dt.timedelta(minutes=interrupt_started_minutes_ago)
    expires_at = interrupt_started_at + dt.timedelta(seconds=timeout_seconds)

    run = MagicMock()
    run.id = run_id
    run.interrupt_expires_at = expires_at
    run.pending_interrupt_data = {"timeout_seconds": timeout_seconds}
    run.node_traces = node_traces or {}
    run.spec = spec or {
        "nodes": [
            {
                "type": "tool",
                "tool": "slack_send_direct_message",
                "inputs": {"workspace_id": "T123", "user_id": "U456", "text": "check-in"},
            }
        ],
        "variables": {"escalation_phone_number": "+14155559999"},
    }
    run.workflow = MagicMock()
    run.workflow.name = workflow_name
    run.user = MagicMock()
    return run


@pytest.mark.unit
class TestEscalationHelpers:
    """Tests for escalation helper functions."""

    def test_get_escalation_state_empty(self):
        run = _make_run(node_traces={})
        assert _get_escalation_state(run) == {}

    def test_get_escalation_state_with_data(self):
        run = _make_run(node_traces={"_pulis_escalation": {"reminder_sent": True}})
        assert _get_escalation_state(run) == {"reminder_sent": True}

    def test_set_escalation_state(self):
        run = _make_run(node_traces={})
        _set_escalation_state(run, "reminder_sent", True)
        assert run.node_traces["_pulis_escalation"]["reminder_sent"] is True

    def test_set_escalation_state_preserves_existing(self):
        run = _make_run(node_traces={"_pulis_escalation": {"reminder_sent": True}})
        _set_escalation_state(run, "call_sent", True)
        assert run.node_traces["_pulis_escalation"]["reminder_sent"] is True
        assert run.node_traces["_pulis_escalation"]["call_sent"] is True

    def test_extract_slack_config(self):
        run = _make_run()
        config = _extract_slack_config(run)
        assert config["workspace_id"] == "T123"
        assert config["user_id"] == "U456"

    def test_extract_slack_config_missing(self):
        run = _make_run(spec={"nodes": []})
        with pytest.raises(ValueError, match="Could not find Slack config"):
            _extract_slack_config(run)

    def test_extract_twilio_config(self):
        run = _make_run()
        config = _extract_twilio_config(run)
        assert config["to_number"] == "+14155559999"

    def test_extract_twilio_config_missing(self):
        run = _make_run(spec={"nodes": [], "variables": {}})
        with pytest.raises(ValueError, match="Missing 'escalation_phone_number'"):
            _extract_twilio_config(run)


@pytest.mark.unit
class TestCheckHitlEscalations:
    """Tests for the main escalation task."""

    @pytest.mark.asyncio
    async def test_no_interrupted_runs(self):
        from seer.worker.tasks.hitl_escalation import check_hitl_escalations

        with patch("seer.worker.tasks.hitl_escalation.WorkflowRun") as mock_model:
            mock_qs = MagicMock()
            mock_qs.select_related.return_value.all = AsyncMock(return_value=[])
            mock_model.filter.return_value = mock_qs

            result = await check_hitl_escalations()
            assert result["reminders_sent"] == 0
            assert result["calls_made"] == 0

    @pytest.mark.asyncio
    async def test_sends_reminder_after_15_min(self):
        """Run interrupted 20min ago should get a Slack reminder."""
        run = _make_run(interrupt_started_minutes_ago=20)

        with (
            patch("seer.worker.tasks.hitl_escalation.WorkflowRun") as mock_model,
            patch("seer.worker.tasks.hitl_escalation._send_slack_reminder", new_callable=AsyncMock) as mock_reminder,
            patch("seer.worker.tasks.hitl_escalation._send_twilio_call", new_callable=AsyncMock) as mock_call,
        ):
            mock_qs = MagicMock()
            mock_qs.select_related.return_value.all = AsyncMock(return_value=[run])
            mock_model.filter.return_value = mock_qs
            mock_model.filter.return_value.update = AsyncMock()

            from seer.worker.tasks.hitl_escalation import check_hitl_escalations
            result = await check_hitl_escalations()

            mock_reminder.assert_called_once_with(run)
            mock_call.assert_not_called()
            assert result["reminders_sent"] == 1

    @pytest.mark.asyncio
    async def test_sends_call_after_30_min(self):
        """Run interrupted 35min ago should get a phone call."""
        run = _make_run(interrupt_started_minutes_ago=35)

        with (
            patch("seer.worker.tasks.hitl_escalation.WorkflowRun") as mock_model,
            patch("seer.worker.tasks.hitl_escalation._send_slack_reminder", new_callable=AsyncMock) as mock_reminder,
            patch("seer.worker.tasks.hitl_escalation._send_twilio_call", new_callable=AsyncMock) as mock_call,
        ):
            mock_qs = MagicMock()
            mock_qs.select_related.return_value.all = AsyncMock(return_value=[run])
            mock_model.filter.return_value = mock_qs
            mock_model.filter.return_value.update = AsyncMock()

            from seer.worker.tasks.hitl_escalation import check_hitl_escalations
            result = await check_hitl_escalations()

            # Should send call (30min threshold) AND reminder (15min threshold)
            # But since call is checked first and both thresholds are met,
            # the call takes priority due to the elif
            mock_call.assert_called_once_with(run)
            assert result["calls_made"] == 1

    @pytest.mark.asyncio
    async def test_no_duplicate_reminder(self):
        """Run that already got a reminder should not get another."""
        run = _make_run(
            interrupt_started_minutes_ago=20,
            node_traces={"_pulis_escalation": {"reminder_sent": True}},
        )

        with (
            patch("seer.worker.tasks.hitl_escalation.WorkflowRun") as mock_model,
            patch("seer.worker.tasks.hitl_escalation._send_slack_reminder", new_callable=AsyncMock) as mock_reminder,
            patch("seer.worker.tasks.hitl_escalation._send_twilio_call", new_callable=AsyncMock) as mock_call,
        ):
            mock_qs = MagicMock()
            mock_qs.select_related.return_value.all = AsyncMock(return_value=[run])
            mock_model.filter.return_value = mock_qs

            from seer.worker.tasks.hitl_escalation import check_hitl_escalations
            result = await check_hitl_escalations()

            mock_reminder.assert_not_called()
            mock_call.assert_not_called()
            assert result["reminders_sent"] == 0

    @pytest.mark.asyncio
    async def test_too_early_for_escalation(self):
        """Run interrupted 5min ago should not get escalated."""
        run = _make_run(interrupt_started_minutes_ago=5)

        with (
            patch("seer.worker.tasks.hitl_escalation.WorkflowRun") as mock_model,
            patch("seer.worker.tasks.hitl_escalation._send_slack_reminder", new_callable=AsyncMock) as mock_reminder,
            patch("seer.worker.tasks.hitl_escalation._send_twilio_call", new_callable=AsyncMock) as mock_call,
        ):
            mock_qs = MagicMock()
            mock_qs.select_related.return_value.all = AsyncMock(return_value=[run])
            mock_model.filter.return_value = mock_qs

            from seer.worker.tasks.hitl_escalation import check_hitl_escalations
            result = await check_hitl_escalations()

            mock_reminder.assert_not_called()
            mock_call.assert_not_called()
