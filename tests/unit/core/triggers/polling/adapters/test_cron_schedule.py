"""Unit tests for CronScheduleAdapter, including catchup storm prevention."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from seer.core.triggers.polling.adapters.cron_schedule import (
    CronScheduleAdapter,
    MAX_CATCHUP_WINDOW,
)
from seer.core.triggers.polling.adapters.base import PollContext
from seer.database import TriggerSubscription, User, OAuthConnection


pytestmark = pytest.mark.unit


@pytest.fixture
def adapter():
    return CronScheduleAdapter()


@pytest.fixture
def mock_ctx():
    subscription = Mock(spec=TriggerSubscription)
    subscription.id = 120
    subscription.provider_config = {
        "cron_expression": "*/15 * * * *",
        "timezone": "America/Chicago",
    }
    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)
    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token=None,
    )


class TestCatchupStormPrevention:
    """Tests for the MAX_CATCHUP_WINDOW guard that prevents backfill storms."""

    @pytest.mark.asyncio
    async def test_stale_cursor_skips_to_present(self, adapter, mock_ctx):
        """When cursor is far in the past, poll() should fast-forward to present
        instead of emitting a catchup event."""
        stale_time = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        cursor = {
            "last_execution_utc": stale_time,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        # Should emit zero events (skip, not backfill)
        assert len(result.events) == 0
        assert result.has_more is False

        # Cursor should be fast-forwarded to approximately now
        new_last_exec = datetime.fromisoformat(result.cursor["last_execution_utc"])
        assert (datetime.now(timezone.utc) - new_last_exec).total_seconds() < 10

        # rate_limit_hint should point to a future time, not 1 second
        assert result.rate_limit_hint is not None
        assert result.rate_limit_hint > 1

    @pytest.mark.asyncio
    async def test_recent_cursor_fires_normally(self, adapter, mock_ctx):
        """When cursor is recent (within catchup window), poll() should fire normally."""
        # Set cursor to 20 minutes ago — within the 1-hour window, next cron at +15min is past
        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        cursor = {
            "last_execution_utc": recent_time,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        # Should emit exactly 1 event (normal fire)
        assert len(result.events) == 1
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_cursor_just_outside_window_skips(self, adapter, mock_ctx):
        """Cursor just outside MAX_CATCHUP_WINDOW should trigger skip."""
        just_outside = (
            datetime.now(timezone.utc) - MAX_CATCHUP_WINDOW - timedelta(minutes=30)
        ).isoformat()
        cursor = {
            "last_execution_utc": just_outside,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        assert len(result.events) == 0
        # Cursor should be updated to near-present
        new_last_exec = datetime.fromisoformat(result.cursor["last_execution_utc"])
        assert (datetime.now(timezone.utc) - new_last_exec).total_seconds() < 10

    @pytest.mark.asyncio
    async def test_cursor_just_inside_window_fires(self, adapter, mock_ctx):
        """Cursor just inside MAX_CATCHUP_WINDOW should fire normally."""
        just_inside = (
            datetime.now(timezone.utc) - timedelta(minutes=30)
        ).isoformat()
        cursor = {
            "last_execution_utc": just_inside,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        # Should fire — next cron after 30 min ago is ~15 min ago, within window
        assert len(result.events) == 1

    @pytest.mark.asyncio
    async def test_daily_cron_stale_cursor_skips(self, adapter, mock_ctx):
        """Daily cron with a week-old cursor should also skip to present."""
        stale_time = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cursor = {
            "last_execution_utc": stale_time,
            "cron_expression": "0 7 * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        assert len(result.events) == 0
        new_last_exec = datetime.fromisoformat(result.cursor["last_execution_utc"])
        assert (datetime.now(timezone.utc) - new_last_exec).total_seconds() < 10


class TestBootstrapCursor:
    """Tests for bootstrap_cursor behavior."""

    @pytest.mark.asyncio
    async def test_bootstrap_starts_from_now(self, adapter, mock_ctx):
        """bootstrap_cursor should set last_execution_utc to approximately now."""
        cursor = await adapter.bootstrap_cursor(mock_ctx)

        last_exec = datetime.fromisoformat(cursor["last_execution_utc"])
        assert (datetime.now(timezone.utc) - last_exec).total_seconds() < 5
        assert cursor["cron_expression"] == "*/15 * * * *"
        assert cursor["timezone"] == "America/Chicago"


class TestNormalPollBehavior:
    """Tests for normal (non-catchup) polling."""

    @pytest.mark.asyncio
    async def test_not_yet_due_returns_empty(self, adapter, mock_ctx):
        """When next cron time is in the future, return empty events."""
        # Set cursor to now — next cron is 15 min from now
        now = datetime.now(timezone.utc).isoformat()
        cursor = {
            "last_execution_utc": now,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        assert len(result.events) == 0
        assert result.has_more is False
        assert result.rate_limit_hint > 0

    @pytest.mark.asyncio
    async def test_event_payload_contains_scheduled_time(self, adapter, mock_ctx):
        """Fired event should contain scheduled_time and actual_time."""
        past = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        cursor = {
            "last_execution_utc": past,
            "cron_expression": "*/15 * * * *",
            "timezone": "America/Chicago",
        }

        result = await adapter.poll(mock_ctx, cursor)

        assert len(result.events) == 1
        event = result.events[0]
        assert "scheduled_time" in event.payload
        assert "actual_time" in event.payload
        assert event.occurred_at is not None
