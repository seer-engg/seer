"""
Integration tests for the full trigger polling pipeline.

Tests _process_subscription end-to-end with a test adapter — no engine
internals are mocked. Only external registries (trigger_registry, adapter_registry)
are patched to inject test adapters.

Covers: adapter errors, event deduplication, rate limit scheduling,
status transitions, and the dispatch pipeline.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.triggers.polling.adapters.base import (
    PollAdapterError,
    PollContext,
    PollResult,
    PolledEvent,
)
from seer.core.triggers.polling.engine import TriggerPollEngine
from seer.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
)


def _utcnow():
    return datetime.now(timezone.utc)


# =============================================================================
# Test Adapter — deterministic, no external calls
# =============================================================================


class StubPollAdapter:
    """A test adapter that returns pre-configured events."""

    trigger_key = "test.poll_trigger"

    def __init__(self, events=None, cursor=None, has_more=False, rate_limit_hint=None):
        self.events = events or []
        self.cursor = cursor or {"page": 1}
        self.has_more = has_more
        self.rate_limit_hint = rate_limit_hint
        self.poll_count = 0
        self.error_on_poll = None  # Set to raise on poll()

    async def bootstrap_cursor(self, ctx: PollContext):
        return {"page": 0}

    async def poll(self, ctx: PollContext, cursor):
        self.poll_count += 1
        if self.error_on_poll:
            raise self.error_on_poll
        return PollResult(
            events=self.events,
            cursor=self.cursor,
            has_more=self.has_more,
            rate_limit_hint=self.rate_limit_hint,
        )


def _make_trigger_definition(requires_connection=False, provider="test"):
    """Create a mock TriggerDefinition for the test trigger key."""
    defn = MagicMock()
    defn.provider = provider
    defn.meta = MagicMock()
    defn.meta.requires_connection = requires_connection
    return defn


async def _create_subscription(workflow, **overrides):
    """Create a poll subscription due for processing."""
    defaults = dict(
        user=workflow.user,
        workflow=workflow,
        trigger_id="t_test",
        trigger_key="test.poll_trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=_utcnow() - timedelta(minutes=5),
        poll_interval_seconds=300,
        poll_status="ok",
    )
    defaults.update(overrides)
    return await TriggerSubscription.create(**defaults)


# =============================================================================
# Full Pipeline: Adapter → Events → Dispatch → Mark Success
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPollingPipeline:
    """End-to-end _process_subscription with test adapter and real DB."""

    async def test_successful_poll_creates_event_and_dispatches(self, db_engine, test_workflow):
        """Happy path: adapter returns event → persisted → dispatched → status ok."""
        sub = await _create_subscription(test_workflow)

        adapter = StubPollAdapter(
            events=[PolledEvent(payload={"msg": "hello"}, provider_event_id="evt_001")],
            cursor={"page": 2},
        )
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        # Event persisted in DB
        events = await TriggerEvent.filter(subscription_id=sub.id).all()
        assert len(events) == 1
        assert events[0].provider_event_id == "evt_001"
        assert events[0].status == TriggerEventStatus.RECEIVED

        # Dispatcher called
        dispatcher.assert_called_once()

        # Subscription marked success with new cursor
        await sub.refresh_from_db()
        assert sub.poll_status == "ok"
        assert sub.poll_cursor_json == {"page": 2}
        assert sub.poll_lock_owner is None
        assert sub.poll_error_json is None

    async def test_empty_poll_still_marks_success(self, db_engine, test_workflow):
        """Adapter returns no events — subscription still marked ok with updated cursor."""
        sub = await _create_subscription(test_workflow, trigger_id="t_empty")

        adapter = StubPollAdapter(events=[], cursor={"page": 1})
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        dispatcher.assert_not_called()

        await sub.refresh_from_db()
        assert sub.poll_status == "ok"
        assert sub.poll_cursor_json == {"page": 1}

    async def test_multiple_events_all_persisted_and_dispatched(self, db_engine, test_workflow):
        """Multiple events from one poll are all persisted and dispatched individually."""
        sub = await _create_subscription(test_workflow, trigger_id="t_multi")

        adapter = StubPollAdapter(
            events=[
                PolledEvent(payload={"i": 1}, provider_event_id="evt_a"),
                PolledEvent(payload={"i": 2}, provider_event_id="evt_b"),
                PolledEvent(payload={"i": 3}, provider_event_id="evt_c"),
            ],
            cursor={"page": 3},
        )
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        events = await TriggerEvent.filter(subscription_id=sub.id).all()
        assert len(events) == 3
        assert dispatcher.call_count == 3


# =============================================================================
# Adapter Error Handling
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestAdapterErrors:
    """PollAdapterError handling: permanent, backoff, and unhandled errors."""

    async def test_permanent_error_disables_subscription(self, db_engine, test_workflow):
        """PollAdapterError(permanent=True) → poll_status='disabled'."""
        sub = await _create_subscription(test_workflow, trigger_id="t_perm")

        adapter = StubPollAdapter()
        adapter.error_on_poll = PollAdapterError(
            "Auth revoked", permanent=True, detail={"code": 401}
        )
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        assert sub.poll_status == "disabled"
        assert sub.poll_error_json["reason"] == "adapter_permanent_error"
        assert sub.poll_lock_owner is None

    async def test_backoff_error_sets_backoff_status(self, db_engine, test_workflow):
        """PollAdapterError(backoff_seconds=120) → poll_status='backoff', next_poll delayed."""
        sub = await _create_subscription(test_workflow, trigger_id="t_backoff")

        adapter = StubPollAdapter()
        adapter.error_on_poll = PollAdapterError(
            "Rate limited", backoff_seconds=120, detail={"retry_after": 120}
        )
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        before = _utcnow()
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        assert sub.poll_status == "backoff"
        assert sub.poll_backoff_seconds == 120
        assert sub.poll_error_json["reason"] == "adapter_error"
        # Next poll should be ~120s from now
        assert sub.next_poll_at >= before + timedelta(seconds=110)
        assert sub.poll_lock_owner is None

    async def test_unhandled_exception_in_process_marks_error(self, db_engine, test_workflow):
        """Unhandled exception in _process_subscription → poll_status='error' via tick()."""
        sub = await _create_subscription(test_workflow, trigger_id="t_unhandled")

        # Adapter that raises a non-PollAdapterError exception
        adapter = StubPollAdapter()
        adapter.error_on_poll = RuntimeError("Connection reset")
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        # Call through tick() — which catches all exceptions from _process_subscription
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine.tick()

        await sub.refresh_from_db()
        assert sub.poll_status == "error"
        assert sub.poll_error_json["reason"] == "Unhandled poller exception"
        assert sub.poll_lock_owner is None

    async def test_missing_adapter_disables_subscription(self, db_engine, test_workflow):
        """Unregistered trigger key → poll_status='disabled', reason='missing_adapter'."""
        sub = await _create_subscription(
            test_workflow, trigger_id="t_no_adapter", trigger_key="nonexistent.trigger"
        )

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        # adapter_registry.get returns None for unknown key
        with patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=None):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        assert sub.poll_status == "disabled"
        assert sub.poll_error_json["reason"] == "missing_adapter"


# =============================================================================
# Event Deduplication
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventDeduplication:
    """Duplicate events should be persisted once, dispatched once."""

    async def test_duplicate_provider_event_id_not_dispatched_twice(self, db_engine, test_workflow):
        """Same provider_event_id polled twice → only 1 event created, 1 dispatch."""
        # provider_connection_id must be non-null for unique_together dedup to work.
        # SQL standard: NULL != NULL in unique constraints, so NULLs bypass dedup.
        # In production, polling subscriptions always have a connection.
        sub = await _create_subscription(
            test_workflow, trigger_id="t_dedup", provider_connection_id=999
        )

        events = [PolledEvent(payload={"msg": "hello"}, provider_event_id="dedup_001")]

        adapter = StubPollAdapter(events=events, cursor={"page": 1})
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        # Mock get_oauth_token — we're testing dedup, not OAuth resolution.
        # provider_connection_id=999 has no real OAuthConnection record.
        mock_token = AsyncMock(return_value=(MagicMock(id=999), "fake_token"))
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
            patch("seer.core.triggers.polling.engine.get_oauth_token", mock_token),
        ):
            # First poll — event created and dispatched
            await engine._process_subscription(sub)

            # Reset cursor to simulate re-poll
            sub.poll_cursor_json = None
            await sub.save(update_fields=["poll_cursor_json"])

            # Second poll — same event, should NOT dispatch again
            await engine._process_subscription(sub)

        # Only 1 event in DB
        all_events = await TriggerEvent.filter(subscription_id=sub.id).all()
        assert len(all_events) == 1

        # Dispatcher called only once (first poll)
        assert dispatcher.call_count == 1

    async def test_event_hash_dedup_when_no_provider_event_id(self, db_engine, test_workflow):
        """Events without provider_event_id use hash-based dedup."""
        # Non-null provider_connection_id needed for unique_together constraint to fire
        sub = await _create_subscription(
            test_workflow, trigger_id="t_hash_dedup", provider_connection_id=999
        )

        # No provider_event_id — engine will compute hash
        events = [PolledEvent(payload={"scheduled": True}, provider_event_id=None)]

        adapter = StubPollAdapter(events=events, cursor={"page": 1})
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        mock_token = AsyncMock(return_value=(MagicMock(id=999), "fake_token"))
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
            patch("seer.core.triggers.polling.engine.get_oauth_token", mock_token),
        ):
            await engine._process_subscription(sub)

        all_events = await TriggerEvent.filter(subscription_id=sub.id).all()
        assert len(all_events) == 1
        assert all_events[0].event_hash is not None
        assert all_events[0].provider_event_id is None
        assert dispatcher.call_count == 1


# =============================================================================
# Scheduling: has_more, rate_limit_hint, normal interval
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPollScheduling:
    """_mark_success scheduling logic: has_more, rate_limit_hint, jitter."""

    async def test_has_more_schedules_1_second_poll(self, db_engine, test_workflow):
        """has_more=True → next_poll_at ≈ now + 1s (greedy polling)."""
        sub = await _create_subscription(test_workflow, trigger_id="t_hasmore")

        adapter = StubPollAdapter(events=[], cursor={"page": 2}, has_more=True)
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        before = _utcnow()
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        # Next poll should be ~1 second from now, not 300 seconds
        assert sub.next_poll_at < before + timedelta(seconds=5)

    async def test_rate_limit_hint_respected(self, db_engine, test_workflow):
        """rate_limit_hint=60 → next_poll_at ≈ now + 60s."""
        sub = await _create_subscription(test_workflow, trigger_id="t_ratelimit")

        adapter = StubPollAdapter(events=[], cursor={"page": 1}, rate_limit_hint=60)
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        before = _utcnow()
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        # Should be ~60s, not 300s
        assert sub.next_poll_at >= before + timedelta(seconds=55)
        assert sub.next_poll_at < before + timedelta(seconds=70)

    async def test_normal_poll_uses_interval_plus_jitter(self, db_engine, test_workflow):
        """Normal case: next_poll_at = now + poll_interval + jitter (0-10%)."""
        sub = await _create_subscription(
            test_workflow, trigger_id="t_normal", poll_interval_seconds=100
        )

        adapter = StubPollAdapter(events=[], cursor={"page": 1})
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        before = _utcnow()
        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        # 100s interval + up to 10% jitter = 100-110s
        assert sub.next_poll_at >= before + timedelta(seconds=95)
        assert sub.next_poll_at < before + timedelta(seconds=115)


# =============================================================================
# Status Recovery
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStatusRecovery:
    """Subscription recovers from error/backoff state on successful poll."""

    async def test_backoff_recovers_to_ok_on_success(self, db_engine, test_workflow):
        """Subscription in 'backoff' state returns to 'ok' after successful poll."""
        sub = await _create_subscription(
            test_workflow,
            trigger_id="t_recover",
            poll_status="backoff",
            poll_error_json={"reason": "adapter_error", "detail": {"retry_after": 60}},
            poll_backoff_seconds=60,
        )

        adapter = StubPollAdapter(
            events=[PolledEvent(payload={"recovery": True}, provider_event_id="evt_recover")],
            cursor={"page": 5},
        )
        defn = _make_trigger_definition()
        dispatcher = AsyncMock()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=dispatcher,
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        assert sub.poll_status == "ok"
        assert sub.poll_error_json is None
        assert sub.poll_backoff_seconds == 0
        assert sub.poll_cursor_json == {"page": 5}
        assert dispatcher.call_count == 1

    async def test_error_recovers_to_ok_on_success(self, db_engine, test_workflow):
        """Subscription in 'error' state returns to 'ok' after successful poll."""
        sub = await _create_subscription(
            test_workflow,
            trigger_id="t_err_recover",
            poll_status="error",
            poll_error_json={"reason": "Unhandled poller exception", "detail": {}},
        )

        adapter = StubPollAdapter(events=[], cursor={"page": 1})
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        await sub.refresh_from_db()
        assert sub.poll_status == "ok"
        assert sub.poll_error_json is None

    async def test_bootstrap_cursor_called_when_no_cursor(self, db_engine, test_workflow):
        """First poll (no cursor) calls adapter.bootstrap_cursor()."""
        sub = await _create_subscription(
            test_workflow, trigger_id="t_bootstrap", poll_cursor_json=None  # explicit
        )

        adapter = StubPollAdapter(events=[], cursor={"page": 1})
        defn = _make_trigger_definition()

        engine = TriggerPollEngine(
            lock_timeout_seconds=60,
            max_batch_size=10,
            trigger_event_dispatcher=AsyncMock(),
        )

        with (
            patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=adapter),
            patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=defn),
        ):
            await engine._process_subscription(sub)

        # Adapter was called (bootstrap then poll)
        assert adapter.poll_count == 1

        # Cursor was saved
        await sub.refresh_from_db()
        assert sub.poll_cursor_json == {"page": 1}
