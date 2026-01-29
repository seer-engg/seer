"""
Integration tests for trigger polling engine.

Tests:
- Subscription leasing and locking
- Poll adapter execution
- Event processing and persistence
- Error handling and retry logic
- Lock expiration and cleanup
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.triggers.polling.engine import TriggerPollEngine
from seer.database.workflow_models import TriggerEvent, TriggerEventStatus, TriggerSubscription


# =============================================================================
# Helper Functions
# =============================================================================


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# TriggerPollEngine Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_due_subscriptions(db_engine, test_workflow):
    """Test leasing subscriptions that are due for polling."""
    past_time = utcnow() - timedelta(minutes=5)

    # Create subscriptions - some due, some not
    sub1 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub1",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,  # Due
        poll_interval_seconds=300,
    )

    sub2 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub2",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,  # Due
        poll_interval_seconds=300,
    )

    future_time = utcnow() + timedelta(minutes=5)
    sub3 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub3",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=future_time,  # Not due yet
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Lease subscriptions
    leased = await engine._lease_due_subscriptions(limit=10)

    # Should lease only due subscriptions
    assert len(leased) == 2
    leased_ids = {s.id for s in leased}
    assert leased_ids == {sub1.id, sub2.id}

    # Verify locks are set
    for sub in leased:
        assert sub.poll_lock_owner == engine.worker_id
        assert sub.poll_lock_expires_at > utcnow()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_respects_limit(db_engine, test_workflow):
    """Test that leasing respects the batch size limit."""
    past_time = utcnow() - timedelta(minutes=5)

    # Create 5 due subscriptions
    for i in range(5):
        await TriggerSubscription.create(
            user=test_workflow.user,
            workflow=test_workflow,
            trigger_id=f"sub{i}",
            trigger_key="test.trigger",
            is_polling=True,
            enabled=True,
            next_poll_at=past_time,
            poll_interval_seconds=300,
        )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=3,  # Limit to 3
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=3)

    # Should lease only 3 subscriptions
    assert len(leased) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_skips_locked_subscriptions(db_engine, test_workflow):
    """Test that already-locked subscriptions are skipped."""
    past_time = utcnow() - timedelta(minutes=5)
    lock_expiry = utcnow() + timedelta(minutes=5)

    # Create locked subscription
    locked_sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="locked",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
        poll_lock_owner="other_worker",
        poll_lock_expires_at=lock_expiry,
    )

    # Create unlocked subscription
    unlocked_sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="unlocked",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should only lease unlocked subscription
    assert len(leased) == 1
    assert leased[0].id == unlocked_sub.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_expired_locks_can_be_reacquired(db_engine, test_workflow):
    """Test that expired locks can be reacquired."""
    past_time = utcnow() - timedelta(minutes=5)
    expired_lock = utcnow() - timedelta(minutes=1)  # Expired

    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="expired_lock",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
        poll_lock_owner="old_worker",
        poll_lock_expires_at=expired_lock,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should lease subscription with expired lock
    assert len(leased) == 1
    assert leased[0].id == sub.id
    assert leased[0].poll_lock_owner == engine.worker_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_skips_disabled_subscriptions(db_engine, test_workflow):
    """Test that disabled subscriptions are not leased."""
    past_time = utcnow() - timedelta(minutes=5)

    # Create disabled subscription
    await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="disabled",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=False,  # Disabled
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should not lease disabled subscription
    assert len(leased) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lease_skips_non_polling_subscriptions(db_engine, test_workflow):
    """Test that non-polling subscriptions (webhooks, forms) are not leased."""
    past_time = utcnow() - timedelta(minutes=5)

    # Create webhook subscription (not polling)
    await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="webhook",
        trigger_key="webhook.github",
        is_polling=False,  # Not polling
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should not lease non-polling subscription
    assert len(leased) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_processes_due_subscriptions(db_engine, test_workflow):
    """Test that tick processes due subscriptions."""
    past_time = utcnow() - timedelta(minutes=5)

    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="tick_test",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Mock the _process_subscription method
    with patch.object(engine, "_process_subscription") as mock_process:
        mock_process.return_value = None

        await engine.tick()

        # Verify subscription was processed
        mock_process.assert_called_once()
        processed_sub = mock_process.call_args[0][0]
        assert processed_sub.id == sub.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_handles_no_due_subscriptions(db_engine):
    """Test that tick handles case with no due subscriptions gracefully."""
    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Should not raise error when no subscriptions
    await engine.tick()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_handles_processing_errors(db_engine, test_workflow):
    """Test that tick handles processing errors and continues."""
    past_time = utcnow() - timedelta(minutes=5)

    sub1 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="error_sub",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    sub2 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="success_sub",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=past_time,
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    processed_subs = []

    async def mock_process(sub):
        processed_subs.append(sub.id)
        if sub.id == sub1.id:
            raise ValueError("Processing failed")

    with patch.object(engine, "_process_subscription", side_effect=mock_process), \
         patch.object(engine, "_mark_error") as mock_mark_error:

        await engine.tick()

        # Both subscriptions should be attempted
        assert len(processed_subs) == 2
        assert sub1.id in processed_subs
        assert sub2.id in processed_subs

        # Error subscription should be marked as error
        mock_mark_error.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscription_ordering_by_next_poll_at(db_engine, test_workflow):
    """Test that subscriptions are processed in order of next_poll_at."""
    base_time = utcnow()

    # Create subscriptions with different next_poll_at times
    sub1 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub1",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=base_time - timedelta(minutes=10),  # Oldest
        poll_interval_seconds=300,
    )

    sub2 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub2",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=base_time - timedelta(minutes=5),  # Middle
        poll_interval_seconds=300,
    )

    sub3 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="sub3",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
        next_poll_at=base_time - timedelta(minutes=1),  # Newest
        poll_interval_seconds=300,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should be ordered by next_poll_at ascending
    assert len(leased) == 3
    assert leased[0].id == sub1.id  # Oldest first
    assert leased[1].id == sub2.id
    assert leased[2].id == sub3.id
