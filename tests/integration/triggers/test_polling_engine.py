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


# =============================================================================
# Auto-Selection Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_auto_selection_updates_subscription(db_engine, test_workflow):
    """Test that runtime auto-selection updates subscription with selected connection."""
    from seer.database.models_oauth import OAuthConnection

    # Create an active OAuth connection
    oauth_connection = await OAuthConnection.create(
        user=test_workflow.user,
        provider="google",
        provider_account_id="test@example.com",
        access_token_enc="encrypted_token",
        status="active",
    )

    # Create subscription without provider_connection_id
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="auto_select_test",
        trigger_key="gmail.new_email",
        is_polling=True,
        enabled=True,
        next_poll_at=utcnow() - timedelta(minutes=5),
        poll_interval_seconds=300,
        provider_connection_id=None,  # No connection set
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Mock trigger definition
    mock_definition = MagicMock()
    mock_definition.provider = "gmail"
    mock_definition.meta = MagicMock()
    mock_definition.meta.requires_connection = True

    with patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=mock_definition), \
         patch("seer.core.triggers.polling.engine.get_oauth_token", return_value=(oauth_connection, "access_token")):

        result = await engine._get_oauth_connection(sub, test_workflow.user)

        # Verify connection was auto-selected
        assert result is not None
        connection, token = result
        assert connection.id == oauth_connection.id
        assert token == "access_token"

        # Verify subscription was updated with auto-selected connection
        await sub.refresh_from_db()
        assert sub.provider_connection_id == oauth_connection.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_auto_selection_persists_to_database(db_engine, test_workflow):
    """Test that auto-selected connection is persisted in database."""
    from seer.database.models_oauth import OAuthConnection

    # Create multiple OAuth connections (should select most recent)
    old_connection = await OAuthConnection.create(
        user=test_workflow.user,
        provider="google",
        provider_account_id="old@example.com",
        access_token_enc="encrypted_token_old",
        status="active",
        created_at=utcnow() - timedelta(days=7),
    )

    new_connection = await OAuthConnection.create(
        user=test_workflow.user,
        provider="google",
        provider_account_id="new@example.com",
        access_token_enc="encrypted_token_new",
        status="active",
        created_at=utcnow(),  # Most recent
    )

    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="persist_test",
        trigger_key="gmail.new_email",
        is_polling=True,
        enabled=True,
        next_poll_at=utcnow() - timedelta(minutes=5),
        poll_interval_seconds=300,
        provider_connection_id=None,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    mock_definition = MagicMock()
    mock_definition.provider = "gmail"
    mock_definition.meta = MagicMock()
    mock_definition.meta.requires_connection = True

    with patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=mock_definition), \
         patch("seer.core.triggers.polling.engine.get_oauth_token", return_value=(new_connection, "token")):

        await engine._get_oauth_connection(sub, test_workflow.user)

        # Verify most recent connection was selected and persisted
        await sub.refresh_from_db()
        assert sub.provider_connection_id == new_connection.id
        assert sub.provider_connection_id != old_connection.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_auto_selection_no_connection_marks_error(db_engine, test_workflow):
    """Test that subscription is marked as error when no connection available."""
    # Create subscription without provider_connection_id and no OAuth connections
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="no_conn_test",
        trigger_key="gmail.new_email",
        is_polling=True,
        enabled=True,
        next_poll_at=utcnow() - timedelta(minutes=5),
        poll_interval_seconds=300,
        provider_connection_id=None,
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    mock_definition = MagicMock()
    mock_definition.provider = "gmail"
    mock_definition.meta = MagicMock()
    mock_definition.meta.requires_connection = True

    with patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=mock_definition):
        result = await engine._get_oauth_connection(sub, test_workflow.user)

        # Should return None since no connection available
        assert result is None

        # Verify subscription was marked as error
        await sub.refresh_from_db()
        assert sub.poll_status == "error"
        assert sub.poll_error_json is not None
        assert sub.poll_error_json["reason"] == "missing_provider_connection"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_auto_selection_explicit_connection_unchanged(db_engine, test_workflow):
    """Test that existing explicit connection is not overridden."""
    from seer.database.models_oauth import OAuthConnection

    # Create two OAuth connections
    explicit_connection = await OAuthConnection.create(
        user=test_workflow.user,
        provider="google",
        provider_account_id="explicit@example.com",
        access_token_enc="encrypted_token_explicit",
        status="active",
    )

    newer_connection = await OAuthConnection.create(
        user=test_workflow.user,
        provider="google",
        provider_account_id="newer@example.com",
        access_token_enc="encrypted_token_newer",
        status="active",
        created_at=utcnow() + timedelta(seconds=10),  # More recent
    )

    # Create subscription WITH explicit provider_connection_id
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="explicit_test",
        trigger_key="gmail.new_email",
        is_polling=True,
        enabled=True,
        next_poll_at=utcnow() - timedelta(minutes=5),
        poll_interval_seconds=300,
        provider_connection_id=explicit_connection.id,  # Explicit connection set
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    mock_definition = MagicMock()
    mock_definition.provider = "gmail"

    with patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=mock_definition), \
         patch("seer.core.triggers.polling.engine.get_oauth_token", return_value=(explicit_connection, "token")) as mock_get_token:

        result = await engine._get_oauth_connection(sub, test_workflow.user)

        # Should use explicit connection, not auto-select
        assert result is not None
        connection, token = result
        assert connection.id == explicit_connection.id

        # Verify subscription connection_id unchanged
        await sub.refresh_from_db()
        assert sub.provider_connection_id == explicit_connection.id

        # Verify get_oauth_token was called with explicit connection
        mock_get_token.assert_called_once_with(
            test_workflow.user,
            connection_id=str(explicit_connection.id)
        )


# =============================================================================
# Missing Workflow Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_handle_events_disables_subscription_when_workflow_missing(db_engine, test_workflow):
    """Test that subscription is disabled when workflow is deleted.

    Regression test: Previously, _handle_events() would set poll_status='disabled'
    but then _mark_success() would overwrite it to 'ok', causing infinite loops.
    """
    past_time = utcnow() - timedelta(minutes=5)

    # Create subscription
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="missing_workflow_test",
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

    # Mock events to simulate having events to process
    mock_event = MagicMock()
    mock_event.payload = {"test": "data"}
    mock_event.raw = {}
    mock_event.occurred_at = utcnow()
    mock_event.provider_event_id = "test-event-id"
    events = [mock_event]

    # Set workflow to None to simulate deleted workflow
    sub.workflow = None

    # Call _handle_events - should disable and return True
    disabled = await engine._handle_events(sub, events)

    assert disabled is True
    await sub.refresh_from_db()
    assert sub.poll_status == "disabled"
    assert sub.poll_error_json is not None
    assert sub.poll_error_json["reason"] == "missing_workflow"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_subscription_skips_mark_success_when_workflow_missing(db_engine, test_workflow):
    """Test that _mark_success is NOT called when subscription is disabled due to missing workflow.

    This is the key test for the regression - previously mark_success would overwrite
    the 'disabled' status back to 'ok'.
    """
    past_time = utcnow() - timedelta(minutes=5)

    # Create subscription
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="skip_mark_success_test",
        trigger_key="gmail.new_email",
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

    # Mock adapter that returns events
    mock_adapter = MagicMock()
    mock_poll_result = MagicMock()
    mock_poll_result.events = [MagicMock(payload={}, raw={}, occurred_at=utcnow(), provider_event_id="evt-1")]
    mock_poll_result.cursor = {"test": "cursor"}
    mock_poll_result.has_more = False
    mock_poll_result.rate_limit_hint = None
    mock_adapter.poll = AsyncMock(return_value=mock_poll_result)
    mock_adapter.bootstrap_cursor = AsyncMock(return_value={})

    # Mock trigger definition
    mock_definition = MagicMock()
    mock_definition.provider = "gmail"
    mock_definition.meta = MagicMock()
    mock_definition.meta.requires_connection = False

    # Mock _handle_events to simulate workflow being missing (returns True = disabled)
    # This avoids the cascade delete issue when actually deleting the workflow
    async def mock_handle_events(subscription, events):
        subscription.poll_status = "disabled"
        subscription.poll_error_json = {"reason": "missing_workflow", "detail": None}
        subscription.poll_lock_owner = None
        subscription.poll_lock_expires_at = None
        await subscription.save(
            update_fields=["poll_status", "poll_error_json", "poll_lock_owner", "poll_lock_expires_at"]
        )
        return True  # Signal that subscription was disabled

    with patch("seer.core.triggers.polling.engine.adapter_registry.get", return_value=mock_adapter), \
         patch("seer.core.triggers.polling.engine.trigger_registry.get", return_value=mock_definition), \
         patch.object(engine, "_handle_events", side_effect=mock_handle_events) as mock_handle, \
         patch.object(engine, "_mark_success") as mock_mark_success:

        await engine._process_subscription(sub)

        # Verify _handle_events was called
        mock_handle.assert_called_once()

        # Verify _mark_success was NOT called (this is the key assertion)
        mock_mark_success.assert_not_called()

        # Verify subscription is disabled
        await sub.refresh_from_db()
        assert sub.poll_status == "disabled"
        assert sub.poll_error_json["reason"] == "missing_workflow"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disabled_subscription_not_leased_again(db_engine, test_workflow):
    """Test that a disabled subscription is never picked up for polling again."""
    past_time = utcnow() - timedelta(minutes=5)

    # Create a disabled subscription
    sub = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="disabled_poll_status_test",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,  # Still enabled flag
        next_poll_at=past_time,
        poll_interval_seconds=300,
        poll_status="disabled",  # But poll_status is disabled
        poll_error_json={"reason": "missing_workflow"},
    )

    mock_dispatcher = AsyncMock()
    engine = TriggerPollEngine(
        lock_timeout_seconds=60,
        max_batch_size=10,
        trigger_event_dispatcher=mock_dispatcher,
    )

    leased = await engine._lease_due_subscriptions(limit=10)

    # Should NOT lease disabled subscription
    assert len(leased) == 0
