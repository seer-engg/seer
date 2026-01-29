"""
Integration tests for TriggerSubscription and TriggerEvent models.

Tests:
- Trigger subscription CRUD operations
- Trigger event creation and deduplication
- Relationships with workflows and users
- Unique constraints and indexes
- Polling-specific fields
"""
from datetime import datetime, timedelta, timezone

import pytest
from tortoise.exceptions import IntegrityError

from seer.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
)


# =============================================================================
# TriggerSubscription Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_trigger_subscription(db_engine, test_workflow):
    """Test creating a trigger subscription."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="trigger_1",
        trigger_key="gmail.new_email",
        title="Gmail Inbox",
        is_polling=True,
        poll_interval_seconds=300,
    )

    assert subscription.id is not None
    assert subscription.trigger_id == "trigger_1"
    assert subscription.trigger_key == "gmail.new_email"
    assert subscription.title == "Gmail Inbox"
    assert subscription.enabled is True
    assert subscription.is_polling is True
    assert subscription.poll_interval_seconds == 300
    assert subscription.created_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_unique_constraint(db_engine, test_workflow):
    """Test unique constraint on (workflow_id, trigger_id)."""
    await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="unique_trigger",
        trigger_key="test.trigger",
    )

    # Attempt to create duplicate trigger_id for same workflow
    with pytest.raises(IntegrityError):
        await TriggerSubscription.create(
            user=test_workflow.user,
            workflow=test_workflow,
            trigger_id="unique_trigger",  # Duplicate!
            trigger_key="test.trigger",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_cascade_delete(db_engine, test_workflow):
    """Test cascade delete when workflow is deleted."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="cascade_test",
        trigger_key="test.trigger",
    )

    subscription_id = subscription.id

    # Delete workflow
    await test_workflow.delete()

    # Subscription should be deleted
    deleted = await TriggerSubscription.filter(id=subscription_id).first()
    assert deleted is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_relationships(db_engine, test_workflow):
    """Test relationships between TriggerSubscription and other models."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="relationship_test",
        trigger_key="test.trigger",
    )

    # Test workflow relationship
    sub_workflow = await subscription.workflow
    assert sub_workflow.id == test_workflow.id

    # Test user relationship
    sub_user = await subscription.user
    assert sub_user.id == test_workflow.user.id

    # Test reverse relationships
    workflow_subs = await test_workflow.trigger_subscriptions.all()
    assert len(workflow_subs) == 1
    assert workflow_subs[0].id == subscription.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_polling_fields(db_engine, test_workflow):
    """Test polling-specific fields."""
    next_poll = datetime.now(timezone.utc) + timedelta(minutes=5)

    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="poll_test",
        trigger_key="gmail.new_email",
        is_polling=True,
        poll_interval_seconds=300,
        next_poll_at=next_poll,
        poll_cursor_json={"last_message_id": "msg_123"},
        poll_status="ok",
    )

    await subscription.refresh_from_db()
    assert subscription.is_polling is True
    assert subscription.poll_interval_seconds == 300
    assert subscription.poll_cursor_json == {"last_message_id": "msg_123"}
    assert subscription.poll_status == "ok"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_webhook_fields(db_engine, test_workflow):
    """Test webhook-specific fields."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="webhook_test",
        trigger_key="webhook.github",
        is_polling=False,
        filters={"events": ["push", "pull_request"]},
        provider_config={"repo": "owner/repo"},
        secret_token="secret_abc123",
        event_data_schema={"type": "object", "properties": {}},
    )

    await subscription.refresh_from_db()
    assert subscription.is_polling is False
    assert subscription.filters == {"events": ["push", "pull_request"]}
    assert subscription.provider_config == {"repo": "owner/repo"}
    assert subscription.secret_token == "secret_abc123"
    assert subscription.event_data_schema == {"type": "object", "properties": {}}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_form_fields(db_engine, test_workflow):
    """Test form-specific fields."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="form_test",
        trigger_key="form.custom",
        form_suffix="contact-form",
        form_fields=[
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "textarea", "required": True},
        ],
        form_config={"submit_button_text": "Submit", "success_message": "Thanks!"},
    )

    await subscription.refresh_from_db()
    assert subscription.form_suffix == "contact-form"
    assert len(subscription.form_fields) == 2
    assert subscription.form_fields[0]["name"] == "email"
    assert subscription.form_config["submit_button_text"] == "Submit"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_enabled_flag(db_engine, test_workflow):
    """Test enabling/disabling trigger subscriptions."""
    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="enabled_test",
        trigger_key="test.trigger",
        enabled=True,
    )

    assert subscription.enabled is True

    # Disable subscription
    subscription.enabled = False
    await subscription.save()
    await subscription.refresh_from_db()
    assert subscription.enabled is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_subscription_poll_lock(db_engine, test_workflow):
    """Test polling lock mechanism."""
    lock_expires = datetime.now(timezone.utc) + timedelta(minutes=1)

    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="lock_test",
        trigger_key="test.trigger",
        is_polling=True,
        poll_lock_owner="worker_1",
        poll_lock_expires_at=lock_expires,
    )

    await subscription.refresh_from_db()
    assert subscription.poll_lock_owner == "worker_1"
    assert subscription.poll_lock_expires_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_triggers_per_workflow(db_engine, test_workflow):
    """Test creating multiple trigger subscriptions for same workflow."""
    sub1 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="trigger_1",
        trigger_key="gmail.new_email",
    )

    sub2 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="trigger_2",
        trigger_key="webhook.github",
    )

    sub3 = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="trigger_3",
        trigger_key="form.custom",
    )

    # Fetch all subscriptions for workflow
    subs = await TriggerSubscription.filter(workflow=test_workflow).all()
    assert len(subs) == 3

    trigger_ids = {s.trigger_id for s in subs}
    assert trigger_ids == {"trigger_1", "trigger_2", "trigger_3"}


# =============================================================================
# TriggerEvent Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_trigger_event(db_engine):
    """Test creating a trigger event."""
    event = await TriggerEvent.create(
        trigger_key="gmail.new_email",
        provider_connection_id=123,
        provider_event_id="evt_gmail_456",
        occurred_at=datetime.now(timezone.utc),
        event={"subject": "Test Email", "from": "test@example.com"},
        status=TriggerEventStatus.RECEIVED,
    )

    assert event.id is not None
    assert event.trigger_key == "gmail.new_email"
    assert event.provider_connection_id == 123
    assert event.provider_event_id == "evt_gmail_456"
    assert event.event == {"subject": "Test Email", "from": "test@example.com"}
    assert event.status == TriggerEventStatus.RECEIVED
    assert event.received_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_unique_constraint_provider_id(db_engine):
    """Test unique constraint on (trigger_key, provider_connection_id, provider_event_id)."""
    await TriggerEvent.create(
        trigger_key="gmail.new_email",
        provider_connection_id=123,
        provider_event_id="unique_event",
        event={"data": "first"},
    )

    # Attempt to create duplicate event
    with pytest.raises(IntegrityError):
        await TriggerEvent.create(
            trigger_key="gmail.new_email",
            provider_connection_id=123,
            provider_event_id="unique_event",  # Duplicate!
            event={"data": "second"},
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_unique_constraint_event_hash(db_engine):
    """Test unique constraint on (trigger_key, provider_connection_id, event_hash)."""
    await TriggerEvent.create(
        trigger_key="webhook.github",
        provider_connection_id=456,
        event_hash="hash_abc123",
        event={"data": "first"},
    )

    # Attempt to create duplicate with same hash
    with pytest.raises(IntegrityError):
        await TriggerEvent.create(
            trigger_key="webhook.github",
            provider_connection_id=456,
            event_hash="hash_abc123",  # Duplicate!
            event={"data": "second"},
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_status_transitions(db_engine):
    """Test trigger event status transitions."""
    event = await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    # Update to ROUTED
    event.status = TriggerEventStatus.ROUTED
    await event.save()
    await event.refresh_from_db()
    assert event.status == TriggerEventStatus.ROUTED

    # Update to PROCESSED
    event.status = TriggerEventStatus.PROCESSED
    await event.save()
    await event.refresh_from_db()
    assert event.status == TriggerEventStatus.PROCESSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_with_error(db_engine):
    """Test storing error information in failed event."""
    event = await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"data": "test"},
        status=TriggerEventStatus.FAILED,
        error={"message": "Processing failed", "code": "TIMEOUT"},
    )

    await event.refresh_from_db()
    assert event.status == TriggerEventStatus.FAILED
    assert event.error == {"message": "Processing failed", "code": "TIMEOUT"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_raw_payload(db_engine):
    """Test storing raw payload alongside normalized event."""
    raw_payload = {
        "headers": {"X-GitHub-Event": "push"},
        "body": {"ref": "refs/heads/main", "commits": []},
    }

    event = await TriggerEvent.create(
        trigger_key="webhook.github",
        event={"type": "push", "branch": "main"},
        raw_payload=raw_payload,
    )

    await event.refresh_from_db()
    assert event.event == {"type": "push", "branch": "main"}
    assert event.raw_payload == raw_payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_nullable_fields(db_engine):
    """Test that nullable fields can be None."""
    event = await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"minimal": "event"},
    )

    assert event.provider_connection_id is None
    assert event.provider_event_id is None
    assert event.event_hash is None
    assert event.occurred_at is None
    assert event.raw_payload is None
    assert event.error is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_events_by_trigger_key(db_engine):
    """Test querying events by trigger_key."""
    await TriggerEvent.create(
        trigger_key="gmail.new_email",
        event={"id": "1"},
    )
    await TriggerEvent.create(
        trigger_key="webhook.github",
        event={"id": "2"},
    )
    await TriggerEvent.create(
        trigger_key="gmail.new_email",
        event={"id": "3"},
    )

    gmail_events = await TriggerEvent.filter(trigger_key="gmail.new_email").all()
    assert len(gmail_events) == 2
    assert {e.event["id"] for e in gmail_events} == {"1", "3"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_events_by_status(db_engine):
    """Test querying events by status."""
    await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"id": "1"},
        status=TriggerEventStatus.RECEIVED,
    )
    await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"id": "2"},
        status=TriggerEventStatus.PROCESSED,
    )
    await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"id": "3"},
        status=TriggerEventStatus.RECEIVED,
    )

    received_events = await TriggerEvent.filter(status=TriggerEventStatus.RECEIVED).all()
    assert len(received_events) == 2
