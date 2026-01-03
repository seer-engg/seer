from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from api.triggers import services as trigger_services
from api.triggers.polling.adapters.base import PolledEvent
from api.triggers.polling.engine import TriggerPollEngine
from api.workflows import models as api_models
from api.workflows import services as workflow_services
from worker.tasks import triggers as worker_trigger_tasks
from shared.database.models_oauth import OAuthConnection
from shared.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
)


async def _create_workflow(user, workflow_spec):
    payload = api_models.WorkflowCreateRequest(
        name="Polling Workflow",
        description="workflow used in polling tests",
        spec=workflow_spec,
    )
    return await workflow_services.create_workflow(user, payload)


@pytest_asyncio.fixture
async def gmail_poll_subscription(db_user, workflow_spec):
    workflow = await _create_workflow(db_user, workflow_spec)
    connection = await OAuthConnection.create(
        user=db_user,
        provider="gmail",
        provider_account_id=f"acct-{uuid4().hex}",
        access_token_enc="test-token",
        refresh_token_enc="test-refresh",
        scopes="https://www.googleapis.com/auth/gmail.readonly",
        status="active",
    )
    response = await workflow_services.create_trigger_subscription(
        db_user,
        api_models.TriggerSubscriptionCreateRequest(
            workflow_id=workflow.workflow_id,
            trigger_key="poll.gmail.email_received",
            provider_connection_id=connection.id,
            bindings={"user_id": 1},
            enabled=True,
        ),
    )
    subscription = await TriggerSubscription.get(id=response.subscription_id)
    await subscription.fetch_related("workflow", "user")
    try:
        yield subscription
    finally:
        await TriggerEvent.filter(
            trigger_key=subscription.trigger_key,
            provider_connection_id=subscription.provider_connection_id,
        ).delete()
        await TriggerSubscription.filter(id=subscription.id).delete()
        await OAuthConnection.filter(id=connection.id).delete()
        await subscription.workflow.delete()


def _sample_polled_event(
    *,
    provider_event_id: str | None = "gmail-event-1",
    occurred_at: datetime | None = None,
):
    return PolledEvent(
        payload={
            "message_id": "abc123",
            "thread_id": "thread123",
            "internal_date_ms": 1735689600000,
        },
        raw={"id": provider_event_id or "shadow-id"},
        provider_event_id=provider_event_id,
        occurred_at=occurred_at or datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_poll_engine_dispatches_polled_events(gmail_poll_subscription, monkeypatch):
    subscription = gmail_poll_subscription
    enqueued_jobs = []

    async def fake_enqueue(*_, **kwargs):
        enqueued_jobs.append(kwargs)

    monkeypatch.setattr(worker_trigger_tasks.process_trigger_event, "kiq", fake_enqueue)

    engine = TriggerPollEngine()
    await engine._handle_events(subscription, [_sample_polled_event()])

    assert len(enqueued_jobs) == 1
    event = await TriggerEvent.get(provider_event_id="gmail-event-1")
    assert event.status == TriggerEventStatus.ROUTED
    assert event.event["data"]["message_id"] == "abc123"


@pytest.mark.asyncio
async def test_poll_engine_dedupes_hash_only_events(gmail_poll_subscription, monkeypatch):
    subscription = gmail_poll_subscription
    enqueued_jobs = []

    async def fake_enqueue(*_, **kwargs):
        enqueued_jobs.append(kwargs)

    monkeypatch.setattr(worker_trigger_tasks.process_trigger_event, "kiq", fake_enqueue)

    engine = TriggerPollEngine()
    occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

    await engine._handle_events(subscription, [_sample_polled_event(provider_event_id=None, occurred_at=occurred_at)])
    await engine._handle_events(subscription, [_sample_polled_event(provider_event_id=None, occurred_at=occurred_at)])

    event_count = await TriggerEvent.filter(
        trigger_key=subscription.trigger_key,
        provider_connection_id=subscription.provider_connection_id,
    ).count()
    assert event_count == 1
    assert len(enqueued_jobs) == 1

