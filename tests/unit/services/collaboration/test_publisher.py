"""Unit tests for org collaboration event publishing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from seer.services.collaboration.models import CollaborationEvent, CollaborationEventType
from seer.services.collaboration.publisher import (
    ORG_STREAM_KEY_PREFIX,
    ORG_STREAM_TTL_SECONDS,
    OrgEventPublisher,
    publish_collaboration_event,
)


@pytest.mark.asyncio
async def test_org_event_publisher_stream_key():
    publisher = OrgEventPublisher(organization_id=42)
    assert publisher.stream_key == f"{ORG_STREAM_KEY_PREFIX}:42"


@pytest.mark.asyncio
async def test_org_event_publisher_publish_xadd_and_expire():
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value="123-0")
    mock_redis.expire = AsyncMock()

    publisher = OrgEventPublisher(organization_id=12)
    event = CollaborationEvent(
        event_type=CollaborationEventType.WORKFLOW_UPDATED,
        organization_id=12,
        resource_type="workflow",
        resource_id="wf_123",
        payload={"name": "Updated"},
    )

    with patch.object(publisher, "_get_redis", return_value=mock_redis):
        msg_id = await publisher.publish(event)

    assert msg_id == "123-0"
    mock_redis.xadd.assert_called_once()
    fields = mock_redis.xadd.call_args[0][1]
    payload = json.loads(fields["data"])
    assert payload["event_type"] == CollaborationEventType.WORKFLOW_UPDATED.value
    assert payload["resource_id"] == "wf_123"
    mock_redis.expire.assert_called_once_with(f"{ORG_STREAM_KEY_PREFIX}:12", ORG_STREAM_TTL_SECONDS)


@pytest.mark.asyncio
async def test_publish_collaboration_event_builds_event_from_actor():
    actor = SimpleNamespace(
        id=7,
        user_id="user_7",
        first_name="Test",
        last_name="User",
        email="test@example.com",
    )

    with patch("seer.services.collaboration.publisher.OrgEventPublisher.publish", new=AsyncMock(return_value="1-0")) as publish_mock:
        with patch("seer.services.collaboration.publisher.OrgEventPublisher.close", new=AsyncMock()) as close_mock:
            msg_id = await publish_collaboration_event(
                organization_id=33,
                event_type=CollaborationEventType.INVITATION_CREATED,
                resource_type="invitation",
                resource_id="55",
                actor=actor,
                payload={"email": "invitee@example.com"},
                correlation_id="corr-1",
            )

    assert msg_id == "1-0"
    event = publish_mock.await_args.args[0]
    assert event.organization_id == 33
    assert event.actor_db_user_id == 7
    assert event.actor_clerk_user_id == "user_7"
    assert event.payload["actor_name"] == "Test User"
    close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_collaboration_event_skips_when_org_missing():
    with patch("seer.services.collaboration.publisher.OrgEventPublisher.publish", new=AsyncMock()) as publish_mock:
        result = await publish_collaboration_event(
            organization_id=None,
            event_type=CollaborationEventType.WORKFLOW_CREATED,
            resource_type="workflow",
        )

    assert result is None
    publish_mock.assert_not_awaited()
