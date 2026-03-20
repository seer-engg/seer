from __future__ import annotations

from typing import Any, Optional

from seer.database import User
from seer.logger import get_logger

from .models import CollaborationEvent, CollaborationEventType

logger = get_logger(__name__)

ORG_STREAM_TTL_SECONDS = 7200
ORG_STREAM_KEY_PREFIX = "org:events"


class OrgEventPublisher:
    """Publishes organization collaboration events to a Redis Stream."""

    def __init__(self, organization_id: int):
        self.organization_id = organization_id
        self.stream_key = f"{ORG_STREAM_KEY_PREFIX}:{organization_id}"
        self._redis: Optional[object] = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: optional lazy import
            from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: avoid circular import at module load time

            self._redis = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
            )
        return self._redis

    async def publish(self, event: CollaborationEvent) -> str | None:
        try:
            redis = await self._get_redis()
            msg_id = await redis.xadd(self.stream_key, {"data": event.model_dump_json()})
            await redis.expire(self.stream_key, ORG_STREAM_TTL_SECONDS)
            return msg_id
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: collaboration publishing is best-effort
            logger.warning(
                "OrgEventPublisher.publish failed for org=%s event=%s: %s",
                self.organization_id,
                event.event_type.value,
                exc,
            )
            return None

    async def close(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:  # pylint: disable=broad-exception-caught # Reason: close is best-effort
                pass
            self._redis = None


def _user_display_name(user: User | None) -> str | None:
    if user is None:
        return None
    display_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return display_name or user.email or user.user_id


async def publish_collaboration_event(
    *,
    organization_id: int | None,
    event_type: CollaborationEventType,
    resource_type: str,
    resource_id: str | None = None,
    actor: User | None = None,
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> str | None:
    """Publish an org-scoped collaboration event if an org is available."""
    if organization_id is None:
        return None

    event_payload = dict(payload or {})
    actor_name = _user_display_name(actor)
    if actor_name:
        event_payload.setdefault("actor_name", actor_name)

    event = CollaborationEvent(
        event_type=event_type,
        organization_id=organization_id,
        actor_clerk_user_id=actor.user_id if actor else None,
        actor_db_user_id=actor.id if actor else None,
        resource_type=resource_type,
        resource_id=resource_id,
        correlation_id=correlation_id,
        payload=event_payload,
    )

    publisher = OrgEventPublisher(organization_id)
    try:
        return await publisher.publish(event)
    finally:
        await publisher.close()
