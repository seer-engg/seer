from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from fastapi import HTTPException, status
from tortoise.exceptions import DoesNotExist

from seer.database import (
    TriggerEvent,
    TriggerSubscription,
)
from seer.logger import get_logger
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.triggers.events import build_event_envelope
from seer.worker.trigger_dispatcher import dispatch_trigger_event
from seer.core.triggers.events import persist_event

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_active_subscription(subscription_id: int) -> TriggerSubscription:
    try:
        subscription = await TriggerSubscription.get(id=subscription_id)
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        ) from None
    if not subscription.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not active",
        )
    return subscription


def _verify_secret(subscription: TriggerSubscription, provided: Optional[str]) -> None:
    expected = subscription.secret_token
    if not expected:
        return
    if not provided or provided.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )


def _load_trigger_provider(trigger_key: str) -> str:
    definition = trigger_registry.maybe_get(trigger_key)
    if definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trigger '{trigger_key}' is not registered",
        )
    return definition.provider



async def handle_generic_webhook(
    subscription_id: int,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    secret: Optional[str],
    provider_event_id: Optional[str],
) -> TriggerEvent:
    logger.info(
        "Handling generic webhook",
        extra={"subscription_id": subscription_id, "provider_event_id": provider_event_id},
    )
    subscription = await _get_active_subscription(subscription_id)
    _verify_secret(subscription, secret)
    provider = _load_trigger_provider(subscription.trigger_key)
    raw_payload = {
        "headers": dict(headers),
        "body": payload,
    }
    envelope = build_event_envelope(
        trigger_id=subscription.trigger_id,
        trigger_key=subscription.trigger_key,
        title=subscription.title,
        provider=provider,
        provider_connection_id=subscription.provider_connection_id,
        payload=payload,
        raw=raw_payload,
        occurred_at=_utcnow(),
    )
    event, created = await persist_event(
        subscription=subscription,
        envelope=envelope,
        provider_event_id=provider_event_id,
        event_hash=None,
        raw=raw_payload,
    )
    if created:
        await dispatch_trigger_event(subscription, event, envelope)
    return event
