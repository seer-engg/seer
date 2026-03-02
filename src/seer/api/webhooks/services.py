from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from fastapi import HTTPException, status

from seer.database import (
    TriggerEvent,
    TriggerSubscription,
)
from seer.logger import get_logger
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope, persist_event
from seer.worker.trigger_dispatcher import dispatch_trigger_event

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_active_subscription_by_slug(webhook_slug: str) -> TriggerSubscription:
    """Look up an active subscription by its unique webhook slug."""
    subscription = await TriggerSubscription.filter(webhook_slug=webhook_slug).first()
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        )
    if not subscription.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not active",
        )
    return subscription


def _load_trigger_provider(trigger_key: str) -> str:
    definition = trigger_registry.maybe_get(trigger_key)
    if definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trigger '{trigger_key}' is not registered",
        )
    return definition.provider



async def _handle_webhook_event(
    subscription: TriggerSubscription,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    provider_event_id: Optional[str],
) -> TriggerEvent:
    """Core webhook event handler (no auth checks - caller must verify access)."""
    provider = _load_trigger_provider(subscription.trigger_key)
    raw_payload = {
        "headers": dict(headers),
        "body": payload,
    }
    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=subscription.trigger_id,
            trigger_key=subscription.trigger_key,
            title=subscription.title,
            provider=provider,
            provider_connection_id=subscription.provider_connection_id,
            payload=payload,
            raw=raw_payload,
            occurred_at=_utcnow(),
        )
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


async def handle_generic_webhook_by_slug(
    webhook_slug: str,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    provider_event_id: Optional[str],
) -> TriggerEvent:
    """Handle a webhook event using slug-based URL security (no secret verification needed)."""
    logger.info(
        "Handling generic webhook by slug",
        extra={"webhook_slug": webhook_slug, "provider_event_id": provider_event_id},
    )
    subscription = await _get_active_subscription_by_slug(webhook_slug)
    return await _handle_webhook_event(
        subscription,
        payload=payload,
        headers=headers,
        provider_event_id=provider_event_id,
    )


async def handle_webhook_for_subscription(
    subscription: TriggerSubscription,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    provider_event_id: Optional[str] = None,
) -> TriggerEvent:
    """Handle a webhook event for a known subscription (e.g., form submissions)."""
    logger.info(
        "Handling webhook for subscription",
        extra={"subscription_id": subscription.id, "provider_event_id": provider_event_id},
    )
    if not subscription.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not active",
        )
    return await _handle_webhook_event(
        subscription,
        payload=payload,
        headers=headers,
        provider_event_id=provider_event_id,
    )
