from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4
from tortoise.exceptions import IntegrityError
from datetime import timezone

from seer.database import TriggerSubscription, TriggerEvent, TriggerEventStatus
from seer.logger import get_logger
logger = get_logger(__name__)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def build_event_envelope(
    *,
    trigger_id: str,
    trigger_key: str,
    title: str,
    provider: str,
    provider_connection_id: Optional[int],
    payload: Dict[str, Any],
    raw: Optional[Dict[str, Any]],
    occurred_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    occurred = occurred_at or _utcnow()
    return {
        "id": f"evt_{uuid4().hex}",
        "trigger_id": trigger_id,  # Unique instance identifier
        "trigger_key": trigger_key,  # Trigger type identifier
        "title": title,  # Human-readable title for reference resolution
        "provider": provider,
        "account_id": provider_connection_id,
        "occurred_at": occurred.isoformat(),
        "received_at": _utcnow().isoformat(),
        "data": payload,
        "raw": raw,
    }




async def persist_event(
    *,
    subscription: TriggerSubscription,
    envelope: Dict[str, Any],
    provider_event_id: Optional[str],
    event_hash: Optional[str],
    raw: Optional[Dict[str, Any]],
) -> tuple[TriggerEvent, bool]:
    occurred_at_str = envelope.get("occurred_at")
    occurred_at = (
        datetime.fromisoformat(occurred_at_str)
        if isinstance(occurred_at_str, str)
        else _utcnow()
    )
    try:
        event = await TriggerEvent.create(
            trigger_key=subscription.trigger_key,
            provider_connection_id=subscription.provider_connection_id,
            provider_event_id=provider_event_id,
            event_hash=event_hash,
            occurred_at=occurred_at,
            event=envelope,
            raw_payload=raw,
            status=TriggerEventStatus.RECEIVED,
        )
        return event, True
    except IntegrityError:
        dedupe_filters = {
            "trigger_key": subscription.trigger_key,
            "provider_connection_id": subscription.provider_connection_id,
        }
        dedupe_key = None
        if provider_event_id:
            dedupe_filters["provider_event_id"] = provider_event_id
            dedupe_key = ("provider_event_id", provider_event_id)
        elif event_hash:
            dedupe_filters["event_hash"] = event_hash
            dedupe_key = ("event_hash", event_hash)
        if dedupe_key:
            existing = await TriggerEvent.get(**dedupe_filters)
            logger.info(
                "Deduped trigger event",
                extra={
                    "trigger_key": subscription.trigger_key,
                    "subscription_id": subscription.id,
                    dedupe_key[0]: dedupe_key[1],
                },
            )
            return existing, False
        raise
