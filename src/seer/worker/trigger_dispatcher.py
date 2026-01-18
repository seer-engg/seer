from __future__ import annotations
from typing import Any, Dict
from fastapi import HTTPException
from starlette import status
from seer.database import TriggerSubscription, TriggerEvent, TriggerEventStatus
from seer.logger import get_logger
logger = get_logger(__name__)

from seer.worker.tasks.triggers import trigger_event_task

async def dispatch_trigger_event(
    subscription: TriggerSubscription,
    event: TriggerEvent,
    envelope: Dict[str, Any],
) -> None:
    # Defensive check: verify workflow and user exist before dispatching
    await subscription.fetch_related("workflow", "user")
    if not subscription.workflow or not subscription.user:
        logger.error(
            "Cannot dispatch event - workflow or user missing",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Workflow or user missing for subscription"},
        )
        return

    try:
        await trigger_event_task.kiq(subscription_id=subscription.id, event_id=event.id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue trigger event",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Failed to enqueue trigger event"},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue trigger event",
        ) from exc
    logger.info(
        "Trigger event enqueued",
        extra={"event_id": event.id, "subscription_id": subscription.id},
    )
    await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.ROUTED)
