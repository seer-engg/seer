from __future__ import annotations

from seer.worker.broker import broker
from seer.logger import get_logger
from seer.services.workflows.triggers import process_trigger_event
logger = get_logger(__name__)


@broker.task
async def trigger_event_task(subscription_id: int, event_id: int) -> None:
    """Process a trigger event by running the workflow bindings and execution."""
    logger.info(
        "Processing trigger event via Taskiq",
        extra={"subscription_id": subscription_id, "event_id": event_id},
    )

    try:
        await process_trigger_event(subscription_id=subscription_id, event_id=event_id)
        logger.info(
            "Trigger event processing completed",
            extra={"subscription_id": subscription_id, "event_id": event_id},
        )
    except Exception:
        logger.exception(
            "Trigger event processing failed with exception",
            extra={"subscription_id": subscription_id, "event_id": event_id},
        )
        raise


__all__ = ["trigger_event_task"]
