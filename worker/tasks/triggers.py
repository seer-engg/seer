from __future__ import annotations

from worker.broker import broker
from shared.logger import get_logger

logger = get_logger(__name__)


@broker.task
async def process_trigger_event(subscription_id: int, event_id: int) -> None:
    """Process a trigger event by running the workflow bindings and execution."""
    logger.info(
        "Processing trigger event via Taskiq",
        extra={"subscription_id": subscription_id, "event_id": event_id},
    )
    from api.triggers.services import process_trigger_run_job  # pylint: disable=import-outside-toplevel # Reason: avoiding circular imports

    try:
        await process_trigger_run_job(subscription_id=subscription_id, event_id=event_id)
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


__all__ = ["process_trigger_event"]
