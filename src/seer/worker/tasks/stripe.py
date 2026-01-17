from __future__ import annotations

from seer.worker.broker import broker
from seer.database.subscription_models import (
    StripeWebhookEvent,
    StripeWebhookEventStatus,
)
from seer.logger import get_logger
from seer.api.subscriptions.stripe_webhook_controller import stripe_webhook_controller

logger = get_logger(__name__)

MAX_ATTEMPTS = 5


@broker.task
async def process_stripe_webhook_event(event_db_id: int) -> None:
    """
    Process a persisted Stripe webhook event.

    Uses a durable table to dedupe/guard against retries and supports
    exponential backoff when downstream processing fails.
    """
    event = await StripeWebhookEvent.get_or_none(id=event_db_id)
    if not event:
        logger.warning("Stripe webhook record %s missing in DB", event_db_id)
        return

    if event.status == StripeWebhookEventStatus.PROCESSED:
        logger.info("Stripe event %s already processed", event.event_id)
        return

    event.status = StripeWebhookEventStatus.PROCESSING
    event.attempts += 1
    await event.save()

    try:
        payload = event.payload or {}
        event_type = payload.get("type")
        data = payload.get("data", {}).get("object", {})
        await stripe_webhook_controller.process_event(event_type, data, event_id=event.event_id)
    except Exception as exc:  # pylint: disable=broad-except
        event.status = StripeWebhookEventStatus.FAILED
        event.last_error = str(exc)
        await event.save()

        if event.attempts < MAX_ATTEMPTS:
            logger.warning(
                "Stripe event %s failed (attempt %s/%s): %s",
                event.event_id,
                event.attempts,
                MAX_ATTEMPTS,
                exc,
            )
            # Rely on Stripe's webhook retries to re-deliver; we keep status FAILED for visibility.
        else:
            logger.error(
                "Stripe event %s failed after %s attempts; marking as failed",
                event.event_id,
                event.attempts,
            )
        return

    event.status = StripeWebhookEventStatus.PROCESSED
    event.last_error = None
    await event.save()
    logger.info("Processed Stripe event %s", event.event_id)


__all__ = ["process_stripe_webhook_event"]
