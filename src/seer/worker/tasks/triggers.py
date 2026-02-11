from __future__ import annotations

from seer.worker.broker_instance import broker
from seer.logger import get_logger
from seer.services.workflows.triggers import process_trigger_event
from seer.database import TriggerSubscription
from seer.observability.sentry_client import set_user_context, set_tag, set_context

logger = get_logger(__name__)


async def _set_sentry_context_for_trigger(subscription_id: int, event_id: int) -> None:
    """
    Set Sentry context for trigger event error tracking.

    Sets user context (id, email, username) and trigger event context.
    All operations are non-blocking and fail silently.
    """
    set_tag("task_type", "trigger_event")
    set_tag("subscription_id", str(subscription_id))
    set_tag("event_id", str(event_id))

    try:
        subscription = await TriggerSubscription.get(id=subscription_id)
        await subscription.fetch_related("user", "workflow")

        user = subscription.user
        if user:
            set_user_context(
                user_id=user.user_id,
                email=getattr(user, "email", None),
                username=f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip() or None,
            )
            # Set user tags for indexed searching in Sentry
            set_tag("user_id", user.user_id)
            if getattr(user, "email", None):
                set_tag("user_email", user.email)

        set_context("trigger_event", {
            "subscription_id": subscription_id,
            "event_id": event_id,
            "workflow_id": getattr(subscription.workflow, "workflow_id", None),
            "trigger_key": getattr(subscription, "trigger_key", None),
        })
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Sentry context setup must never block task execution
        logger.debug("Failed to set Sentry context for trigger", exc_info=True)


@broker.task
async def trigger_event_task(subscription_id: int, event_id: int) -> None:
    """Process a trigger event by running the workflow bindings and execution."""
    logger.info(
        "Processing trigger event via Taskiq",
        extra={"subscription_id": subscription_id, "event_id": event_id},
    )

    await _set_sentry_context_for_trigger(subscription_id, event_id)

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
