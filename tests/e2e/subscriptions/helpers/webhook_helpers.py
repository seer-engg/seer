"""
Webhook verification helpers for E2E subscription testing.

Provides utilities for verifying webhook delivery and processing.
"""
import asyncio
from typing import Optional

from seer.database.subscription_models import (
    StripeWebhookEvent,
    StripeWebhookEventStatus,
)
from seer.logger import get_logger

logger = get_logger("tests.webhook_helpers")


class WebhookVerifier:
    """
    Helper for verifying webhook delivery and processing in tests.

    Provides methods to wait for webhooks and verify their processing status.
    """

    def __init__(self, timeout: float = 10.0):
        """
        Initialize the webhook verifier.

        Args:
            timeout: Default timeout in seconds for webhook operations
        """
        self.timeout = timeout

    async def wait_for_webhook(
        self, event_id: str, timeout: Optional[float] = None
    ) -> Optional[StripeWebhookEvent]:
        """
        Wait for a webhook event to be received and stored in the database.

        Args:
            event_id: The Stripe event ID to wait for
            timeout: Max time to wait in seconds (uses default if not provided)

        Returns:
            The webhook event record if found, None if timeout
        """
        timeout = timeout or self.timeout
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            event = await StripeWebhookEvent.get_or_none(event_id=event_id)
            if event:
                logger.info("Webhook event %s received", event_id)
                return event

            await asyncio.sleep(0.1)  # Check every 100ms

        logger.warning("Timeout waiting for webhook event %s", event_id)
        return None

    async def verify_webhook_processed(
        self, event_id: str, timeout: Optional[float] = None
    ) -> bool:
        """
        Verify that a webhook event has been processed successfully.

        Args:
            event_id: The Stripe event ID to verify
            timeout: Max time to wait in seconds (uses default if not provided)

        Returns:
            True if processed successfully, False otherwise
        """
        timeout = timeout or self.timeout
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            event = await StripeWebhookEvent.get_or_none(event_id=event_id)

            if event and event.status == StripeWebhookEventStatus.PROCESSED:
                logger.info("Webhook event %s processed successfully", event_id)
                return True

            if event and event.status == StripeWebhookEventStatus.FAILED:
                logger.error(
                    "Webhook event %s failed processing: %s",
                    event_id,
                    event.error_message,
                )
                return False

            await asyncio.sleep(0.1)

        logger.warning("Timeout waiting for webhook event %s to be processed", event_id)
        return False

    async def verify_webhook_idempotency(self, event_id: str) -> bool:
        """
        Verify that a webhook event can be reprocessed idempotently.

        Args:
            event_id: The Stripe event ID to check

        Returns:
            True if idempotency works correctly, False otherwise
        """
        # Get the event record
        event = await StripeWebhookEvent.get_or_none(event_id=event_id)
        if not event:
            logger.error("Webhook event %s not found", event_id)
            return False

        if event.status != StripeWebhookEventStatus.PROCESSED:
            logger.error(
                "Webhook event %s not in PROCESSED state: %s", event_id, event.status
            )
            return False

        # Try to create duplicate (should be prevented by unique constraint)
        from tortoise.exceptions import IntegrityError

        try:
            await StripeWebhookEvent.create(
                event_id=event_id,
                type=event.type,
                payload=event.payload,
                status=StripeWebhookEventStatus.RECEIVED,
            )
            logger.error(
                "Duplicate webhook event %s was not prevented by unique constraint",
                event_id,
            )
            return False
        except IntegrityError:
            # Expected - duplicate prevented
            logger.info("Webhook event %s idempotency verified", event_id)
            return True

    async def get_webhook_error(self, event_id: str) -> Optional[str]:
        """
        Get the error message for a failed webhook event.

        Args:
            event_id: The Stripe event ID

        Returns:
            The error message if failed, None otherwise
        """
        event = await StripeWebhookEvent.get_or_none(event_id=event_id)
        if event and event.status == StripeWebhookEventStatus.FAILED:
            return event.error_message
        return None

    async def count_webhook_attempts(self, event_id: str) -> int:
        """
        Count the number of processing attempts for a webhook event.

        Args:
            event_id: The Stripe event ID

        Returns:
            Number of processing attempts
        """
        event = await StripeWebhookEvent.get_or_none(event_id=event_id)
        if not event:
            return 0

        # Check if event has retry_count field
        if hasattr(event, "retry_count"):
            return event.retry_count

        # Otherwise just check if it was processed
        return 1 if event.status == StripeWebhookEventStatus.PROCESSED else 0


async def simulate_webhook_failure(event_id: str) -> bool:
    """
    Simulate a webhook processing failure for testing retry logic.

    Args:
        event_id: The Stripe event ID

    Returns:
        True if successfully simulated failure, False otherwise
    """
    event = await StripeWebhookEvent.get_or_none(event_id=event_id)
    if not event:
        logger.error("Webhook event %s not found", event_id)
        return False

    # Mark as failed
    event.status = StripeWebhookEventStatus.FAILED
    event.error_message = "Simulated failure for testing"
    await event.save()

    logger.info("Simulated failure for webhook event %s", event_id)
    return True
