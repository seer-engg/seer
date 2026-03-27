"""
Unit tests for worker.tasks.stripe module.

Tests:
- Idempotency (already-processed events are skipped)
- Status transitions (PROCESSING → PROCESSED / FAILED)
- Attempt counting and max attempts
- Error persistence
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database.subscription_models import StripeWebhookEventStatus
from seer.worker.tasks.stripe import MAX_ATTEMPTS


# =============================================================================
# Process Stripe Webhook Event
# =============================================================================


@pytest.mark.unit
class TestProcessStripeWebhookEvent:
    """Tests for process_stripe_webhook_event task."""

    @pytest.mark.asyncio
    async def test_missing_event_returns_early(self):
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        with patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=None)

            # Should not raise
            await process_stripe_webhook_event(event_db_id=999)

    @pytest.mark.asyncio
    async def test_already_processed_event_skipped(self):
        """Idempotency: processing same event twice should be safe."""
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        event = MagicMock()
        event.status = StripeWebhookEventStatus.PROCESSED
        event.event_id = "evt_123"
        event.save = AsyncMock()

        with patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=event)

            await process_stripe_webhook_event(event_db_id=1)

        # Should not call save (no state change)
        event.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_processing(self):
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        event = MagicMock()
        event.status = StripeWebhookEventStatus.RECEIVED
        event.event_id = "evt_123"
        event.attempts = 0
        event.payload = {"type": "invoice.paid", "data": {"object": {"id": "inv_1"}}}
        event.save = AsyncMock()

        with (
            patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model,
            patch("seer.worker.tasks.stripe.stripe_webhook_controller") as mock_controller,
        ):
            mock_model.get_or_none = AsyncMock(return_value=event)
            mock_controller.process_event = AsyncMock()

            await process_stripe_webhook_event(event_db_id=1)

        assert event.status == StripeWebhookEventStatus.PROCESSED
        assert event.last_error is None
        mock_controller.process_event.assert_called_once_with(
            "invoice.paid", {"id": "inv_1"}, event_id="evt_123",
        )

    @pytest.mark.asyncio
    async def test_failure_increments_attempt_count(self):
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        event = MagicMock()
        event.status = StripeWebhookEventStatus.RECEIVED
        event.event_id = "evt_123"
        event.attempts = 0
        event.payload = {"type": "invoice.paid", "data": {"object": {}}}
        event.save = AsyncMock()

        with (
            patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model,
            patch("seer.worker.tasks.stripe.stripe_webhook_controller") as mock_controller,
        ):
            mock_model.get_or_none = AsyncMock(return_value=event)
            mock_controller.process_event = AsyncMock(side_effect=RuntimeError("Stripe API error"))

            await process_stripe_webhook_event(event_db_id=1)

        assert event.status == StripeWebhookEventStatus.FAILED
        assert event.attempts == 1
        assert "Stripe API error" in event.last_error

    @pytest.mark.asyncio
    async def test_max_attempts_reached_logs_error(self):
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        event = MagicMock()
        event.status = StripeWebhookEventStatus.FAILED
        event.event_id = "evt_123"
        event.attempts = MAX_ATTEMPTS - 1  # Will become MAX_ATTEMPTS after increment
        event.payload = {"type": "invoice.paid", "data": {"object": {}}}
        event.save = AsyncMock()

        with (
            patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model,
            patch("seer.worker.tasks.stripe.stripe_webhook_controller") as mock_controller,
            patch("seer.worker.tasks.stripe.logger") as mock_logger,
        ):
            mock_model.get_or_none = AsyncMock(return_value=event)
            mock_controller.process_event = AsyncMock(side_effect=RuntimeError("fail"))

            await process_stripe_webhook_event(event_db_id=1)

        # Should use logger.error (not warning) for max attempts
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_none_payload_handled(self):
        from seer.worker.tasks.stripe import process_stripe_webhook_event

        event = MagicMock()
        event.status = StripeWebhookEventStatus.RECEIVED
        event.event_id = "evt_123"
        event.attempts = 0
        event.payload = None
        event.save = AsyncMock()

        with (
            patch("seer.worker.tasks.stripe.StripeWebhookEvent") as mock_model,
            patch("seer.worker.tasks.stripe.stripe_webhook_controller") as mock_controller,
        ):
            mock_model.get_or_none = AsyncMock(return_value=event)
            mock_controller.process_event = AsyncMock()

            await process_stripe_webhook_event(event_db_id=1)

        # Should handle None payload gracefully
        mock_controller.process_event.assert_called_once_with(
            None, {}, event_id="evt_123",
        )


@pytest.mark.unit
class TestStripeConstants:
    """Tests for stripe task constants."""

    def test_max_attempts_is_reasonable(self):
        assert 3 <= MAX_ATTEMPTS <= 10
