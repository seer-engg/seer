"""
Unit tests for Stripe webhook controller.

Tests:
- process_event: Event routing
- _handle_checkout_session_completed: Checkout flow
- _handle_invoice_event: Invoice handling
- _handle_setup_intent_succeeded: Payment method setup
- _resolve_subscription_for_invoice: Subscription resolution
- _fetch_latest_stripe_subscription: Stripe API interaction
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def webhook_controller():
    """Create a StripeWebhookController instance with mocked config."""
    with patch("seer.api.subscriptions.stripe_webhook_controller.config") as mock_config:
        mock_config.stripe_secret_key = "sk_test_123"
        from seer.api.subscriptions.stripe_webhook_controller import StripeWebhookController
        yield StripeWebhookController()


@pytest.fixture
def mock_checkout_session_event():
    """Mock Stripe checkout.session.completed event data."""
    return {
        "customer": "cus_123",
        "subscription": "sub_123",
        "mode": "subscription",
        "metadata": {
            "user_id": "user_abc",
            "is_early_adopter": "true",
        },
    }


@pytest.fixture
def mock_invoice_event():
    """Mock Stripe invoice event data."""
    return {
        "id": "in_123",
        "customer": "cus_123",
        "subscription": "sub_123",
        "amount_paid": 2999,
        "currency": "usd",
    }


@pytest.fixture
def mock_setup_intent_event():
    """Mock Stripe setup_intent.succeeded event data."""
    return {
        "id": "seti_123",
        "customer": "cus_123",
        "payment_method": "pm_123",
        "metadata": {
            "billing_profile_id": "1",
        },
    }


@pytest.fixture
def mock_subscription_event():
    """Mock Stripe subscription event data."""
    return {
        "id": "sub_123",
        "customer": "cus_123",
        "status": "active",
        "items": {
            "data": [
                {"price": {"id": "price_123"}}
            ]
        },
        "current_period_start": 1704067200,
        "current_period_end": 1706745600,
    }


# =============================================================================
# Event Routing Tests (process_event)
# =============================================================================


@pytest.mark.unit
class TestProcessEvent:
    """Tests for process_event method."""

    @pytest.mark.asyncio
    async def test_process_event_missing_event_type(self, webhook_controller):
        """Test that missing event type is handled gracefully."""
        with patch.object(webhook_controller, "_handle_checkout_session_completed") as mock_handler:
            await webhook_controller.process_event(None, {})
            mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_routes_checkout_session_completed(
        self, webhook_controller, mock_checkout_session_event
    ):
        """Test routing checkout.session.completed to handler."""
        with patch.object(
            webhook_controller, "_handle_checkout_session_completed", new_callable=AsyncMock
        ) as mock_handler:
            await webhook_controller.process_event(
                "checkout.session.completed",
                mock_checkout_session_event
            )
            mock_handler.assert_called_once_with(mock_checkout_session_event)

    @pytest.mark.asyncio
    async def test_process_event_routes_subscription_created(
        self, webhook_controller, mock_subscription_event
    ):
        """Test routing customer.subscription.created to sync."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()
            await webhook_controller.process_event(
                "customer.subscription.created",
                mock_subscription_event
            )
            mock_service.sync_subscription_from_stripe.assert_called_once_with(
                mock_subscription_event
            )

    @pytest.mark.asyncio
    async def test_process_event_routes_subscription_updated(
        self, webhook_controller, mock_subscription_event
    ):
        """Test routing customer.subscription.updated to sync."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()
            await webhook_controller.process_event(
                "customer.subscription.updated",
                mock_subscription_event
            )
            mock_service.sync_subscription_from_stripe.assert_called_once_with(
                mock_subscription_event
            )

    @pytest.mark.asyncio
    async def test_process_event_routes_subscription_deleted(
        self, webhook_controller, mock_subscription_event
    ):
        """Test routing customer.subscription.deleted to handler."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.handle_subscription_deleted = AsyncMock()
            await webhook_controller.process_event(
                "customer.subscription.deleted",
                mock_subscription_event
            )
            mock_service.handle_subscription_deleted.assert_called_once_with(
                mock_subscription_event
            )

    @pytest.mark.asyncio
    async def test_process_event_routes_invoice_paid(
        self, webhook_controller, mock_invoice_event
    ):
        """Test routing invoice.paid to handler."""
        with patch.object(
            webhook_controller, "_handle_invoice_event", new_callable=AsyncMock
        ) as mock_handler:
            await webhook_controller.process_event("invoice.paid", mock_invoice_event)
            mock_handler.assert_called_once_with("invoice.paid", mock_invoice_event)

    @pytest.mark.asyncio
    async def test_process_event_routes_invoice_payment_failed(
        self, webhook_controller, mock_invoice_event
    ):
        """Test routing invoice.payment_failed to handler."""
        with patch.object(
            webhook_controller, "_handle_invoice_event", new_callable=AsyncMock
        ) as mock_handler:
            await webhook_controller.process_event("invoice.payment_failed", mock_invoice_event)
            mock_handler.assert_called_once_with("invoice.payment_failed", mock_invoice_event)

    @pytest.mark.asyncio
    async def test_process_event_routes_invoice_payment_succeeded(
        self, webhook_controller, mock_invoice_event
    ):
        """Test routing invoice.payment_succeeded to handler."""
        with patch.object(
            webhook_controller, "_handle_invoice_event", new_callable=AsyncMock
        ) as mock_handler:
            await webhook_controller.process_event("invoice.payment_succeeded", mock_invoice_event)
            mock_handler.assert_called_once_with("invoice.payment_succeeded", mock_invoice_event)

    @pytest.mark.asyncio
    async def test_process_event_routes_setup_intent_succeeded(
        self, webhook_controller, mock_setup_intent_event
    ):
        """Test routing setup_intent.succeeded to handler."""
        with patch.object(
            webhook_controller, "_handle_setup_intent_succeeded", new_callable=AsyncMock
        ) as mock_handler:
            await webhook_controller.process_event(
                "setup_intent.succeeded",
                mock_setup_intent_event
            )
            mock_handler.assert_called_once_with(mock_setup_intent_event)

    @pytest.mark.asyncio
    async def test_process_event_ignores_unsupported_events(self, webhook_controller):
        """Test that unsupported events are ignored."""
        # Should not raise and should not call any handlers
        await webhook_controller.process_event("charge.succeeded", {"id": "ch_123"})
        await webhook_controller.process_event("payment_intent.created", {"id": "pi_123"})
        await webhook_controller.process_event("customer.created", {"id": "cus_123"})

    @pytest.mark.asyncio
    async def test_process_event_includes_event_id_in_log(self, webhook_controller):
        """Test that event_id is included in logging."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.logger"
        ) as mock_logger:
            await webhook_controller.process_event(
                "charge.succeeded",
                {},
                event_id="evt_123"
            )
            mock_logger.info.assert_called()
            # Check the first call contains the event_id (Processing webhook message)
            all_calls = mock_logger.info.call_args_list
            first_call_args = str(all_calls[0])
            assert "evt_123" in first_call_args


# =============================================================================
# Checkout Session Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandleCheckoutSessionCompleted:
    """Tests for _handle_checkout_session_completed method."""

    @pytest.mark.asyncio
    async def test_syncs_customer_to_clerk_when_both_ids_present(
        self, webhook_controller, mock_checkout_session_event
    ):
        """Test Clerk sync when customer_id and user_id are present."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ) as mock_sync_clerk, patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(
                mock_checkout_session_event
            )

            mock_sync_clerk.assert_called_once_with("user_abc", "cus_123")

    @pytest.mark.asyncio
    async def test_does_not_sync_clerk_without_user_id(self, webhook_controller):
        """Test no Clerk sync when user_id is missing."""
        event_data = {
            "customer": "cus_123",
            "subscription": "sub_123",
            "metadata": {},
        }
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ) as mock_sync_clerk, patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(event_data)

            mock_sync_clerk.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_sync_clerk_without_customer_id(self, webhook_controller):
        """Test no Clerk sync when customer_id is missing."""
        event_data = {
            "subscription": "sub_123",
            "metadata": {"user_id": "user_abc"},
        }
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ) as mock_sync_clerk, patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(event_data)

            mock_sync_clerk.assert_not_called()

    @pytest.mark.asyncio
    async def test_syncs_subscription_when_subscription_id_present(
        self, webhook_controller, mock_checkout_session_event
    ):
        """Test subscription sync when subscription_id is present."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(
                mock_checkout_session_event
            )

            mock_service.sync_subscription_from_stripe.assert_called_once_with(
                "sub_123",
                is_early_adopter=True
            )

    @pytest.mark.asyncio
    async def test_early_adopter_flag_false_when_not_true(self, webhook_controller):
        """Test early_adopter is False when metadata value is not 'true'."""
        event_data = {
            "customer": "cus_123",
            "subscription": "sub_123",
            "metadata": {"is_early_adopter": "false"},
        }
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(event_data)

            mock_service.sync_subscription_from_stripe.assert_called_once_with(
                "sub_123",
                is_early_adopter=False
            )

    @pytest.mark.asyncio
    async def test_does_not_sync_subscription_without_subscription_id(self, webhook_controller):
        """Test no subscription sync when subscription_id is missing."""
        event_data = {
            "customer": "cus_123",
            "metadata": {},
        }
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_checkout_session_completed(event_data)

            mock_service.sync_subscription_from_stripe.assert_not_called()


# =============================================================================
# Invoice Event Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandleInvoiceEvent:
    """Tests for _handle_invoice_event method."""

    @pytest.mark.asyncio
    async def test_syncs_subscription_when_resolved(
        self, webhook_controller, mock_invoice_event
    ):
        """Test subscription sync when subscription can be resolved."""
        with patch.object(
            webhook_controller,
            "_resolve_subscription_for_invoice",
            new_callable=AsyncMock,
            return_value="sub_123"
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_invoice_event("invoice.paid", mock_invoice_event)

            mock_service.sync_subscription_from_stripe.assert_called_once_with("sub_123")

    @pytest.mark.asyncio
    async def test_logs_warning_when_subscription_not_resolved(
        self, webhook_controller, mock_invoice_event
    ):
        """Test warning logged when subscription cannot be resolved."""
        with patch.object(
            webhook_controller,
            "_resolve_subscription_for_invoice",
            new_callable=AsyncMock,
            return_value=None
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service, patch(
            "seer.api.subscriptions.stripe_webhook_controller.logger"
        ) as mock_logger:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_invoice_event("invoice.paid", mock_invoice_event)

            mock_service.sync_subscription_from_stripe.assert_not_called()
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "Unable to resolve subscription" in call_args

    @pytest.mark.asyncio
    async def test_logs_warning_for_payment_failed(
        self, webhook_controller, mock_invoice_event
    ):
        """Test warning logged for payment_failed events."""
        with patch.object(
            webhook_controller,
            "_resolve_subscription_for_invoice",
            new_callable=AsyncMock,
            return_value="sub_123"
        ), patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service, patch(
            "seer.api.subscriptions.stripe_webhook_controller.logger"
        ) as mock_logger:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller._handle_invoice_event(
                "invoice.payment_failed",
                mock_invoice_event
            )

            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "payment failed" in call_args


# =============================================================================
# Subscription Resolution Tests
# =============================================================================


@pytest.mark.unit
class TestResolveSubscriptionForInvoice:
    """Tests for _resolve_subscription_for_invoice method."""

    @pytest.mark.asyncio
    async def test_returns_subscription_id_from_invoice(self, webhook_controller):
        """Test returns subscription_id when present on invoice."""
        invoice = {"subscription": "sub_123", "customer": "cus_123"}

        result = await webhook_controller._resolve_subscription_for_invoice(invoice)

        assert result == "sub_123"

    @pytest.mark.asyncio
    async def test_fetches_latest_subscription_when_no_subscription_id(
        self, webhook_controller
    ):
        """Test falls back to fetching latest subscription."""
        invoice = {"customer": "cus_123"}
        mock_subscription = MagicMock()

        with patch.object(
            webhook_controller,
            "_fetch_latest_stripe_subscription",
            new_callable=AsyncMock,
            return_value=mock_subscription
        ) as mock_fetch:
            result = await webhook_controller._resolve_subscription_for_invoice(invoice)

            mock_fetch.assert_called_once_with("cus_123")
            assert result == mock_subscription

    @pytest.mark.asyncio
    async def test_returns_none_when_no_customer_id(self, webhook_controller):
        """Test returns None when no customer_id on invoice."""
        invoice = {"id": "in_123"}

        result = await webhook_controller._resolve_subscription_for_invoice(invoice)

        assert result is None


# =============================================================================
# Fetch Latest Stripe Subscription Tests
# =============================================================================


@pytest.mark.unit
class TestFetchLatestStripeSubscription:
    """Tests for _fetch_latest_stripe_subscription method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_stripe_key(self):
        """Test returns None when Stripe key is not configured."""
        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.config"
        ) as mock_config:
            mock_config.stripe_secret_key = None
            from seer.api.subscriptions.stripe_webhook_controller import StripeWebhookController
            controller = StripeWebhookController()

            result = await controller._fetch_latest_stripe_subscription("cus_123")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_subscription_from_stripe_api(self, webhook_controller):
        """Test returns subscription from Stripe API."""
        mock_subscription = {"id": "sub_123", "status": "active"}
        mock_response = {"data": [mock_subscription]}

        with patch("stripe.Subscription.list", return_value=mock_response):
            result = await webhook_controller._fetch_latest_stripe_subscription("cus_123")

            assert result == mock_subscription

    @pytest.mark.asyncio
    async def test_returns_none_on_stripe_error(self, webhook_controller):
        """Test returns None on Stripe API error."""
        import stripe

        with patch("stripe.Subscription.list") as mock_list:
            mock_list.side_effect = stripe.error.StripeError("API error")  # type: ignore[attr-defined]

            result = await webhook_controller._fetch_latest_stripe_subscription("cus_123")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_subscriptions(self, webhook_controller):
        """Test returns None when customer has no subscriptions."""
        mock_response = {"data": []}

        with patch("stripe.Subscription.list", return_value=mock_response):
            result = await webhook_controller._fetch_latest_stripe_subscription("cus_123")

            assert result is None

    @pytest.mark.asyncio
    async def test_calls_stripe_api_with_correct_params(self, webhook_controller):
        """Test Stripe API called with correct parameters."""
        mock_response = {"data": []}

        with patch("stripe.Subscription.list", return_value=mock_response) as mock_list:
            await webhook_controller._fetch_latest_stripe_subscription("cus_123")

            mock_list.assert_called_once_with(
                customer="cus_123",
                status="all",
                limit=1,
                expand=["data.items.data.price"],
            )


# =============================================================================
# Setup Intent Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandleSetupIntentSucceeded:
    """Tests for _handle_setup_intent_succeeded method."""

    @pytest.mark.asyncio
    async def test_updates_billing_profile_payment_method(self, webhook_controller):
        """Test updates billing profile has_payment_method flag."""
        setup_intent_data = {
            "id": "seti_123",
            "customer": "cus_123",
            "payment_method": "pm_123",
        }
        mock_profile = MagicMock()
        mock_profile.has_payment_method = False
        mock_profile.save = AsyncMock()

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=mock_profile)

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

            mock_billing.get_or_none.assert_called_once_with(stripe_customer_id="cus_123")
            assert mock_profile.has_payment_method is True
            assert mock_profile.payment_method_added_at is not None
            mock_profile.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_data_with_object_wrapper(self, webhook_controller):
        """Test handles setup intent data wrapped in 'object' key."""
        setup_intent_data = {
            "object": {
                "id": "seti_123",
                "customer": "cus_123",
                "payment_method": "pm_123",
            }
        }
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=mock_profile)

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

            mock_billing.get_or_none.assert_called_once_with(stripe_customer_id="cus_123")

    @pytest.mark.asyncio
    async def test_logs_warning_when_customer_id_missing(self, webhook_controller):
        """Test logs warning when customer_id is missing."""
        setup_intent_data = {"id": "seti_123", "payment_method": "pm_123"}

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.logger"
        ) as mock_logger, patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock()

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "missing customer ID" in call_args
            mock_billing.get_or_none.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_warning_when_billing_profile_not_found(self, webhook_controller):
        """Test logs warning when billing profile is not found."""
        setup_intent_data = {
            "id": "seti_123",
            "customer": "cus_123",
            "payment_method": "pm_123",
        }

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.logger"
        ) as mock_logger, patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=None)

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "No billing profile found" in call_args

    @pytest.mark.asyncio
    async def test_saves_with_correct_update_fields(self, webhook_controller):
        """Test billing profile saved with correct update_fields."""
        setup_intent_data = {
            "id": "seti_123",
            "customer": "cus_123",
            "payment_method": "pm_123",
        }
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=mock_profile)

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

            mock_profile.save.assert_called_once_with(
                update_fields=["has_payment_method", "payment_method_added_at"]
            )

    @pytest.mark.asyncio
    async def test_sets_payment_method_added_at_to_utc_now(self, webhook_controller):
        """Test payment_method_added_at is set to current UTC time."""
        setup_intent_data = {
            "id": "seti_123",
            "customer": "cus_123",
            "payment_method": "pm_123",
        }
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()

        before_call = datetime.now(timezone.utc)

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=mock_profile)

            await webhook_controller._handle_setup_intent_succeeded(setup_intent_data)

        after_call = datetime.now(timezone.utc)

        assert mock_profile.payment_method_added_at >= before_call
        assert mock_profile.payment_method_added_at <= after_call
        assert mock_profile.payment_method_added_at.tzinfo == timezone.utc


# =============================================================================
# Integration Tests (Full Event Flow)
# =============================================================================


@pytest.mark.unit
class TestWebhookEventFlow:
    """Integration tests for complete webhook event flows."""

    @pytest.mark.asyncio
    async def test_checkout_session_completed_full_flow(self, webhook_controller):
        """Test full checkout.session.completed flow."""
        event_data = {
            "customer": "cus_123",
            "subscription": "sub_123",
            "metadata": {
                "user_id": "user_abc",
                "is_early_adopter": "true",
            },
        }

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.sync_stripe_customer_to_clerk",
            new_callable=AsyncMock
        ) as mock_clerk, patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller.process_event(
                "checkout.session.completed",
                event_data,
                event_id="evt_123"
            )

            mock_clerk.assert_called_once_with("user_abc", "cus_123")
            mock_service.sync_subscription_from_stripe.assert_called_once_with(
                "sub_123",
                is_early_adopter=True
            )

    @pytest.mark.asyncio
    async def test_invoice_paid_full_flow(self, webhook_controller):
        """Test full invoice.paid flow."""
        invoice_data = {
            "id": "in_123",
            "customer": "cus_123",
            "subscription": "sub_123",
            "amount_paid": 2999,
        }

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.stripe_service"
        ) as mock_service:
            mock_service.sync_subscription_from_stripe = AsyncMock()

            await webhook_controller.process_event(
                "invoice.paid",
                invoice_data,
                event_id="evt_456"
            )

            mock_service.sync_subscription_from_stripe.assert_called_once_with("sub_123")

    @pytest.mark.asyncio
    async def test_setup_intent_succeeded_full_flow(self, webhook_controller):
        """Test full setup_intent.succeeded flow."""
        setup_intent_data = {
            "id": "seti_123",
            "customer": "cus_123",
            "payment_method": "pm_123",
        }
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()

        with patch(
            "seer.api.subscriptions.stripe_webhook_controller.BillingProfile"
        ) as mock_billing:
            mock_billing.get_or_none = AsyncMock(return_value=mock_profile)

            await webhook_controller.process_event(
                "setup_intent.succeeded",
                setup_intent_data,
                event_id="evt_789"
            )

            mock_billing.get_or_none.assert_called_once_with(stripe_customer_id="cus_123")
            assert mock_profile.has_payment_method is True
            mock_profile.save.assert_called_once()
