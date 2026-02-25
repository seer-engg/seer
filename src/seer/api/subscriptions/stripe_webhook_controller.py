"""Dedicated controller for processing Stripe webhook events."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union

import stripe

from seer.api.subscriptions import stripe_service
from seer.api.subscriptions.clerk_sync import sync_stripe_customer_to_clerk
from seer.config import config
from seer.database.overage_models import OverageSettings
from seer.database.subscription_models import BillingProfile, SubscriptionTier
from seer.logger import get_logger

logger = get_logger("api.subscriptions.stripe_webhook_controller")

# Tiers eligible for overage pricing
OVERAGE_ELIGIBLE_TIERS = {SubscriptionTier.PRO, SubscriptionTier.PRO_PLUS}


class StripeWebhookController:
    """Centralized Stripe webhook dispatcher with robust subscription resolution."""

    def __init__(self) -> None:
        if config.stripe_secret_key:
            stripe.api_key = stripe.api_key or config.stripe_secret_key

    async def process_event(self, event_type: str | None, data: dict, *, event_id: str | None = None) -> None:
        """Route Stripe webhook events to handlers."""
        if not event_type:
            logger.warning("Stripe event missing type; skipping")
            return

        logger.info(
            "Processing Stripe webhook%s: %s",
            f" {event_id}" if event_id else "",
            event_type,
        )

        if event_type == "checkout.session.completed":
            await self._handle_checkout_session_completed(data)
            return

        if event_type in ("customer.subscription.created", "customer.subscription.updated"):
            await stripe_service.sync_subscription_from_stripe(data)
            return

        if event_type == "customer.subscription.deleted":
            await stripe_service.handle_subscription_deleted(data)
            # Clean up overage settings
            customer_id = data.get("customer")
            if customer_id:
                await self._handle_subscription_deleted_overage(customer_id)
            return

        if event_type in {
            "invoice.payment_failed",
            "invoice.payment_succeeded",
            "invoice.paid",
        }:
            await self._handle_invoice_event(event_type, data)
            return

        if event_type == "setup_intent.succeeded":
            await self._handle_setup_intent_succeeded(data)
            return

        logger.info("Not consuming Stripe event %s", event_type)

    async def _handle_checkout_session_completed(self, data: dict) -> None:
        customer_id = data.get("customer")
        user_id = data.get("metadata", {}).get("user_id")
        if customer_id and user_id:
            await sync_stripe_customer_to_clerk(user_id, customer_id)

        subscription_id = data.get("subscription")
        if subscription_id:
            await stripe_service.sync_subscription_from_stripe(subscription_id)

    async def _handle_invoice_event(self, event_type: str, invoice: dict) -> None:
        subscription_source = await self._resolve_subscription_for_invoice(invoice)
        if subscription_source:
            await stripe_service.sync_subscription_from_stripe(subscription_source)
        else:
            logger.warning(
                "Unable to resolve subscription for invoice %s (customer=%s)",
                invoice.get("id"),
                invoice.get("customer"),
            )

        if event_type == "invoice.paid":
            # Reset overage counter for new billing period
            await self._handle_invoice_paid(invoice)
        elif event_type == "invoice.payment_failed":
            logger.warning("Invoice payment failed for customer %s", invoice.get("customer"))
            # Check if this is related to overage charges
            await self._handle_overage_payment_failure(invoice)

    async def _resolve_subscription_for_invoice(
        self, invoice: dict
    ) -> Optional[Union[str, stripe.Subscription]]:
        # Prefer explicit subscription id on the invoice; otherwise fetch latest for the customer from Stripe.
        subscription_id = invoice.get("subscription")
        if subscription_id:
            return subscription_id

        customer_id = invoice.get("customer")
        if not customer_id:
            return None

        return await self._fetch_latest_stripe_subscription(customer_id)

    async def _fetch_latest_stripe_subscription(self, customer_id: str) -> Optional[stripe.Subscription]:
        if not config.stripe_secret_key:
            return None

        stripe.api_key = stripe.api_key or config.stripe_secret_key

        try:
            response = stripe.Subscription.list(
                customer=customer_id,
                status="all",
                limit=1,
                expand=["data.items.data.price"],
            )
        except stripe.error.StripeError as exc:  # type: ignore[attr-defined]
            logger.error("Failed to list Stripe subscriptions for customer %s: %s", customer_id, exc)
            return None

        subscriptions = response.get("data") if isinstance(response, dict) else None
        if not subscriptions:
            return None

        return subscriptions[0]

    async def _handle_setup_intent_succeeded(self, data: dict) -> None:
        """
        Handle successful Setup Intent - update BillingProfile with payment method status.

        Called when a user successfully adds a payment method during onboarding.
        Updates the has_payment_method flag to allow app access.
        """
        setup_intent = data.get("object") if "object" in data else data
        customer_id = setup_intent.get("customer")

        if not customer_id:
            logger.warning("Setup Intent succeeded event missing customer ID")
            return

        billing_profile = await BillingProfile.get_or_none(stripe_customer_id=customer_id)
        if not billing_profile:
            logger.warning(
                "No billing profile found for Stripe customer %s on Setup Intent success",
                customer_id
            )
            return

        billing_profile.has_payment_method = True
        billing_profile.payment_method_added_at = datetime.now(timezone.utc)
        await billing_profile.save(update_fields=["has_payment_method", "payment_method_added_at"])

        logger.info(
            "Updated payment method status for customer %s (Setup Intent: %s)",
            customer_id,
            setup_intent.get("id")
        )

    async def _handle_invoice_paid(self, invoice: dict) -> None:
        """
        Handle invoice.paid event - reset overage counter for new billing period.

        Args:
            invoice: The Stripe invoice object.
        """
        customer_id = invoice.get("customer")
        if not customer_id:
            return

        billing_profile = await BillingProfile.get_or_none(stripe_customer_id=customer_id)
        if not billing_profile:
            return

        overage_settings = await OverageSettings.get_or_none(billing_profile=billing_profile)
        if not overage_settings or not overage_settings.enabled:
            return

        # Get the new period start from subscription
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return

        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            period_start = subscription.get("current_period_start")
            if period_start:
                # pylint: disable=import-outside-toplevel  # Avoid circular import
                from seer.api.subscriptions.overage_service import reset_period_overage

                period_start_dt = datetime.fromtimestamp(period_start, tz=timezone.utc)
                await reset_period_overage(overage_settings, period_start_dt)

                logger.info(
                    "Reset overage counter for customer %s, new period starts %s",
                    customer_id,
                    period_start_dt.isoformat(),
                )
        except stripe.error.StripeError as exc:
            logger.error(
                "Failed to retrieve subscription %s for overage reset: %s",
                subscription_id,
                exc,
            )

    async def _handle_overage_payment_failure(self, invoice: dict) -> None:
        """
        Handle payment failure that might be related to overage charges.

        If the invoice contains overage line items and payment fails,
        we may need to disable overage for the user.

        Args:
            invoice: The Stripe invoice object.
        """
        customer_id = invoice.get("customer")
        if not customer_id:
            return

        # Check if this invoice has overage-related line items
        lines = invoice.get("lines", {}).get("data", [])
        has_overage_line = any(
            line.get("price", {}).get("lookup_key") == "llm_overage_metered"
            or line.get("metadata", {}).get("purpose") == "llm_overage"
            for line in lines
        )

        if not has_overage_line:
            return

        billing_profile = await BillingProfile.get_or_none(stripe_customer_id=customer_id)
        if not billing_profile:
            return

        overage_settings = await OverageSettings.get_or_none(billing_profile=billing_profile)
        if not overage_settings or not overage_settings.enabled:
            return

        # Disable overage to prevent further charges
        overage_settings.enabled = False
        await overage_settings.save(update_fields=["enabled", "updated_at"])

        logger.warning(
            "Disabled overage for customer %s due to payment failure on invoice %s",
            customer_id,
            invoice.get("id"),
        )
        # TODO: Send notification to user about overage being disabled

    async def _handle_subscription_deleted_overage(self, customer_id: str) -> None:
        """
        Handle subscription deletion - clean up overage settings.

        Args:
            customer_id: The Stripe customer ID.
        """
        billing_profile = await BillingProfile.get_or_none(stripe_customer_id=customer_id)
        if not billing_profile:
            return

        overage_settings = await OverageSettings.get_or_none(billing_profile=billing_profile)
        if not overage_settings:
            return

        # Disable overage and clear the subscription item ID
        if overage_settings.enabled:
            overage_settings.enabled = False
            overage_settings.stripe_metered_subscription_item_id = None
            await overage_settings.save(
                update_fields=["enabled", "stripe_metered_subscription_item_id", "updated_at"]
            )

            logger.info(
                "Cleaned up overage settings for customer %s after subscription deletion",
                customer_id,
            )


stripe_webhook_controller = StripeWebhookController()


__all__ = ["stripe_webhook_controller", "StripeWebhookController"]
