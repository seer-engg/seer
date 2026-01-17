"""Dedicated controller for processing Stripe webhook events."""
from __future__ import annotations

from typing import Any, Optional, Union

import stripe

from seer.api.subscriptions import stripe_service
from seer.api.subscriptions.clerk_sync import sync_stripe_customer_to_clerk
from seer.config import config
from seer.database.subscription_models import BillingProfile, BillingSubscription
from seer.logger import get_logger

logger = get_logger("api.subscriptions.stripe_webhook_controller")


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
            return

        if event_type in {
            "invoice.payment_failed",
            "invoice.payment_succeeded",
            "invoice.paid",
        }:
            await self._handle_invoice_event(event_type, data)
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

        if event_type == "invoice.payment_failed":
            logger.warning("Invoice payment failed for customer %s", invoice.get("customer"))

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


stripe_webhook_controller = StripeWebhookController()


__all__ = ["stripe_webhook_controller", "StripeWebhookController"]
