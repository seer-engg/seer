# pylint: disable=broad-exception-caught
# Reason: Stripe operations require broad error handling
"""
Overage service layer for usage-based pricing.

Handles Stripe metered subscription items, usage record reporting,
and overage settings management.
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import stripe

from seer.api.subscriptions.pricing_catalog import get_overage_metered_price_id
from seer.config import config
from seer.database.overage_models import (
    OverageRecordStatus,
    OverageSettings,
    OverageUsageRecord,
)
from seer.database.organization_models import Organization
from seer.database.subscription_models import (
    BillingSubscription,
    StripeCustomer,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.database.usage_models import LLMUsageRecord
from seer.logger import get_logger
from seer.observability.constants import tiered_usage_limits

logger = get_logger("api.subscriptions.overage_service")

# Tiers eligible for overage pricing
OVERAGE_ELIGIBLE_TIERS = {SubscriptionTier.PRO, SubscriptionTier.PRO_PLUS}

# Stripe Billing Meter event name (must match meter configured in Stripe Dashboard)
OVERAGE_METER_EVENT_NAME = "llm_overage_usage"


def _get_overage_price_id() -> str:
    """
    Get the Stripe metered price ID for overage billing.

    Fetches dynamically from Stripe via pricing catalog cache.

    Returns:
        Stripe price ID for LLM credit overages.

    Raises:
        ValueError: If no overage metered price is configured in Stripe.
    """
    price_id = get_overage_metered_price_id()
    if not price_id:
        raise ValueError(
            "No overage metered price found in Stripe. "
            "Create a metered price with lookup_key containing 'overage' or metadata.type='overage'"
        )
    return price_id


async def get_or_create_overage_settings(organization: Organization) -> OverageSettings:
    """
    Get or create overage settings for an organization.

    Args:
        organization: The organization to get settings for.

    Returns:
        OverageSettings instance.
    """
    settings, _ = await OverageSettings.get_or_create(
        organization=organization,
        defaults={
            "spending_cap_cents": tiered_usage_limits.OVERAGE_DEFAULT_CAP_CENTS,
            "margin_multiplier": Decimal(str(tiered_usage_limits.OVERAGE_DEFAULT_MARGIN_MULTIPLIER)),
        },
    )
    return settings


async def is_overage_eligible(subscription: BillingSubscription) -> bool:
    """
    Check if a subscription is eligible for overage pricing.

    Requires:
    - Paid tier (PRO or PRO_PLUS)
    - Active subscription status
    - Payment method on file

    Args:
        subscription: The billing subscription to check.

    Returns:
        True if eligible for overage pricing.
    """
    if subscription.tier not in OVERAGE_ELIGIBLE_TIERS:
        return False

    if subscription.status not in {SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING}:
        return False

    # Check payment method on organization
    await subscription.fetch_related("organization")
    if not subscription.organization.has_payment_method:
        return False

    return True


async def attach_overage_pricing(subscription: BillingSubscription) -> Optional[str]:
    """
    Add metered overage price as a subscription item.

    Args:
        subscription: The billing subscription to add overage pricing to.

    Returns:
        Stripe subscription item ID if successful, None otherwise.
    """
    if not config.stripe_secret_key:
        logger.warning("Stripe not configured, cannot attach overage pricing")
        return None

    stripe.api_key = config.stripe_secret_key

    if not subscription.stripe_subscription_id:
        logger.warning("No Stripe subscription ID for subscription %s", subscription.id)
        return None

    try:
        price_id = _get_overage_price_id()
    except ValueError as exc:
        logger.error("Cannot attach overage pricing: %s", exc)
        return None

    try:
        # Add metered price as a new subscription item
        subscription_item = stripe.SubscriptionItem.create(
            subscription=subscription.stripe_subscription_id,
            price=price_id,
            metadata={"purpose": "llm_overage"},
        )

        logger.info(
            "Added overage subscription item %s to subscription %s",
            subscription_item.id,
            subscription.stripe_subscription_id,
        )

        return subscription_item.id

    except stripe.error.StripeError as exc:
        logger.error(
            "Failed to add overage subscription item: %s",
            exc,
        )
        return None


async def detach_overage_pricing(subscription: BillingSubscription, subscription_item_id: str) -> bool:
    """
    Remove metered overage subscription item.

    Args:
        subscription: The billing subscription.
        subscription_item_id: The Stripe subscription item ID to remove.

    Returns:
        True if successful, False otherwise.
    """
    if not config.stripe_secret_key:
        logger.warning("Stripe not configured, cannot detach overage pricing")
        return False

    stripe.api_key = config.stripe_secret_key

    try:
        # Delete the subscription item (prorate_behavior defaults to create_prorations)
        stripe.SubscriptionItem.delete(
            subscription_item_id,
            proration_behavior="none",  # Don't create credits for unused time
            clear_usage=False,  # Keep usage records for billing
        )

        logger.info(
            "Removed overage subscription item %s from subscription %s",
            subscription_item_id,
            subscription.stripe_subscription_id,
        )

        return True

    except stripe.error.StripeError as exc:
        logger.error(
            "Failed to remove overage subscription item %s: %s",
            subscription_item_id,
            exc,
        )
        return False


async def report_usage_to_stripe(
    overage_settings: OverageSettings,
    llm_record: Optional[LLMUsageRecord],
    base_cost_cents: int,
    billed_amount_cents: int,
) -> Optional[OverageUsageRecord]:
    """
    Report overage usage to Stripe using Billing Meter Events API.

    Args:
        overage_settings: The overage settings for the user.
        llm_record: Optional reference to the LLM usage record.
        base_cost_cents: The actual LLM cost in cents.
        billed_amount_cents: The billed amount (cost × margin) in cents.

    Returns:
        OverageUsageRecord if successful, None otherwise.
    """
    if not config.stripe_secret_key:
        logger.warning("Stripe not configured, cannot report usage")
        return None

    stripe.api_key = config.stripe_secret_key

    # Get the Stripe customer ID from the organization
    await overage_settings.fetch_related("organization")
    organization = overage_settings.organization

    if not organization.stripe_customer_id:
        logger.error(
            "No Stripe customer for overage settings %s (org %s)",
            overage_settings.id,
            organization.id,
        )
        return None

    stripe_customer = await StripeCustomer.get(id=organization.stripe_customer_id)
    stripe_customer_id = stripe_customer.stripe_customer_id

    # Create local record first
    usage_record = await OverageUsageRecord.create(
        overage_settings=overage_settings,
        llm_usage_record=llm_record,
        base_cost_cents=base_cost_cents,
        billed_amount_cents=billed_amount_cents,
        status=OverageRecordStatus.PENDING,
    )

    try:
        # Report to Stripe using Billing Meter Events API
        # Event name must match the meter configured in Stripe Dashboard
        # Note: Omit timestamp to let Stripe use their server time (avoids clock sync issues)
        meter_event = stripe.billing.MeterEvent.create(
            event_name=OVERAGE_METER_EVENT_NAME,
            payload={
                "value": str(billed_amount_cents),
                "stripe_customer_id": stripe_customer_id,
            },
        )

        # Update local record with Stripe response
        usage_record.stripe_usage_record_id = meter_event.identifier
        usage_record.reported_to_stripe_at = datetime.now(timezone.utc)
        usage_record.status = OverageRecordStatus.REPORTED
        await usage_record.save()

        # Update running total in overage settings
        overage_settings.current_period_overage_cents += billed_amount_cents
        await overage_settings.save(update_fields=["current_period_overage_cents", "updated_at"])

        logger.info(
            "Reported overage usage to Stripe meter: %d cents for customer %s",
            billed_amount_cents,
            stripe_customer_id,
        )

        return usage_record

    except stripe.error.StripeError as exc:
        # Mark record as failed
        usage_record.status = OverageRecordStatus.FAILED
        usage_record.error_message = str(exc)
        await usage_record.save()

        logger.error(
            "Failed to report overage usage to Stripe: %s",
            exc,
        )

        return usage_record


async def enable_overage(
    organization: Organization,
    subscription: BillingSubscription,
    spending_cap_cents: Optional[int] = None,
) -> OverageSettings:
    """
    Enable usage-based pricing for an organization.

    Args:
        organization: The organization.
        subscription: The billing subscription.
        spending_cap_cents: Optional custom spending cap.

    Returns:
        Updated OverageSettings.

    Raises:
        ValueError: If not eligible or Stripe attachment fails.
    """
    if not await is_overage_eligible(subscription):
        raise ValueError("Subscription is not eligible for overage pricing")

    settings = await get_or_create_overage_settings(organization)

    # Attach metered pricing to Stripe subscription
    subscription_item_id = await attach_overage_pricing(subscription)
    if not subscription_item_id:
        raise ValueError("Failed to attach overage pricing to Stripe subscription")

    # Update settings
    settings.enabled = True
    settings.enabled_at = datetime.now(timezone.utc)
    settings.stripe_metered_subscription_item_id = subscription_item_id

    if spending_cap_cents is not None:
        # Validate cap range
        min_cap = tiered_usage_limits.OVERAGE_MIN_CAP_CENTS
        max_cap = tiered_usage_limits.OVERAGE_MAX_CAP_CENTS
        settings.spending_cap_cents = max(min_cap, min(max_cap, spending_cap_cents))

    # Align period with subscription
    if subscription.current_period_start:
        settings.current_period_start = subscription.current_period_start

    await settings.save()

    logger.info(
        "Enabled overage for organization %s with cap $%.2f",
        organization.id,
        settings.spending_cap_cents / 100,
    )

    return settings


async def disable_overage(
    organization: Organization,
    subscription: BillingSubscription,
) -> OverageSettings:
    """
    Disable usage-based pricing for an organization.

    Note: Pending charges will still be billed, only new overages are blocked.

    Args:
        organization: The organization.
        subscription: The billing subscription.

    Returns:
        Updated OverageSettings.
    """
    settings = await get_or_create_overage_settings(organization)

    if not settings.enabled:
        return settings

    # Detach metered pricing from Stripe (if attached)
    if settings.stripe_metered_subscription_item_id:
        await detach_overage_pricing(subscription, settings.stripe_metered_subscription_item_id)

    # Update settings
    settings.enabled = False
    settings.stripe_metered_subscription_item_id = None
    await settings.save()

    logger.info(
        "Disabled overage for organization %s (pending charges: %d cents)",
        organization.id,
        settings.current_period_overage_cents,
    )

    return settings


async def update_spending_cap(
    organization: Organization,
    spending_cap_cents: int,
) -> OverageSettings:
    """
    Update the spending cap for an organization.

    Args:
        organization: The organization.
        spending_cap_cents: New spending cap in cents.

    Returns:
        Updated OverageSettings.

    Raises:
        ValueError: If cap is outside allowed range.
    """
    min_cap = tiered_usage_limits.OVERAGE_MIN_CAP_CENTS
    max_cap = tiered_usage_limits.OVERAGE_MAX_CAP_CENTS

    if spending_cap_cents < min_cap or spending_cap_cents > max_cap:
        raise ValueError(
            f"Spending cap must be between ${min_cap / 100:.2f} and ${max_cap / 100:.2f}"
        )

    settings = await get_or_create_overage_settings(organization)
    settings.spending_cap_cents = spending_cap_cents
    await settings.save(update_fields=["spending_cap_cents", "updated_at"])

    logger.info(
        "Updated spending cap for organization %s to $%.2f",
        organization.id,
        spending_cap_cents / 100,
    )

    return settings


async def reset_period_overage(overage_settings: OverageSettings, period_start: datetime) -> None:
    """
    Reset the current period overage counter for a new billing period.

    Called when a new billing period starts (e.g., invoice.paid event).

    Args:
        overage_settings: The overage settings to reset.
        period_start: Start of the new billing period.
    """
    overage_settings.current_period_overage_cents = 0
    overage_settings.current_period_start = period_start
    await overage_settings.save(update_fields=["current_period_overage_cents", "current_period_start", "updated_at"])

    logger.info(
        "Reset overage counter for settings %s, new period starts %s",
        overage_settings.id,
        period_start.isoformat(),
    )


async def get_overage_usage_summary(overage_settings: OverageSettings) -> dict:
    """
    Get a summary of overage usage for the current period.

    Args:
        overage_settings: The overage settings.

    Returns:
        Dictionary with usage summary.
    """
    # Count records by status
    pending_count = await OverageUsageRecord.filter(
        overage_settings=overage_settings,
        status=OverageRecordStatus.PENDING,
    ).count()

    reported_count = await OverageUsageRecord.filter(
        overage_settings=overage_settings,
        status=OverageRecordStatus.REPORTED,
    ).count()

    failed_count = await OverageUsageRecord.filter(
        overage_settings=overage_settings,
        status=OverageRecordStatus.FAILED,
    ).count()

    return {
        "enabled": overage_settings.enabled,
        "spending_cap_cents": overage_settings.spending_cap_cents,
        "spending_cap_dollars": float(overage_settings.spending_cap_dollars),
        "current_usage_cents": overage_settings.current_period_overage_cents,
        "current_usage_dollars": float(overage_settings.current_period_overage_dollars),
        "remaining_cents": overage_settings.remaining_cap_cents,
        "remaining_dollars": overage_settings.remaining_cap_cents / 100,
        "cap_reached": overage_settings.is_cap_reached(),
        "margin_multiplier": float(overage_settings.margin_multiplier),
        "period_start": overage_settings.current_period_start.isoformat() if overage_settings.current_period_start else None,
        "enabled_at": overage_settings.enabled_at.isoformat() if overage_settings.enabled_at else None,
        "records": {
            "pending": pending_count,
            "reported": reported_count,
            "failed": failed_count,
        },
    }
