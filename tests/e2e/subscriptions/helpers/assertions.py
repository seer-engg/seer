"""
Custom assertions for E2E subscription testing.

Provides specialized assertion helpers for validating subscription state,
trial periods, invoices, and Stripe synchronization.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe

from seer.database.subscription_models import BillingSubscription, SubscriptionStatus
from seer.logger import get_logger

logger = get_logger("tests.assertions")


async def assert_subscription_synced(
    db_subscription: BillingSubscription, stripe_subscription_id: str
) -> None:
    """
    Assert that a database subscription is synced with Stripe.

    Verifies that all critical fields match between the database record
    and the Stripe subscription object.

    Args:
        db_subscription: The database subscription record
        stripe_subscription_id: The Stripe subscription ID to compare against

    Raises:
        AssertionError: If any fields don't match
    """
    # Fetch Stripe subscription
    stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)

    # Compare status
    stripe_status = stripe_sub.status
    db_status = db_subscription.status.value

    if stripe_status != db_status:
        raise AssertionError(
            f"Status mismatch: Stripe={stripe_status}, DB={db_status}"
        )

    # Compare Stripe subscription ID
    if db_subscription.stripe_subscription_id != stripe_subscription_id:
        raise AssertionError(
            f"Subscription ID mismatch: DB={db_subscription.stripe_subscription_id}, "
            f"Stripe={stripe_subscription_id}"
        )

    # Compare period dates (with 1 second tolerance for timestamp conversion)
    # Note: During trial, current_period_end may not exist
    stripe_period_end_ts = stripe_sub.get("current_period_end")
    if stripe_period_end_ts:
        stripe_period_end = datetime.fromtimestamp(
            stripe_period_end_ts, tz=timezone.utc
        )
        db_period_end = db_subscription.current_period_end

        if db_period_end:
            diff = abs((stripe_period_end - db_period_end).total_seconds())
            if diff > 1:
                raise AssertionError(
                    f"Period end mismatch: Stripe={stripe_period_end}, "
                    f"DB={db_period_end}, diff={diff}s"
                )

    logger.info(
        "Subscription %s is synced: status=%s, period_end=%s",
        stripe_subscription_id,
        stripe_status,
        db_subscription.current_period_end,
    )


def assert_trial_period_correct(
    subscription: stripe.Subscription, expected_days: int = 14
) -> None:
    """
    Assert that a subscription has the correct trial period.

    Args:
        subscription: The Stripe subscription object
        expected_days: Expected trial period in days (default: 14)

    Raises:
        AssertionError: If trial period is incorrect
    """
    if not subscription.trial_end:
        raise AssertionError("Subscription has no trial end date")

    trial_end = datetime.fromtimestamp(subscription.trial_end, tz=timezone.utc)
    created = datetime.fromtimestamp(subscription.created, tz=timezone.utc)

    trial_duration = trial_end - created
    expected_duration = timedelta(days=expected_days)

    # Allow 1 second tolerance
    diff_seconds = abs((trial_duration - expected_duration).total_seconds())

    if diff_seconds > 1:
        raise AssertionError(
            f"Trial period mismatch: expected={expected_days} days, "
            f"actual={trial_duration.days} days {trial_duration.seconds} seconds"
        )

    logger.info(
        "Trial period verified: %d days (created=%s, trial_end=%s)",
        expected_days,
        created.isoformat(),
        trial_end.isoformat(),
    )


async def assert_no_charges_during_trial(stripe_customer_id: str) -> None:
    """
    Assert that no charges have been made during a trial period.

    Args:
        stripe_customer_id: The Stripe customer ID to check

    Raises:
        AssertionError: If any charges are found
    """
    charges = stripe.Charge.list(customer=stripe_customer_id, limit=100)

    if charges.data:
        charge_details = [
            f"{charge.id}: ${charge.amount / 100} {charge.currency}"
            for charge in charges.data
        ]
        raise AssertionError(
            f"Found {len(charges.data)} charges during trial: {charge_details}"
        )

    logger.info("Verified no charges for customer %s", stripe_customer_id)


def assert_invoice_amount(
    invoice: stripe.Invoice, expected_amount_cents: int
) -> None:
    """
    Assert that an invoice has the expected amount.

    Args:
        invoice: The Stripe invoice object
        expected_amount_cents: Expected amount in cents

    Raises:
        AssertionError: If amount doesn't match
    """
    actual_amount = invoice.amount_due

    if actual_amount != expected_amount_cents:
        raise AssertionError(
            f"Invoice amount mismatch: expected=${expected_amount_cents / 100}, "
            f"actual=${actual_amount / 100}"
        )

    logger.info(
        "Invoice %s amount verified: $%.2f",
        invoice.id,
        actual_amount / 100,
    )


def assert_subscription_status(
    subscription: stripe.Subscription, expected_status: str
) -> None:
    """
    Assert that a subscription has the expected status.

    Args:
        subscription: The Stripe subscription object
        expected_status: Expected status (e.g., "trialing", "active", "canceled")

    Raises:
        AssertionError: If status doesn't match
    """
    actual_status = subscription.status

    if actual_status != expected_status:
        raise AssertionError(
            f"Subscription status mismatch: expected={expected_status}, "
            f"actual={actual_status}"
        )

    logger.info("Subscription %s status verified: %s", subscription.id, actual_status)


def assert_period_dates_progression(
    old_period_end: datetime,
    new_period_end: datetime,
    expected_interval_days: int,
) -> None:
    """
    Assert that subscription period dates progressed correctly.

    Args:
        old_period_end: Previous period end date
        new_period_end: New period end date
        expected_interval_days: Expected interval in days (30 for monthly, 365 for annual)

    Raises:
        AssertionError: If progression is incorrect
    """
    diff = new_period_end - old_period_end
    diff_days = diff.days

    # Allow 1 day tolerance for months with different lengths
    if abs(diff_days - expected_interval_days) > 1:
        raise AssertionError(
            f"Period date progression incorrect: expected={expected_interval_days} days, "
            f"actual={diff_days} days"
        )

    logger.info(
        "Period date progression verified: %s -> %s (%d days)",
        old_period_end.isoformat(),
        new_period_end.isoformat(),
        diff_days,
    )


async def assert_subscription_deleted(user_id: str) -> None:
    """
    Assert that a user's subscription has been deleted/reverted to free.

    Args:
        user_id: The user ID to check

    Raises:
        AssertionError: If subscription is still active
    """
    from seer.database.subscription_models import (
        BillingProfile,
        BillingSubscription,
        SubscriptionTier,
    )
    from seer.database.models import User

    # Get user first
    user = await User.get_or_none(user_id=user_id)
    if not user:
        raise AssertionError(f"No user found with user_id {user_id}")

    # Get billing profile via owner_user relationship
    profile = await BillingProfile.get_or_none(owner_user=user)

    if not profile:
        raise AssertionError(f"No billing profile found for user {user_id}")

    # Get subscription if it exists
    subscription = await BillingSubscription.get_or_none(billing_profile=profile)

    if subscription and subscription.tier != SubscriptionTier.FREE:
        raise AssertionError(
            f"User {user_id} still has paid tier: {subscription.tier.value}"
        )

    if subscription and subscription.stripe_subscription_id:
        raise AssertionError(
            f"User {user_id} still has Stripe subscription ID: "
            f"{subscription.stripe_subscription_id}"
        )

    logger.info("Verified user %s subscription deleted/reverted to free", user_id)


def assert_webhook_delivered(
    webhook_event: Optional[object], event_type: str
) -> None:
    """
    Assert that a webhook event was delivered.

    Args:
        webhook_event: The webhook event record (or None)
        event_type: Expected event type for error messages

    Raises:
        AssertionError: If webhook was not delivered
    """
    if webhook_event is None:
        raise AssertionError(f"Webhook event {event_type} was not delivered")

    logger.info("Webhook event %s delivered", event_type)
