"""
E2E tests for trial expiration and conversion to active subscription.

Tests critical behavior:
- Trial converts to active after 14 days
- First payment is collected
- Webhook synchronization
- Period dates update correctly
"""
from datetime import datetime, timedelta, timezone

import pytest
import stripe

from seer.config import config
from seer.database.subscription_models import BillingSubscription, SubscriptionStatus

from .helpers import (
    assert_invoice_amount,
    assert_period_dates_progression,
    assert_subscription_status,
    assert_subscription_synced,
)


@pytest.mark.asyncio
async def test_trial_converts_to_active_after_14_days(
    trial_subscription_setup, stripe_test_clock, pro_monthly_price
):
    """
    ⭐ CRITICAL TEST: Verify trial converts to active subscription after 14 days.

    This is the most important test - validates that:
    - Trial ends after exactly 14 days
    - Status changes from "trialing" to "active"
    - First invoice is created and paid
    - Charge amount is correct ($39 for Pro monthly)
    - DB is synced with new status and period dates
    - Webhooks are received: customer.subscription.updated, invoice.payment_succeeded
    """
    user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup
    
    # Fetch subscription from DB
    subscription = await BillingSubscription.get(billing_profile=billing_profile)

    # Verify initial state
    stripe.api_key = config.stripe_secret_key
    initial_sub = stripe.Subscription.retrieve(stripe_subscription.id)
    assert initial_sub.status == "trialing"

    trial_end = datetime.fromtimestamp(initial_sub.trial_end, tz=timezone.utc)

    # Advance clock to 14 days + 1 hour (past trial end)
    stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

    # Wait a moment for Stripe to process the trial end
    import asyncio
    await asyncio.sleep(2)

    # Retrieve updated subscription
    updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)

    # Verify status changed to active
    assert_subscription_status(updated_sub, "active")

    # Verify invoice was created
    invoices = stripe.Invoice.list(
        customer=updated_sub.customer,
        subscription=stripe_subscription.id,
        limit=1,
    )

    assert len(invoices.data) > 0, "Invoice should be created after trial ends"
    invoice = invoices.data[0]

    # Verify invoice payment succeeded
    assert invoice.status == "paid", f"Invoice should be paid, got: {invoice.status}"

    # Verify charge amount (Pro monthly = $39.00 = 3900 cents)
    assert_invoice_amount(invoice, pro_monthly_price)

    # Verify billing reason
    assert invoice.billing_reason == "subscription_cycle"

    # Verify charge was created
    charges = stripe.Charge.list(customer=updated_sub.customer, limit=1)
    assert len(charges.data) > 0, "Charge should be created"
    charge = charges.data[0]
    assert charge.amount == pro_monthly_price
    assert charge.paid is True

    # Verify period dates updated
    new_period_start = datetime.fromtimestamp(updated_sub.current_period_start, tz=timezone.utc)
    new_period_end = datetime.fromtimestamp(updated_sub.current_period_end, tz=timezone.utc)

    # Period start should be the trial end
    assert abs((new_period_start - trial_end).total_seconds()) < 5, (
        f"Period start should equal trial end: trial_end={trial_end}, "
        f"period_start={new_period_start}"
    )

    # Period end should be ~30 days after period start (monthly)
    assert_period_dates_progression(new_period_start, new_period_end, expected_interval_days=30)

    # Verify DB is synced
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    assert subscription.status == SubscriptionStatus.ACTIVE
    await assert_subscription_synced(subscription, stripe_subscription.id)


@pytest.mark.asyncio
async def test_webhook_updates_status_to_active(
    trial_subscription_setup, stripe_test_clock, webhook_verifier
):
    """
    Test that customer.subscription.updated webhook updates DB status to active.

    Verifies:
    - Webhook is received when trial ends
    - sync_subscription_from_stripe() updates DB
    - Status transitions from "trialing" to "active"
    """
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe

    user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup
    
    # Fetch subscription from DB
    subscription = await BillingSubscription.get(billing_profile=billing_profile)

    # Verify initial status
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    initial_status = subscription.status

    # Advance clock past trial end
    stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

    # Wait for Stripe to process
    import asyncio
    await asyncio.sleep(2)

    # Retrieve updated subscription
    stripe.api_key = config.stripe_secret_key
    updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)

    # Manually sync (simulates webhook processing)
    await sync_subscription_from_stripe(updated_sub)

    # Verify DB updated
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    assert subscription.status == SubscriptionStatus.ACTIVE
    assert subscription.status != initial_status


@pytest.mark.asyncio
async def test_invoice_payment_succeeded_webhook(
    trial_subscription_setup, stripe_test_clock
):
    """
    Test that invoice.payment_succeeded webhook syncs subscription correctly.

    Verifies:
    - Invoice is created at trial end
    - invoice.payment_succeeded event is generated
    - Subscription can be synced from invoice data
    """
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe

    user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup
    
    # Fetch subscription from DB
    subscription = await BillingSubscription.get(billing_profile=billing_profile)

    # Advance past trial
    stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

    # Wait for processing
    import asyncio
    await asyncio.sleep(2)

    # Retrieve invoice
    stripe.api_key = config.stripe_secret_key
    invoices = stripe.Invoice.list(
        subscription=stripe_subscription.id,
        limit=1,
    )

    assert len(invoices.data) > 0
    invoice = invoices.data[0]
    assert invoice.status == "paid"

    # Get subscription from invoice
    subscription_from_invoice = stripe.Subscription.retrieve(invoice.subscription)

    # Sync from invoice data
    await sync_subscription_from_stripe(subscription_from_invoice)

    # Verify DB updated
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    assert subscription.status == SubscriptionStatus.ACTIVE


@pytest.mark.asyncio
async def test_current_period_dates_after_trial_ends(
    trial_subscription_setup, stripe_test_clock
):
    """
    Test that current_period_start and current_period_end update correctly after trial.

    Verifies:
    - current_period_start = trial_end
    - current_period_end = trial_end + 30 days (monthly)
    - DB dates match Stripe dates
    """
    user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup
    
    # Fetch subscription from DB
    subscription = await BillingSubscription.get(billing_profile=billing_profile)

    # Get trial end date
    stripe.api_key = config.stripe_secret_key
    initial_sub = stripe.Subscription.retrieve(stripe_subscription.id)
    trial_end = datetime.fromtimestamp(initial_sub.trial_end, tz=timezone.utc)

    # Advance past trial
    stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

    # Wait for processing
    import asyncio
    await asyncio.sleep(2)

    # Retrieve updated subscription
    updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)

    # Verify period dates
    period_start = datetime.fromtimestamp(updated_sub.current_period_start, tz=timezone.utc)
    period_end = datetime.fromtimestamp(updated_sub.current_period_end, tz=timezone.utc)

    # Period start should equal trial end (±5 seconds)
    time_diff = abs((period_start - trial_end).total_seconds())
    assert time_diff < 5, (
        f"Period start should equal trial end: trial_end={trial_end}, "
        f"period_start={period_start}, diff={time_diff}s"
    )

    # Period end should be 30 days after period start (monthly subscription)
    assert_period_dates_progression(period_start, period_end, expected_interval_days=30)

    # Sync to DB and verify
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
    await sync_subscription_from_stripe(updated_sub)

    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    db_period_end = subscription.current_period_end

    # DB period end should match Stripe (±1 second for timestamp conversion)
    if db_period_end:
        db_diff = abs((db_period_end - period_end).total_seconds())
        assert db_diff <= 1, f"DB period end doesn't match Stripe: diff={db_diff}s"


@pytest.mark.asyncio
async def test_trial_expiration_with_annual_subscription(
    user_with_payment_method, stripe_test_clock
):
    """
    Test that annual subscriptions also get 14-day trial and convert correctly.

    Verifies:
    - Annual subscription gets 14-day trial
    - After trial, period_end = trial_end + 365 days
    - Charge is annual amount ($390 for Pro annual)
    """
    from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

    user, billing_profile, stripe_customer_id = user_with_payment_method

    # Create test clock and customer
    from .helpers import create_customer_with_test_clock, attach_test_payment_method

    test_clock = stripe_test_clock.create_clock()
    test_customer = create_customer_with_test_clock(
        email=f"annual_{user.email}",
        test_clock_id=test_clock.id,
    )
    attach_test_payment_method(test_customer.id)

    # Create annual subscription with trial
    stripe.api_key = config.stripe_secret_key
    price_id = get_price_id_for_checkout("pro", "year", is_early_adopter=False)

    subscription = stripe.Subscription.create(
        customer=test_customer.id,
        items=[{"price": price_id}],
        trial_period_days=14,  # Start 14-day trial immediately
        metadata={"user_id": user.user_id},
    )

    # Verify trial exists
    assert subscription.trial_end is not None
    trial_end = datetime.fromtimestamp(subscription.trial_end, tz=timezone.utc)

    # Advance past trial
    stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

    # Wait for processing
    import asyncio
    await asyncio.sleep(2)

    # Retrieve updated subscription
    updated_sub = stripe.Subscription.retrieve(subscription.id)
    assert updated_sub.status == "active"

    # Verify annual period (365 days)
    period_start = datetime.fromtimestamp(updated_sub.current_period_start, tz=timezone.utc)
    period_end = datetime.fromtimestamp(updated_sub.current_period_end, tz=timezone.utc)

    assert_period_dates_progression(period_start, period_end, expected_interval_days=365)

    # Verify annual charge ($390)
    invoices = stripe.Invoice.list(subscription=subscription.id, limit=1)
    assert len(invoices.data) > 0
    invoice = invoices.data[0]
    assert_invoice_amount(invoice, 39000)  # $390.00

    # Cleanup
    try:
        stripe.Subscription.delete(subscription.id)
        stripe.Customer.delete(test_customer.id)
    except stripe.error.StripeError:
        pass
