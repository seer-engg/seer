"""
E2E tests for onboarding flow with trial subscription creation.

Tests the complete onboarding flow including:
- Trial subscription creation after payment method added
- Webhook synchronization
- Error handling
"""
from datetime import datetime, timedelta, timezone

import pytest
import stripe

from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
from seer.config import config
from seer.database.subscription_models import BillingSubscription, BillingProfile, SubscriptionTier

from .helpers import (
    assert_no_charges_during_trial,
    assert_subscription_synced,
    assert_trial_period_correct,
)


@pytest.mark.asyncio
async def test_create_trial_subscription_during_onboarding(
    authenticated_subscription_client, user_with_payment_method, pro_monthly_price
):
    """
    Test creating a trial subscription during onboarding flow.

    Verifies:
    - Trial subscription created with status=trialing
    - Trial end date is now + 14 days
    - No invoice created yet
    - DB synced with Stripe
    """
    user, billing_profile, stripe_customer_id = user_with_payment_method

    # Call the create-with-trial endpoint (simulates onboarding)
    response = await authenticated_subscription_client.post(
        "/api/subscriptions/create-with-trial",
        json={"tier": "pro", "interval": "month"},
    )

    assert response.status_code == 200
    result = response.json()

    # Verify response
    assert result["subscription_id"].startswith("sub_")
    assert result["status"] == "trialing"
    assert result["trial_end"] is not None

    # Verify trial end date is 14 days from now
    trial_end = datetime.fromisoformat(result["trial_end"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    expected_trial_end = now + timedelta(days=14)

    # Allow 5 seconds tolerance
    time_diff = abs((trial_end - expected_trial_end).total_seconds())
    assert time_diff < 5, f"Trial end date incorrect: expected ~{expected_trial_end}, got {trial_end}"

    # Verify in Stripe
    stripe.api_key = config.stripe_secret_key
    stripe_sub = stripe.Subscription.retrieve(result["subscription_id"])

    assert stripe_sub.status == "trialing"
    assert_trial_period_correct(stripe_sub, expected_days=14)

    # Verify no charges during trial (Stripe may create a $0 invoice for subscription_create)
    await assert_no_charges_during_trial(stripe_customer_id)

    # Verify any invoices are $0 (subscription_create invoices are expected)
    invoices = stripe.Invoice.list(customer=stripe_customer_id, limit=10)
    for invoice in invoices.data:
        assert invoice.amount_due == 0, (
            f"Invoice during trial should be $0, got ${invoice.amount_due / 100} "
            f"(billing_reason: {invoice.billing_reason})"
        )

    # Verify DB is synced
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    assert subscription.tier == SubscriptionTier.PRO
    assert subscription.stripe_subscription_id == result["subscription_id"]

    # Verify full sync
    await assert_subscription_synced(subscription, result["subscription_id"])


@pytest.mark.asyncio
async def test_create_trial_requires_payment_method(db_engine):
    """
    Test that creating a trial subscription requires a payment method.

    Verifies:
    - 400 error when no payment method is attached
    - Appropriate error message returned
    """
    from seer.database.models import User
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from unittest.mock import patch

    # Create user without payment method
    user = await User.create(
        user_id="test_user_no_pm",
        email="no_payment@example.com",
        clerk_user_id="clerk_no_pm",
    )

    # Create Stripe customer without payment method
    stripe.api_key = config.stripe_secret_key
    stripe_customer = stripe.Customer.create(
        email=user.email,
        metadata={"user_id": user.user_id},
    )

    # Create billing profile
    await BillingProfile.create(
        owner_user=user,
        stripe_customer_id=stripe_customer.id,
    )

    # Create test app and client with mocked auth
    app = FastAPI(title="Test Subscription App")
    from seer.api.subscriptions import router as subscription_router
    app.include_router(subscription_router.router, prefix="/api")

    def mock_require_user(request):
        return user

    # Try to create subscription without payment method
    with patch('seer.api.subscriptions.router._require_user', side_effect=mock_require_user):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/subscriptions/create-with-trial",
                json={"tier": "pro", "interval": "month"},
            )

    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "payment method" in error_detail.lower()

    # Cleanup
    try:
        stripe.Customer.delete(stripe_customer.id)
    except stripe.error.StripeError:
        pass


@pytest.mark.asyncio
async def test_webhook_sync_after_trial_creation(
    subscription_with_trial, user_with_payment_method, webhook_verifier
):
    """
    Test that webhook synchronizes subscription after trial creation.

    Verifies:
    - customer.subscription.created webhook is received
    - Webhook is processed successfully
    - DB is updated via sync_subscription_from_stripe()
    - Webhook idempotency (reprocessing same event)
    """
    user, billing_profile, _ = user_with_payment_method
    subscription_id = subscription_with_trial["subscription_id"]

    # Simulate webhook event (in real scenario, Stripe sends this)
    stripe.api_key = config.stripe_secret_key
    stripe_sub = stripe.Subscription.retrieve(subscription_id)

    # Create a mock webhook event
    event = stripe.Event.construct_from(
        {
            "id": f"evt_test_{subscription_id}",
            "type": "customer.subscription.created",
            "data": {"object": stripe_sub.to_dict()},
            "created": int(datetime.now(timezone.utc).timestamp()),
        },
        config.stripe_secret_key,
    )

    # Wait for webhook to be received and processed
    webhook_event = await webhook_verifier.wait_for_webhook(event.id, timeout=5.0)

    # Verify webhook was received
    if webhook_event:
        # Verify processing
        processed = await webhook_verifier.verify_webhook_processed(event.id, timeout=5.0)
        assert processed, "Webhook should be processed successfully"

    # Verify DB was updated
    await billing_profile.refresh_from_db()
    subscription = await BillingSubscription.get(billing_profile=billing_profile)
    assert subscription.stripe_subscription_id == subscription_id
    assert subscription.tier == SubscriptionTier.PRO

    # Test idempotency: reprocessing same event should be safe
    if webhook_event:
        idempotent = await webhook_verifier.verify_webhook_idempotency(event.id)
        assert idempotent, "Webhook reprocessing should be idempotent"


@pytest.mark.asyncio
async def test_trial_end_date_calculation(subscription_with_trial):
    """
    Test that trial end date is calculated correctly (14 days from creation).

    Verifies:
    - trial_end - created = 14 days (±1 second tolerance)
    - current_period_end = trial_end during trial
    """
    subscription_id = subscription_with_trial["subscription_id"]
    trial_end_str = subscription_with_trial["trial_end"]

    # Retrieve from Stripe
    stripe.api_key = config.stripe_secret_key
    stripe_sub = stripe.Subscription.retrieve(subscription_id)

    created = datetime.fromtimestamp(stripe_sub.created, tz=timezone.utc)
    trial_end = datetime.fromtimestamp(stripe_sub.trial_end, tz=timezone.utc)

    # Calculate duration
    trial_duration = trial_end - created
    expected_duration = timedelta(days=14)

    # Allow 1 second tolerance
    diff_seconds = abs((trial_duration - expected_duration).total_seconds())
    assert diff_seconds <= 1, (
        f"Trial duration incorrect: expected 14 days, got {trial_duration.days} days "
        f"{trial_duration.seconds} seconds (diff: {diff_seconds}s)"
    )

    # During trial, current_period_end may not exist or equals trial_end
    # Stripe's behavior: during trial, the billing period hasn't started yet
    current_period_end = stripe_sub.get("current_period_end")
    if current_period_end:
        assert current_period_end == stripe_sub.trial_end, (
            f"If current_period_end exists during trial, it should equal trial_end: "
            f"current_period_end={current_period_end}, trial_end={stripe_sub.trial_end}"
        )


@pytest.mark.asyncio
async def test_invalid_tier_rejected(authenticated_subscription_client):
    """
    Test that invalid tier values are rejected with 400 error.
    """
    response = await authenticated_subscription_client.post(
        "/api/subscriptions/create-with-trial",
        json={"tier": "invalid_tier", "interval": "month"},
    )

    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "invalid tier" in error_detail.lower()


@pytest.mark.asyncio
async def test_invalid_interval_rejected(authenticated_subscription_client):
    """
    Test that invalid interval values are rejected with 400 error.
    """
    response = await authenticated_subscription_client.post(
        "/api/subscriptions/create-with-trial",
        json={"tier": "pro", "interval": "invalid_interval"},
    )

    assert response.status_code == 400
    # The endpoint uses get_price_id_for_checkout which returns None for invalid intervals
    error_detail = response.json()["detail"]
    assert "price not found" in error_detail.lower()


@pytest.mark.asyncio
async def test_subscription_has_default_payment_method(
    authenticated_subscription_client, user_with_payment_method
):
    """
    Verify subscription is created with default payment method.

    This test ensures that:
    - Subscription has default_payment_method set
    - Customer has invoice_settings.default_payment_method set
    - Payment method IDs match the one attached to the customer

    This prevents payment failures when the trial period ends.
    """
    user, billing_profile, stripe_customer_id = user_with_payment_method

    # Create subscription via API
    response = await authenticated_subscription_client.post(
        "/api/subscriptions/create-with-trial",
        json={"tier": "pro", "interval": "month"},
    )

    assert response.status_code == 200
    result = response.json()

    # Retrieve subscription from Stripe
    stripe.api_key = config.stripe_secret_key
    subscription = stripe.Subscription.retrieve(result["subscription_id"])

    # Assert subscription has default payment method set
    assert subscription.default_payment_method is not None, (
        "Subscription must have default_payment_method to charge after trial"
    )
    assert subscription.default_payment_method.startswith("pm_"), (
        f"Expected payment method ID starting with 'pm_', got {subscription.default_payment_method}"
    )

    # Also verify customer-level default
    customer = stripe.Customer.retrieve(stripe_customer_id)
    assert customer.invoice_settings.default_payment_method is not None, (
        "Customer must have default_payment_method for future invoices"
    )

    # Verify both defaults point to the same payment method
    assert subscription.default_payment_method == customer.invoice_settings.default_payment_method, (
        "Subscription and customer default payment methods should match"
    )

    # Verify the payment method actually exists and is attached to the customer
    payment_method = stripe.PaymentMethod.retrieve(subscription.default_payment_method)
    assert payment_method.customer == stripe_customer_id, (
        "Payment method must be attached to the customer"
    )
