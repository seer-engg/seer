"""
Shared fixtures for E2E subscription testing.

Provides core fixtures for creating test users, Stripe customers,
payment methods, and trial subscriptions with test clocks.
"""
import pytest
import stripe
from httpx import AsyncClient

from seer.api.subscriptions.stripe_service import get_or_create_stripe_customer
from seer.config import config
from seer.database.models import User
from seer.database.subscription_models import BillingProfile, SubscriptionTier

from .helpers import (
    StripeTestClockManager,
    WebhookVerifier,
    attach_test_payment_method,
    create_customer_with_test_clock,
)


@pytest.fixture
async def user_with_payment_method(db_engine):
    """
    Create test user with payment method already added.

    Yields:
        tuple: (user, billing_profile, stripe_customer_id)

    Cleanup:
        Deletes Stripe customer and DB records
    """
    # Create test user
    user = await User.create(
        user_id="test_user_trial",
        email="test_trial@example.com",
        clerk_user_id="clerk_test_trial",
    )

    # Create Stripe customer
    stripe.api_key = config.stripe_secret_key
    stripe_customer = stripe.Customer.create(
        email=user.email,
        metadata={"user_id": user.user_id},
    )

    # Attach test payment method
    attach_test_payment_method(stripe_customer.id)

    # Create billing profile
    billing_profile = await BillingProfile.create(
        owner_user=user,
        stripe_customer_id=stripe_customer.id,
    )

    # Create early adopter counters (required for subscription creation)
    from seer.database.subscription_models import EarlyAdopterCounter
    await EarlyAdopterCounter.get_or_create(tier="pro", defaults={"count": 0})
    await EarlyAdopterCounter.get_or_create(tier="pro_plus", defaults={"count": 0})

    yield user, billing_profile, stripe_customer.id

    # Cleanup: Delete Stripe customer
    try:
        stripe.Customer.delete(stripe_customer.id)
    except stripe.error.StripeError:
        pass  # Already deleted or doesn't exist

    # DB cleanup handled by db_engine fixture


@pytest.fixture
def stripe_test_clock():
    """
    Stripe test clock for time manipulation.

    Returns:
        StripeTestClockManager: Manager for creating and advancing test clocks

    Cleanup:
        Deletes all created test clocks
    """
    manager = StripeTestClockManager()
    yield manager
    manager.cleanup()


@pytest.fixture
async def trial_subscription_setup(user_with_payment_method, stripe_test_clock):
    """
    Create trial subscription with test clock.

    Yields:
        tuple: (user, billing_profile, stripe_subscription, test_clock)

    Note:
        The subscription is created with a test clock for time-based testing
    """
    user, billing_profile, stripe_customer_id = user_with_payment_method

    # Create test clock
    test_clock = stripe_test_clock.create_clock()

    # Create new customer associated with test clock
    test_customer = create_customer_with_test_clock(
        email=f"clock_{user.email}",
        test_clock_id=test_clock.id,
    )

    # Attach payment method to test clock customer
    attach_test_payment_method(test_customer.id)

    # Update billing profile with test clock customer
    billing_profile.stripe_customer_id = test_customer.id
    await billing_profile.save()

    # Create subscription with trial (will use test clock)
    from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

    price_id = get_price_id_for_checkout("pro", "month", is_early_adopter=False)

    stripe_subscription = stripe.Subscription.create(
        customer=test_customer.id,
        items=[{"price": price_id}],
        trial_period_days=14,  # Start 14-day trial immediately for testing
        metadata={"user_id": user.user_id},
    )

    # Sync subscription to database (creates BillingSubscription record)
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
    await sync_subscription_from_stripe(stripe_subscription)

    yield user, billing_profile, stripe_subscription, test_clock

    # Cleanup: Cancel subscription
    try:
        stripe.Subscription.delete(stripe_subscription.id)
    except stripe.error.StripeError:
        pass

    # Cleanup: Delete test customer
    try:
        stripe.Customer.delete(test_customer.id)
    except stripe.error.StripeError:
        pass


@pytest.fixture
def webhook_verifier():
    """
    Helper for verifying webhook delivery and processing.

    Returns:
        WebhookVerifier: Verifier instance with wait_for_webhook() method
    """
    return WebhookVerifier(timeout=10.0)


@pytest.fixture
async def authenticated_subscription_client(user_with_payment_method):
    """
    Authenticated API client for subscription endpoints.

    Returns:
        AsyncClient: HTTP client with auth headers for the test user
    """
    from fastapi import FastAPI
    from unittest.mock import patch
    from httpx import ASGITransport

    user, _, _ = user_with_payment_method

    # Create a minimal app with just subscription routes (no auth middleware)
    app = FastAPI(title="Test Subscription App")

    # Import and include subscription routes
    from seer.api.subscriptions import router as subscription_router
    app.include_router(subscription_router.router, prefix="/api")

    # Create mock function that returns the user
    def mock_require_user(request):
        return user

    # Mock the _require_user functions to return our test user
    with patch('seer.api.subscriptions.router._require_user', side_effect=mock_require_user), \
         patch('seer.api.subscriptions.setup_intent._require_user', side_effect=mock_require_user):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def subscription_with_trial(authenticated_subscription_client, user_with_payment_method):
    """
    Create a subscription with trial via the API endpoint.

    This fixture tests the actual onboarding flow by calling the
    /api/subscriptions/create-with-trial endpoint.

    Yields:
        dict: Response from create-with-trial endpoint containing subscription_id,
              status, and trial_end
    """
    user, billing_profile, _ = user_with_payment_method

    # Call the API endpoint
    response = await authenticated_subscription_client.post(
        "/api/subscriptions/create-with-trial",
        json={"tier": "pro", "interval": "month"},
    )

    assert response.status_code == 200
    result = response.json()

    yield result

    # Cleanup: Cancel subscription
    if result.get("subscription_id"):
        try:
            stripe.Subscription.delete(result["subscription_id"])
        except stripe.error.StripeError:
            pass


@pytest.fixture
def pro_monthly_price():
    """Get the Pro monthly price in cents."""
    return 3900  # $39.00


@pytest.fixture
def pro_annual_price():
    """Get the Pro annual price in cents."""
    return 39000  # $390.00


@pytest.fixture
def pro_plus_monthly_price():
    """Get the Pro+ monthly price in cents."""
    return 7900  # $79.00


@pytest.fixture
def pro_plus_annual_price():
    """Get the Pro+ annual price in cents."""
    return 79000  # $790.00
