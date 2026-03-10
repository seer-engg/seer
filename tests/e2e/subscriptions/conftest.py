"""
Shared fixtures for E2E subscription testing.

Provides core fixtures for creating test users, Stripe customers,
payment methods, and trial subscriptions with test clocks.
"""
import pytest
import stripe
from httpx import AsyncClient

from seer.api.subscriptions.pricing_catalog import invalidate_pricing_cache
from seer.config import config
from seer.database.models import User
from seer.database.organization_models import Organization, OrganizationType

from .helpers import (
    StripeTestClockManager,
    WebhookVerifier,
    attach_test_payment_method,
    create_customer_with_test_clock,
)


def _infer_tier_from_name(name: str) -> str:
    """Infer tier slug from a Stripe product name (e.g. 'Pro Plus' → 'pro_plus')."""
    lowered = name.lower().strip()
    if "pro_plus" in lowered or "pro plus" in lowered:
        return "pro_plus"
    return "pro"


def _ensure_stripe_price_metadata() -> None:
    """Ensure Stripe products and prices have metadata required by the dynamic pricing catalog.

    Patches products with ``tier``, ``display_name``, ``features``, ``sort_order``
    and prices with ``tier``, ``variant``, ``trial_period_days`` when missing.
    """
    stripe.api_key = config.stripe_secret_key
    if not stripe.api_key:
        return

    # --- Products: always set tier metadata from product name ---
    product_tier_map: dict[str, str] = {}
    products = stripe.Product.list(active=True, limit=100)
    for product in products.data:
        prod_meta = product.get("metadata") or {}
        tier = _infer_tier_from_name(product.get("name", ""))
        updates: dict = {
            "tier": tier,
            "display_name": product.get("name", tier),
        }
        if not prod_meta.get("features"):
            updates["features"] = '["Unlimited workflows","Priority support"]'
        updates["sort_order"] = "1" if tier == "pro" else "2"

        new_meta = dict(prod_meta)
        new_meta.update(updates)
        stripe.Product.modify(product.id, metadata=new_meta)

        product_tier_map[product.id] = tier

    # --- Prices: always set tier / variant / trial_period_days ---
    prices = stripe.Price.list(active=True, limit=100)

    # Track which products already have which intervals
    product_intervals: dict[str, set[str]] = {}
    for price in prices.data:
        metadata = price.get("metadata") or {}
        product_id = price.get("product")
        if isinstance(product_id, dict):
            product_id = product_id.get("id")
        tier = product_tier_map.get(product_id, "pro")

        recurring = price.get("recurring") or {}
        interval = recurring.get("interval", "")
        product_intervals.setdefault(product_id, set()).add(interval)

        updates = {
            "tier": tier,
            "variant": "regular",
        }

        # Set trial_period_days for monthly prices
        if interval == "month" and not metadata.get("trial_period_days"):
            updates["trial_period_days"] = "14"

        new_metadata = dict(metadata)
        new_metadata.update(updates)
        stripe.Price.modify(price.id, metadata=new_metadata)

    # --- Create missing annual prices for products that only have monthly ---
    for product_id, intervals in product_intervals.items():
        if "year" not in intervals and "month" in intervals:
            tier = product_tier_map.get(product_id, "pro")
            # Find the monthly price to base the annual price on
            monthly_price = next(
                (p for p in prices.data
                 if (p.get("product") == product_id or
                     (isinstance(p.get("product"), dict) and p["product"].get("id") == product_id))
                 and (p.get("recurring") or {}).get("interval") == "month"),
                None,
            )
            if monthly_price:
                monthly_amount = monthly_price.get("unit_amount", 0)
                annual_amount = monthly_amount * 10  # ~17% discount
                stripe.Price.create(
                    product=product_id,
                    unit_amount=annual_amount,
                    currency=monthly_price.get("currency", "usd"),
                    recurring={"interval": "year"},
                    metadata={
                        "tier": tier,
                        "variant": "regular",
                        "trial_period_days": "14",
                    },
                )

    # Invalidate cache so fresh data is loaded
    invalidate_pricing_cache()


@pytest.fixture
async def user_with_payment_method(db_engine):
    """
    Create test user with payment method already added.

    Yields:
        tuple: (user, organization, stripe_customer_id)

    Cleanup:
        Deletes Stripe customer and DB records
    """
    # Ensure Stripe prices have required metadata for dynamic pricing catalog
    _ensure_stripe_price_metadata()

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

    # Create personal organization with Stripe customer (org-centric billing)
    from seer.database.subscription_models import StripeCustomer
    stripe_customer_record = await StripeCustomer.create(
        stripe_customer_id=stripe_customer.id,
        created_by_user=user,
    )
    organization = await Organization.create(
        owner=user,
        name=f"{user.first_name or 'User'}'s Workspace",
        slug=f"personal-{user.user_id}",
        type=OrganizationType.PERSONAL,
        stripe_customer=stripe_customer_record,
        has_payment_method=True,
    )

    yield user, organization, stripe_customer.id

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
        tuple: (user, organization, stripe_subscription, test_clock)

    Note:
        The subscription is created with a test clock for time-based testing
    """
    user, organization, stripe_customer_id = user_with_payment_method

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
    organization.stripe_customer_id = test_customer.id
    await organization.save()

    # Create subscription with trial (will use test clock)
    from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

    price_id = get_price_id_for_checkout("pro", "month")

    stripe_subscription = stripe.Subscription.create(
        customer=test_customer.id,
        items=[{"price": price_id}],
        trial_period_days=14,  # Start 14-day trial immediately for testing
        metadata={"user_id": user.user_id},
    )

    # Sync subscription to database (creates BillingSubscription record)
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
    await sync_subscription_from_stripe(stripe_subscription)

    yield user, organization, stripe_subscription, test_clock

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
    user, organization, _ = user_with_payment_method

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
