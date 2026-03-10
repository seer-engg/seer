"""
Integration tests for Setup Intent API endpoints.

Tests payment method collection during onboarding flow.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.database.models import User
from seer.database.organization_models import Organization, OrganizationType
from seer.database.subscription_models import StripeCustomer


@pytest.fixture
async def test_user(db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
    """Create a test user with organization (org-centric billing)."""
    user = await User.create(
        user_id="test_user_123",
        email="test@example.com",
        first_name="Test",
        last_name="User",
    )
    stripe_customer = await StripeCustomer.create(
        stripe_customer_id="cus_test123",
        created_by_user=user,
    )
    organization = await Organization.create(
        owner=user,
        name=f"{user.first_name or 'User'}'s Workspace",
        slug=f"personal-{user.user_id}",
        type=OrganizationType.PERSONAL,
        stripe_customer=stripe_customer,
        has_payment_method=False,
    )
    yield user, organization
    await organization.delete()
    await stripe_customer.delete()
    await user.delete()


@pytest.fixture
def app_with_subscriptions(mock_app: FastAPI):
    """Create test app with subscriptions router."""
    from seer.api.router import router as api_router  # pylint: disable=import-outside-toplevel  # Reason: Dynamic import for test fixture

    mock_app.include_router(api_router)
    return mock_app


@pytest.fixture
def mock_stripe_config():
    """Mock Stripe configuration."""
    with patch("seer.api.subscriptions.setup_intent.config") as mock_config:
        mock_config.stripe_publishable_key = "pk_test_123"
        mock_config.stripe_secret_key = "sk_test_123"
        mock_config.stripe_webhook_secret = "whsec_test_123"
        mock_config.is_stripe_configured = True
        yield mock_config


@pytest.fixture
def mock_stripe_setup_intent():
    """Mock Stripe SetupIntent.create response."""
    with patch("seer.api.subscriptions.setup_intent.stripe.SetupIntent") as mock_si:
        mock_si.create.return_value = MagicMock(
            id="seti_test123",
            client_secret="seti_test123_secret_abc",
            status="requires_payment_method",
            customer="cus_test123",
        )
        mock_si.retrieve.return_value = MagicMock(
            id="seti_test123",
            status="succeeded",
            customer="cus_test123",
        )
        yield mock_si


@pytest.mark.asyncio
class TestSetupIntentEndpoints:
    """Test Setup Intent API endpoints."""

    async def test_create_setup_intent_success(self, app_with_subscriptions: FastAPI, test_user, mock_stripe_config, mock_stripe_setup_intent):  # pylint: disable=unused-argument # Reason: mock_stripe_config needed for fixture setup
        """Test creating a Setup Intent for payment method collection."""
        user, organization = test_user

        # Mock authentication
        with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
            with patch("seer.api.subscriptions.setup_intent._require_organization", return_value=organization):
                with patch("seer.api.subscriptions.setup_intent.get_or_create_org_stripe_customer", return_value="cus_test123"):
                    async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                        response = await client.post("/api/subscriptions/setup-intent")

        assert response.status_code == 200
        data = response.json()
        assert data["client_secret"] == "seti_test123_secret_abc"
        assert data["stripe_customer_id"] == "cus_test123"
        mock_stripe_setup_intent.create.assert_called_once()

    async def test_create_setup_intent_unauthenticated(self, app_with_subscriptions: FastAPI, mock_stripe_config):  # pylint: disable=unused-argument # Reason: mock_stripe_config needed for fixture setup
        """Test creating Setup Intent without authentication."""
        with patch("seer.api.subscriptions.setup_intent._require_user", side_effect=Exception("Authentication required")):
            async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                with pytest.raises(Exception):
                    await client.post("/api/subscriptions/setup-intent")

    async def test_confirm_setup_intent_success(self, app_with_subscriptions: FastAPI, test_user, mock_stripe_config, mock_stripe_setup_intent):  # pylint: disable=unused-argument # Reason: mock_stripe_config needed for fixture setup
        """Test confirming a successful Setup Intent."""
        user, organization = test_user

        # Ensure payment method not set initially
        assert organization.has_payment_method is False
        assert organization.payment_method_added_at is None

        with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
            with patch("seer.api.subscriptions.setup_intent._require_organization", return_value=organization):
                async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                    response = await client.post(
                        "/api/subscriptions/setup-intent/confirm",
                        json={"setup_intent_id": "seti_test123"}
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Payment method successfully added"

        # Verify organization updated
        await organization.refresh_from_db()
        assert organization.has_payment_method is True
        assert organization.payment_method_added_at is not None
        mock_stripe_setup_intent.retrieve.assert_called_once_with("seti_test123")

    async def test_confirm_setup_intent_not_succeeded(self, app_with_subscriptions: FastAPI, test_user, mock_stripe_config):  # pylint: disable=unused-argument # Reason: mock_stripe_config needed for fixture setup
        """Test confirming Setup Intent that hasn't succeeded yet."""
        user, organization = test_user

        with patch("seer.api.subscriptions.setup_intent.stripe.SetupIntent") as mock_si:
            mock_si.retrieve.return_value = MagicMock(
                id="seti_test123",
                status="requires_payment_method",  # Not succeeded
                customer="cus_test123",
            )

            with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
                with patch("seer.api.subscriptions.setup_intent._require_organization", return_value=organization):
                    async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                        response = await client.post(
                            "/api/subscriptions/setup-intent/confirm",
                            json={"setup_intent_id": "seti_test123"}
                        )

        assert response.status_code == 400
        assert "expected 'succeeded'" in response.json()["detail"].lower()

        # Verify organization not updated
        await organization.refresh_from_db()
        assert organization.has_payment_method is False

    async def test_get_payment_method_status_has_method(self, app_with_subscriptions: FastAPI, test_user):
        """Test getting payment method status when user has payment method."""
        user, organization = test_user

        # Set payment method
        organization.has_payment_method = True
        organization.payment_method_added_at = datetime.now(timezone.utc)
        await organization.save()

        with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
            with patch("seer.api.subscriptions.setup_intent._require_organization", return_value=organization):
                async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                    response = await client.get("/api/subscriptions/payment-method/status")

        assert response.status_code == 200
        data = response.json()
        assert data["has_payment_method"] is True
        assert data["payment_method_added_at"] is not None

    async def test_get_payment_method_status_no_method(self, app_with_subscriptions: FastAPI, test_user):
        """Test getting payment method status when user has no payment method."""
        user, organization = test_user

        with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
            with patch("seer.api.subscriptions.setup_intent._require_organization", return_value=organization):
                async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                    response = await client.get("/api/subscriptions/payment-method/status")

        assert response.status_code == 200
        data = response.json()
        assert data["has_payment_method"] is False
        assert data["payment_method_added_at"] is None

    async def test_get_payment_method_status_no_organization(self, app_with_subscriptions: FastAPI, db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Test getting payment method status when user has no organization context."""
        from fastapi import HTTPException

        user = await User.create(
            user_id="no_billing_user",
            email="nobilling@example.com",
        )

        try:
            with patch("seer.api.subscriptions.setup_intent._require_user", return_value=user):
                with patch("seer.api.subscriptions.setup_intent._require_organization", side_effect=HTTPException(status_code=401, detail="Organization context required")):
                    async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                        response = await client.get("/api/subscriptions/payment-method/status")

            assert response.status_code == 401
            assert response.json()["detail"] == "Organization context required"
        finally:
            await user.delete()


@pytest.mark.asyncio
class TestWebhookHandler:
    """Test Setup Intent webhook handler."""

    async def test_setup_intent_succeeded_webhook(self, test_user):
        """Test processing setup_intent.succeeded webhook."""
        from seer.api.subscriptions.stripe_webhook_controller import stripe_webhook_controller

        user, organization = test_user

        # Ensure payment method not set initially
        assert organization.has_payment_method is False

        # Simulate webhook event
        webhook_data = {
            "object": {
                "id": "seti_test123",
                "customer": "cus_test123",
                "status": "succeeded",
            }
        }

        await stripe_webhook_controller._handle_setup_intent_succeeded(webhook_data)

        # Verify organization updated
        await organization.refresh_from_db()
        assert organization.has_payment_method is True
        assert organization.payment_method_added_at is not None

    async def test_setup_intent_succeeded_webhook_no_customer(self):
        """Test webhook with missing customer ID."""
        from seer.api.subscriptions.stripe_webhook_controller import stripe_webhook_controller

        webhook_data = {
            "object": {
                "id": "seti_test123",
                "status": "succeeded",
                # Missing customer
            }
        }

        # Should not raise exception, just log warning
        await stripe_webhook_controller._handle_setup_intent_succeeded(webhook_data)

    async def test_setup_intent_succeeded_webhook_no_organization(self, db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Test webhook for non-existent organization."""
        from seer.api.subscriptions.stripe_webhook_controller import stripe_webhook_controller

        webhook_data = {
            "object": {
                "id": "seti_test123",
                "customer": "cus_nonexistent",
                "status": "succeeded",
            }
        }

        # Should not raise exception, just log warning
        await stripe_webhook_controller._handle_setup_intent_succeeded(webhook_data)


@pytest.mark.asyncio
class TestStripeConfig:
    """Test Stripe config endpoint."""

    async def test_get_stripe_config_success(self, app_with_subscriptions: FastAPI):
        """Test getting Stripe publishable key."""
        with patch("seer.api.subscriptions.router.config") as mock_config:
            mock_config.stripe_publishable_key = "pk_test_123"

            async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                response = await client.get("/api/subscriptions/config")

        assert response.status_code == 200
        data = response.json()
        assert data["publishable_key"] == "pk_test_123"

    async def test_get_stripe_config_not_configured(self, app_with_subscriptions: FastAPI):
        """Test getting Stripe config when not configured."""
        with patch("seer.api.subscriptions.router.config") as mock_config:
            mock_config.stripe_publishable_key = None

            async with AsyncClient(transport=ASGITransport(app=app_with_subscriptions), base_url="http://test") as client:
                response = await client.get("/api/subscriptions/config")

        assert response.status_code == 503
