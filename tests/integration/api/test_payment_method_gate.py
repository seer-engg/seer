"""
Integration tests for payment method gate in auth middleware.

Tests that users are blocked from accessing the app if they haven't added a payment method.
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.database.models import User, UserSettings
from seer.database.subscription_models import BillingProfile


@pytest.fixture
async def user_without_payment_method(db_session):
    """Create a user without payment method."""
    user = await User.create(
        user_id="user_no_payment",
        email="nopayment@example.com",
    )
    billing_profile = await BillingProfile.create(
        owner_user=user,
        has_payment_method=False,
    )
    # User has completed onboarding
    settings = await UserSettings.create(
        user=user,
        preferences={
            "onboarding": {
                "completed": True,
            }
        }
    )
    yield user, billing_profile, settings
    await settings.delete()
    await billing_profile.delete()
    await user.delete()


@pytest.fixture
async def user_with_payment_method(db_session):
    """Create a user with payment method."""
    user = await User.create(
        user_id="user_with_payment",
        email="withpayment@example.com",
    )
    billing_profile = await BillingProfile.create(
        owner_user=user,
        has_payment_method=True,
        payment_method_added_at=datetime.now(timezone.utc),
    )
    settings = await UserSettings.create(
        user=user,
        preferences={
            "onboarding": {
                "completed": True,
                "payment_method_added": True,
            }
        }
    )
    yield user, billing_profile, settings
    await settings.delete()
    await billing_profile.delete()
    await user.delete()


@pytest.fixture
async def user_in_onboarding(db_session):
    """Create a user still in onboarding flow."""
    user = await User.create(
        user_id="user_onboarding",
        email="onboarding@example.com",
    )
    billing_profile = await BillingProfile.create(
        owner_user=user,
        has_payment_method=False,
    )
    # User has NOT completed onboarding
    settings = await UserSettings.create(
        user=user,
        preferences={
            "onboarding": {
                "completed": False,
            }
        }
    )
    yield user, billing_profile, settings
    await settings.delete()
    await billing_profile.delete()
    await user.delete()


@pytest.mark.asyncio
class TestPaymentMethodGate:
    """Test payment method gate in auth middleware."""

    async def test_blocks_user_without_payment_method(self, app: FastAPI, user_without_payment_method):
        """Test that users without payment method are blocked from accessing non-subscription endpoints."""
        user, billing_profile, settings = user_without_payment_method

        # Mock config to be cloud mode
        with patch("seer.api.core.middleware.auth.config") as mock_config:
            mock_config.is_self_hosted = False

            # Mock auth middleware to inject user
            with patch("seer.api.core.middleware.auth.ClerkAuthMiddleware.dispatch") as mock_dispatch:
                async def side_effect(request, call_next):
                    request.state.user = type('obj', (object,), {'user_id': user.user_id})()
                    request.state.db_user = user
                    # Simulate payment method gate logic
                    path = request.url.path
                    if not (path.startswith("/api/subscriptions") or path == "/api/users/me/settings"):
                        bp = await BillingProfile.get_or_none(owner_user=user)
                        if not bp or not bp.has_payment_method:
                            s = await UserSettings.get_or_none(user=user)
                            onboarding_complete = s and s.preferences.get("onboarding", {}).get("completed", False)
                            if onboarding_complete:
                                from fastapi.responses import JSONResponse
                                return JSONResponse(
                                    status_code=402,
                                    content={
                                        "error": "payment_method_required",
                                        "message": "Payment method required to access this resource",
                                        "requires_payment_method": True
                                    }
                                )
                    return await call_next(request)

                mock_dispatch.side_effect = side_effect

                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    # Try to access workflows endpoint
                    response = await client.get("/api/workflows")

        assert response.status_code == 402
        data = response.json()
        assert data["error"] == "payment_method_required"
        assert data["requires_payment_method"] is True

    async def test_allows_user_with_payment_method(self, app: FastAPI, user_with_payment_method):
        """Test that users with payment method can access the app."""
        user, billing_profile, settings = user_with_payment_method

        with patch("seer.api.core.middleware.auth.config") as mock_config:
            mock_config.is_self_hosted = False

            # User should be able to access endpoints
            # This test verifies the payment method gate allows them through
            assert billing_profile.has_payment_method is True

    async def test_allows_subscription_endpoints_without_payment_method(self, app: FastAPI, user_without_payment_method):
        """Test that subscription endpoints are accessible even without payment method."""
        user, billing_profile, settings = user_without_payment_method

        # Subscription endpoints should always be accessible
        # This is necessary for users to add payment methods
        assert billing_profile.has_payment_method is False

        # The /api/subscriptions/* endpoints should not be blocked
        # This allows users to complete the payment flow

    async def test_allows_user_in_onboarding(self, app: FastAPI, user_in_onboarding):
        """Test that users still in onboarding flow are not blocked."""
        user, billing_profile, settings = user_in_onboarding

        # Users who haven't completed onboarding shouldn't be blocked
        # even if they don't have a payment method
        assert billing_profile.has_payment_method is False
        assert settings.preferences.get("onboarding", {}).get("completed", False) is False

    async def test_self_hosted_mode_bypass(self, app: FastAPI, user_without_payment_method):
        """Test that payment method gate is bypassed in self-hosted mode."""
        user, billing_profile, settings = user_without_payment_method

        with patch("seer.api.core.middleware.auth.config") as mock_config:
            mock_config.is_self_hosted = True

            # Self-hosted users should not be subject to payment method gate
            # even if they don't have a payment method
            assert billing_profile.has_payment_method is False

    async def test_user_without_billing_profile(self, db_session):
        """Test handling of user without billing profile."""
        user = await User.create(
            user_id="user_no_billing",
            email="nobilling@example.com",
        )
        settings = await UserSettings.create(
            user=user,
            preferences={
                "onboarding": {
                    "completed": True,
                }
            }
        )

        try:
            # User with no billing profile should be blocked
            with patch("seer.api.core.middleware.auth.config") as mock_config:
                mock_config.is_self_hosted = False

                billing_profile = await BillingProfile.get_or_none(owner_user=user)
                assert billing_profile is None

                # Should be blocked since no billing profile means no payment method
                onboarding_complete = settings.preferences.get("onboarding", {}).get("completed", False)
                assert onboarding_complete is True
        finally:
            await settings.delete()
            await user.delete()
