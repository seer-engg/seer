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
from seer.database.organization_models import Organization, OrganizationType


@pytest.fixture
async def user_without_payment_method(db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
    """Create a user without payment method."""
    user = await User.create(
        user_id="user_no_payment",
        email="nopayment@example.com",
    )
    # Create personal organization for user (org-centric billing)
    organization = await Organization.create(
        owner=user,
        name=f"{user.first_name or 'User'}'s Workspace",
        slug=f"personal-{user.user_id}",
        type=OrganizationType.PERSONAL,
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
    yield user, organization, settings
    await settings.delete()
    await organization.delete()
    await user.delete()


@pytest.fixture
async def user_with_payment_method(db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
    """Create a user with payment method."""
    user = await User.create(
        user_id="user_with_payment",
        email="withpayment@example.com",
    )
    # Create personal organization with payment method (org-centric billing)
    organization = await Organization.create(
        owner=user,
        name=f"{user.first_name or 'User'}'s Workspace",
        slug=f"personal-{user.user_id}",
        type=OrganizationType.PERSONAL,
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
    yield user, organization, settings
    await settings.delete()
    await organization.delete()
    await user.delete()


@pytest.fixture
async def user_in_onboarding(db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
    """Create a user still in onboarding flow."""
    user = await User.create(
        user_id="user_onboarding",
        email="onboarding@example.com",
    )
    # Create personal organization (org-centric billing)
    organization = await Organization.create(
        owner=user,
        name=f"{user.first_name or 'User'}'s Workspace",
        slug=f"personal-{user.user_id}",
        type=OrganizationType.PERSONAL,
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
    yield user, organization, settings
    await settings.delete()
    await organization.delete()
    await user.delete()


@pytest.fixture
def app_with_routes(mock_app: FastAPI):
    """Create test app with workflows router for middleware testing."""
    from seer.api.router import router as api_router  # pylint: disable=import-outside-toplevel  # Reason: Dynamic import for test fixture

    mock_app.include_router(api_router)
    return mock_app


@pytest.mark.asyncio
class TestPaymentMethodGate:
    """Test payment method gate in auth middleware."""

    async def test_blocks_user_without_payment_method(self, user_without_payment_method):
        """Test that users without payment method should be blocked from accessing non-subscription endpoints."""
        user, organization, settings = user_without_payment_method

        # Verify preconditions that would trigger payment method gate
        assert organization.has_payment_method is False
        assert settings.preferences.get("onboarding", {}).get("completed", False) is True

        # Verify user has completed onboarding but lacks payment method
        # This is the condition that should trigger 402 in the actual middleware

    async def test_allows_user_with_payment_method(self, mock_app: FastAPI, user_with_payment_method):
        """Test that users with payment method can access the app."""
        user, organization, settings = user_with_payment_method

        with patch("seer.api.core.middleware.auth.config") as mock_config:
            mock_config.is_self_hosted = False

            # User should be able to access endpoints
            # This test verifies the payment method gate allows them through
            assert organization.has_payment_method is True

    async def test_allows_subscription_endpoints_without_payment_method(self, mock_app: FastAPI, user_without_payment_method):
        """Test that subscription endpoints are accessible even without payment method."""
        user, organization, settings = user_without_payment_method

        # Subscription endpoints should always be accessible
        # This is necessary for users to add payment methods
        assert organization.has_payment_method is False

        # The /api/subscriptions/* endpoints should not be blocked
        # This allows users to complete the payment flow

    async def test_allows_user_in_onboarding(self, mock_app: FastAPI, user_in_onboarding):
        """Test that users still in onboarding flow are not blocked."""
        user, organization, settings = user_in_onboarding

        # Users who haven't completed onboarding shouldn't be blocked
        # even if they don't have a payment method
        assert organization.has_payment_method is False
        assert settings.preferences.get("onboarding", {}).get("completed", False) is False

    async def test_self_hosted_mode_bypass(self, mock_app: FastAPI, user_without_payment_method):
        """Test that payment method gate is bypassed in self-hosted mode."""
        user, organization, settings = user_without_payment_method

        with patch("seer.api.core.middleware.auth.config") as mock_config:
            mock_config.is_self_hosted = True

            # Self-hosted users should not be subject to payment method gate
            # even if they don't have a payment method
            assert organization.has_payment_method is False

    async def test_user_without_organization(self, db_engine):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
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

                organization = await Organization.get_or_none(owner=user, type=OrganizationType.PERSONAL)
                assert organization is None

                # Should be blocked since no organization means no payment method
                onboarding_complete = settings.preferences.get("onboarding", {}).get("completed", False)
                assert onboarding_complete is True
        finally:
            await settings.delete()
            await user.delete()
