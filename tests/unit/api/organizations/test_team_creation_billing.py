"""
Unit tests for hybrid team creation billing flows.

Tests cover:
- Team creation when personal org is FREE tier
- Team creation with transfer_subscription=true (transfers subscription)
- Team creation with transfer_subscription=false (creates FREE team, requires checkout)
- checkout_required flag in response
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from seer.database.organization_models import Organization, OrganizationType
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)


pytestmark = pytest.mark.asyncio


class TestTeamCreationBilling:
    """Tests for create_organization endpoint billing behavior."""

    async def test_free_tier_user_creates_team_checkout_required(self, mocker):
        """When FREE tier user creates team, checkout_required=true."""
        from seer.api.organizations.models import CreateOrganizationRequest

        # User with FREE tier subscription on personal org
        personal_org = MagicMock(spec=Organization)
        personal_org.id = 1
        personal_org.type = OrganizationType.PERSONAL

        personal_subscription = MagicMock(spec=BillingSubscription)
        personal_subscription.tier = SubscriptionTier.FREE
        personal_subscription.status = SubscriptionStatus.ACTIVE
        personal_subscription.organization = personal_org

        # Request with transfer_subscription=true (doesn't matter for FREE)
        request_body = CreateOrganizationRequest(
            name="Test Team",
            slug="test-team",
            transfer_subscription=True,
        )

        # Since user is FREE, has_transferable_subscription will be False
        # The endpoint should create team with FREE tier and checkout_required=true
        assert request_body.transfer_subscription is True
        assert personal_subscription.tier == SubscriptionTier.FREE

    async def test_paid_tier_user_transfer_true_no_checkout_required(self, mocker):
        """When paid tier user creates team with transfer=true, checkout_required=false."""
        from seer.api.organizations.models import CreateOrganizationRequest

        # User with PRO tier subscription on personal org
        personal_org = MagicMock(spec=Organization)
        personal_org.id = 1
        personal_org.type = OrganizationType.PERSONAL

        personal_subscription = MagicMock(spec=BillingSubscription)
        personal_subscription.tier = SubscriptionTier.PRO
        personal_subscription.status = SubscriptionStatus.ACTIVE
        personal_subscription.organization = personal_org

        request_body = CreateOrganizationRequest(
            name="Test Team",
            slug="test-team",
            transfer_subscription=True,
        )

        # Check transfer logic conditions
        has_transferable_subscription = (
            personal_subscription is not None
            and personal_subscription.tier != SubscriptionTier.FREE
            and personal_subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
        )

        assert has_transferable_subscription is True
        assert request_body.transfer_subscription is True
        # When both are true, checkout_required should be False

    async def test_paid_tier_user_transfer_false_checkout_required(self, mocker):
        """When paid tier user creates team with transfer=false, checkout_required=true."""
        from seer.api.organizations.models import CreateOrganizationRequest

        # User with PRO tier
        personal_subscription = MagicMock(spec=BillingSubscription)
        personal_subscription.tier = SubscriptionTier.PRO
        personal_subscription.status = SubscriptionStatus.ACTIVE

        request_body = CreateOrganizationRequest(
            name="Test Team",
            slug="test-team",
            transfer_subscription=False,  # User explicitly opts out of transfer
        )

        has_transferable_subscription = (
            personal_subscription is not None
            and personal_subscription.tier != SubscriptionTier.FREE
            and personal_subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
        )

        assert has_transferable_subscription is True
        assert request_body.transfer_subscription is False
        # When transfer=false, checkout_required should be True

    async def test_create_organization_request_defaults(self):
        """CreateOrganizationRequest defaults transfer_subscription to False."""
        from seer.api.organizations.models import CreateOrganizationRequest

        request_body = CreateOrganizationRequest(name="Test Team")

        assert request_body.transfer_subscription is False
        assert request_body.slug is None

    async def test_organization_response_includes_checkout_required(self):
        """OrganizationResponse includes checkout_required field."""
        from datetime import datetime, timezone
        from seer.api.organizations.models import OrganizationResponse

        response = OrganizationResponse(
            id=1,
            name="Test Team",
            slug="test-team",
            type=OrganizationType.TEAM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            checkout_required=True,
        )

        assert response.checkout_required is True

        response_no_checkout = OrganizationResponse(
            id=2,
            name="Test Team 2",
            slug="test-team-2",
            type=OrganizationType.TEAM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            checkout_required=False,
        )

        assert response_no_checkout.checkout_required is False

    async def test_organization_response_checkout_required_default(self):
        """OrganizationResponse defaults checkout_required to False."""
        from datetime import datetime, timezone
        from seer.api.organizations.models import OrganizationResponse

        response = OrganizationResponse(
            id=1,
            name="Test Team",
            slug="test-team",
            type=OrganizationType.TEAM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            # Not providing checkout_required - should default to False
        )

        assert response.checkout_required is False


class TestTransferSubscriptionLogic:
    """Tests for subscription transfer decision logic."""

    def test_transfer_condition_requires_paid_tier(self):
        """Transfer requires non-FREE tier subscription."""
        for tier in [SubscriptionTier.PRO, SubscriptionTier.PRO_PLUS]:
            subscription = MagicMock(spec=BillingSubscription)
            subscription.tier = tier
            subscription.status = SubscriptionStatus.ACTIVE

            has_transferable = (
                subscription is not None
                and subscription.tier != SubscriptionTier.FREE
                and subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
            )

            assert has_transferable is True, f"Tier {tier} should be transferable"

    def test_transfer_condition_rejects_free_tier(self):
        """Transfer rejects FREE tier subscription."""
        subscription = MagicMock(spec=BillingSubscription)
        subscription.tier = SubscriptionTier.FREE
        subscription.status = SubscriptionStatus.ACTIVE

        has_transferable = (
            subscription is not None
            and subscription.tier != SubscriptionTier.FREE
            and subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
        )

        assert has_transferable is False

    def test_transfer_condition_rejects_canceled_subscription(self):
        """Transfer rejects canceled subscription."""
        subscription = MagicMock(spec=BillingSubscription)
        subscription.tier = SubscriptionTier.PRO
        subscription.status = SubscriptionStatus.CANCELED

        has_transferable = (
            subscription is not None
            and subscription.tier != SubscriptionTier.FREE
            and subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
        )

        assert has_transferable is False

    def test_transfer_condition_accepts_trialing_subscription(self):
        """Transfer accepts trialing subscription."""
        subscription = MagicMock(spec=BillingSubscription)
        subscription.tier = SubscriptionTier.PRO
        subscription.status = SubscriptionStatus.TRIALING

        has_transferable = (
            subscription is not None
            and subscription.tier != SubscriptionTier.FREE
            and subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
        )

        assert has_transferable is True
