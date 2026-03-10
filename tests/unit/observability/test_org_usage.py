"""
Unit tests for organization-aware usage tracking.

Tests cover:
- Organization-scoped tier/limits resolution
- Effective tier/limits/billing functions
- Org-aware usage tracking
- Org-level query functions
"""
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from seer.database.organization_models import Organization, OrganizationType
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.database.usage_models import LLMUsageRecord, ResourceType, UsageCounter
from seer.observability.models import TierLimits


pytestmark = pytest.mark.asyncio


# =============================================================================
# Service Layer Tests
# =============================================================================


class TestResolveOrgTier:
    """Tests for resolve_org_tier function.

    With org-centric billing, resolve_org_tier queries BillingSubscription
    directly via organization FK (no BillingProfile indirection).
    """

    async def test_resolve_org_tier_with_pro_subscription(self, mocker):
        """Org with PRO subscription returns PRO tier."""
        from seer.observability.service import resolve_org_tier

        # Create mock organization
        org = MagicMock(spec=Organization)
        org.id = 1

        # Create mock subscription directly linked to org
        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.ACTIVE

        # Mock direct org -> subscription lookup
        mocker.patch.object(
            BillingSubscription,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=mock_subscription,
        )

        tier = await resolve_org_tier(org)

        assert tier == SubscriptionTier.PRO
        BillingSubscription.get_or_none.assert_called_once_with(organization=org)

    async def test_resolve_org_tier_no_subscription_returns_free(self, mocker):
        """Org without subscription returns FREE tier."""
        from seer.observability.service import resolve_org_tier

        org = MagicMock(spec=Organization)
        org.id = 1

        # No subscription for org
        mocker.patch.object(
            BillingSubscription,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=None,
        )

        tier = await resolve_org_tier(org)

        assert tier == SubscriptionTier.FREE

    async def test_resolve_org_tier_canceled_subscription_returns_free(self, mocker):
        """Org with canceled subscription returns FREE tier."""
        from seer.observability.service import resolve_org_tier

        org = MagicMock(spec=Organization)
        org.id = 1

        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.CANCELED

        mocker.patch.object(
            BillingSubscription,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=mock_subscription,
        )

        tier = await resolve_org_tier(org)

        assert tier == SubscriptionTier.FREE

    async def test_resolve_org_tier_trialing_subscription_returns_tier(self, mocker):
        """Org with trialing subscription returns its tier."""
        from seer.observability.service import resolve_org_tier

        org = MagicMock(spec=Organization)
        org.id = 1

        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.TRIALING

        mocker.patch.object(
            BillingSubscription,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=mock_subscription,
        )

        tier = await resolve_org_tier(org)

        assert tier == SubscriptionTier.PRO

    async def test_resolve_org_tier_incomplete_subscription_returns_free(self, mocker):
        """Org with incomplete subscription returns FREE tier."""
        from seer.observability.service import resolve_org_tier

        org = MagicMock(spec=Organization)
        org.id = 1

        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.INCOMPLETE

        mocker.patch.object(
            BillingSubscription,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=mock_subscription,
        )

        tier = await resolve_org_tier(org)

        assert tier == SubscriptionTier.FREE


class TestGetEffectiveTier:
    """Tests for get_effective_tier function."""

    async def test_effective_tier_uses_org_for_team(self, mocker):
        """For team orgs, uses org's subscription tier."""
        from seer.observability.service import get_effective_tier

        user = MagicMock()
        org = MagicMock(spec=Organization)
        org.type = OrganizationType.TEAM

        # Mock resolve_org_tier to return PRO
        mock_resolve_org = mocker.patch(
            "seer.observability.service.resolve_org_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.PRO,
        )
        # Mock resolve_user_tier (should not be called)
        mock_resolve_user = mocker.patch(
            "seer.observability.service.resolve_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.FREE,
        )

        tier = await get_effective_tier(user, org)

        assert tier == SubscriptionTier.PRO
        mock_resolve_org.assert_called_once_with(org)
        mock_resolve_user.assert_not_called()

    async def test_effective_tier_uses_user_for_personal_org(self, mocker):
        """For personal orgs, uses user's subscription tier."""
        from seer.observability.service import get_effective_tier

        user = MagicMock()
        org = MagicMock(spec=Organization)
        org.type = OrganizationType.PERSONAL

        mock_resolve_org = mocker.patch(
            "seer.observability.service.resolve_org_tier",
            new_callable=AsyncMock,
        )
        mock_resolve_user = mocker.patch(
            "seer.observability.service.resolve_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.FREE,
        )

        tier = await get_effective_tier(user, org)

        assert tier == SubscriptionTier.FREE
        mock_resolve_org.assert_not_called()
        mock_resolve_user.assert_called_once_with(user)

    async def test_effective_tier_uses_user_when_no_org(self, mocker):
        """When no org provided, uses user's subscription tier."""
        from seer.observability.service import get_effective_tier

        user = MagicMock()

        mock_resolve_user = mocker.patch(
            "seer.observability.service.resolve_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.PRO,
        )

        tier = await get_effective_tier(user, None)

        assert tier == SubscriptionTier.PRO
        mock_resolve_user.assert_called_once_with(user)


class TestGetEffectiveLimits:
    """Tests for get_effective_limits function."""

    async def test_effective_limits_uses_org_for_team(self, mocker):
        """For team orgs, uses org's subscription limits."""
        from seer.observability.service import get_effective_limits

        user = MagicMock()
        org = MagicMock(spec=Organization)
        org.type = OrganizationType.TEAM

        mock_org_limits = MagicMock(spec=TierLimits)
        mock_user_limits = MagicMock(spec=TierLimits)

        mocker.patch(
            "seer.observability.service.get_limits_for_org",
            new_callable=AsyncMock,
            return_value=mock_org_limits,
        )
        mocker.patch(
            "seer.observability.service.get_limits_for_user",
            new_callable=AsyncMock,
            return_value=mock_user_limits,
        )

        limits = await get_effective_limits(user, org)

        assert limits == mock_org_limits


# =============================================================================
# Tracking Layer Tests
# =============================================================================


class TestTrackLLMUsageOrg:
    """Tests for track_llm_usage with organization parameter."""

    async def test_track_llm_usage_creates_record_with_org(self, mocker):
        """track_llm_usage creates LLMUsageRecord with organization FK set."""
        from seer.observability.tracking import track_llm_usage

        user = MagicMock()
        user.user_id = "test-user"
        org = MagicMock(spec=Organization)
        org.id = 123
        org.type = OrganizationType.TEAM

        mock_record = MagicMock(spec=LLMUsageRecord)

        mocker.patch.object(
            LLMUsageRecord,
            "create",
            new_callable=AsyncMock,
            return_value=mock_record,
        )
        mocker.patch(
            "seer.observability.tracking.get_effective_billing_period",
            new_callable=AsyncMock,
            return_value=(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
        )
        mocker.patch.object(
            UsageCounter,
            "get_or_create",
            new_callable=AsyncMock,
            return_value=(MagicMock(id=1), True),
        )
        mocker.patch.object(
            UsageCounter,
            "filter",
            return_value=MagicMock(update=AsyncMock()),
        )
        mocker.patch(
            "seer.observability.tracking._handle_potential_overage",
            new_callable=AsyncMock,
        )

        result = await track_llm_usage(
            user=user,
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=Decimal("0.01"),
            organization=org,
        )

        assert result == mock_record

        # Verify LLMUsageRecord.create was called with organization
        create_call = LLMUsageRecord.create.call_args
        assert create_call.kwargs.get("organization") == org

    async def test_track_llm_usage_updates_dual_counters_for_team_org(self, mocker):
        """For team orgs, updates both user-in-org and org-level counters."""
        from seer.observability.tracking import track_llm_usage

        user = MagicMock()
        user.user_id = "test-user"
        org = MagicMock(spec=Organization)
        org.id = 123
        org.type = OrganizationType.TEAM

        mocker.patch.object(
            LLMUsageRecord,
            "create",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        )
        mocker.patch(
            "seer.observability.tracking.get_effective_billing_period",
            new_callable=AsyncMock,
            return_value=(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
        )

        counter_calls = []

        async def mock_get_or_create(**kwargs):
            counter_calls.append(kwargs)
            return (MagicMock(id=len(counter_calls)), True)

        mocker.patch.object(
            UsageCounter,
            "get_or_create",
            side_effect=mock_get_or_create,
        )
        mocker.patch.object(
            UsageCounter,
            "filter",
            return_value=MagicMock(update=AsyncMock()),
        )
        mocker.patch(
            "seer.observability.tracking._handle_potential_overage",
            new_callable=AsyncMock,
        )

        await track_llm_usage(
            user=user,
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=Decimal("0.01"),
            organization=org,
        )

        # Should have two counter calls: user-in-org and org-level
        assert len(counter_calls) == 2

        # First call: user-in-org counter
        assert counter_calls[0]["user"] == user
        assert counter_calls[0]["organization"] == org

        # Second call: org-level counter (user=None)
        assert counter_calls[1]["user"] is None
        assert counter_calls[1]["organization"] == org


# =============================================================================
# Credit Gate Tests
# =============================================================================


class TestCheckCreditLimitOrg:
    """Tests for check_credit_limit with organization parameter."""

    async def test_check_credit_limit_uses_org_limits_for_team(self, mocker):
        """For team orgs, checks against org's limits."""
        from seer.observability.credit_gate import check_credit_limit

        user = MagicMock()
        org = MagicMock(spec=Organization)
        org.id = 123
        org.type = OrganizationType.TEAM

        mock_limits = MagicMock(spec=TierLimits)
        mock_limits.has_unlimited_credits = True  # Bypass all checks

        mocker.patch(
            "seer.observability.credit_gate.get_effective_limits",
            new_callable=AsyncMock,
            return_value=mock_limits,
        )

        # Should not raise
        await check_credit_limit(user, org)

    async def test_check_credit_limit_uses_org_usage_for_team(self, mocker):
        """For team orgs, checks org-level usage against limits."""
        from seer.observability.credit_gate import check_credit_limit
        from seer.observability.exceptions import CreditLimitExceeded

        user = MagicMock()
        org = MagicMock(spec=Organization)
        org.id = 123
        org.type = OrganizationType.TEAM

        mock_limits = MagicMock(spec=TierLimits)
        mock_limits.has_unlimited_credits = False
        mock_limits.has_unlimited_5h_credits = True
        mock_limits.has_unlimited_weekly_credits = True
        mock_limits.llm_credits_monthly = 10.0

        mocker.patch(
            "seer.observability.credit_gate.get_effective_limits",
            new_callable=AsyncMock,
            return_value=mock_limits,
        )
        mocker.patch(
            "seer.observability.credit_gate.get_effective_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.PRO,
        )
        # Return usage over 120% of limit
        mocker.patch(
            "seer.observability.credit_gate.get_org_monthly_llm_credits_used",
            new_callable=AsyncMock,
            return_value=Decimal("15.0"),  # 150% of $10 limit
        )
        mocker.patch(
            "seer.observability.credit_gate._get_effective_overage_settings",
            new_callable=AsyncMock,
            return_value=None,  # No overage enabled
        )

        with pytest.raises(CreditLimitExceeded) as exc_info:
            await check_credit_limit(user, org)

        assert exc_info.value.limit == 10.0
        assert exc_info.value.current == 15.0


# =============================================================================
# Org Query Function Tests
# =============================================================================


class TestOrgQueryFunctions:
    """Tests for organization-scoped query functions."""

    async def test_get_org_workflow_count(self, mocker):
        """get_org_workflow_count queries workflows by organization."""
        from seer.observability.tracking import get_org_workflow_count
        from seer.database import Workflow

        org = MagicMock(spec=Organization)
        org.id = 123

        mock_filter = mocker.patch.object(
            Workflow,
            "filter",
            return_value=MagicMock(count=AsyncMock(return_value=5)),
        )

        count = await get_org_workflow_count(org)

        assert count == 5
        mock_filter.assert_called_once_with(organization=org)

    async def test_get_org_monthly_llm_credits_used(self, mocker):
        """get_org_monthly_llm_credits_used queries org-level counter."""
        from seer.observability.tracking import get_org_monthly_llm_credits_used

        org = MagicMock(spec=Organization)
        org.id = 123

        mock_counter = MagicMock()
        mock_counter.value = Decimal("25.50")

        mocker.patch(
            "seer.observability.tracking.get_billing_period_for_org",
            new_callable=AsyncMock,
            return_value=(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
        )
        mocker.patch.object(
            UsageCounter,
            "get_or_none",
            new_callable=AsyncMock,
            return_value=mock_counter,
        )

        credits = await get_org_monthly_llm_credits_used(org)

        assert credits == Decimal("25.50")

        # Verify counter was queried with user=None (org-level)
        counter_call = UsageCounter.get_or_none.call_args
        assert counter_call.kwargs.get("user") is None
        assert counter_call.kwargs.get("organization") == org


# =============================================================================
# Subscription Lookup Tests
# =============================================================================


class TestGetSubscriptionForOrg:
    """Tests for get_subscription_for_org function.

    With org-centric billing, get_subscription_for_org queries
    BillingSubscription directly via organization FK.
    """

    async def test_creates_subscription_if_missing(self, mocker):
        """get_subscription_for_org creates subscription if missing."""
        from seer.observability.service import get_subscription_for_org

        org = MagicMock(spec=Organization)
        org.id = 123

        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.FREE
        mock_subscription.status = SubscriptionStatus.ACTIVE

        # Mock get_or_create for subscription directly on organization
        mocker.patch.object(
            BillingSubscription,
            "get_or_create",
            new_callable=AsyncMock,
            return_value=(mock_subscription, True),  # Created new subscription
        )

        # Mock config
        mock_config = MagicMock()
        mock_config.is_self_hosted = False
        mocker.patch("seer.observability.service.config", mock_config)

        subscription = await get_subscription_for_org(org)

        assert subscription == mock_subscription
        assert subscription.tier == SubscriptionTier.FREE

        # Verify subscription was created with organization FK and FREE tier
        sub_call = BillingSubscription.get_or_create.call_args
        assert sub_call.kwargs.get("organization") == org
        assert sub_call.kwargs.get("defaults")["tier"] == SubscriptionTier.FREE
        assert sub_call.kwargs.get("defaults")["status"] == SubscriptionStatus.ACTIVE

    async def test_returns_existing_subscription(self, mocker):
        """get_subscription_for_org returns existing subscription without creating."""
        from seer.observability.service import get_subscription_for_org

        org = MagicMock(spec=Organization)
        org.id = 123

        mock_subscription = MagicMock(spec=BillingSubscription)
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.ACTIVE

        # Mock get_or_create for subscription (already exists)
        mocker.patch.object(
            BillingSubscription,
            "get_or_create",
            new_callable=AsyncMock,
            return_value=(mock_subscription, False),  # Already existed
        )

        mock_config = MagicMock()
        mock_config.is_self_hosted = False
        mocker.patch("seer.observability.service.config", mock_config)

        subscription = await get_subscription_for_org(org)

        assert subscription == mock_subscription
        assert subscription.tier == SubscriptionTier.PRO

    async def test_returns_none_for_self_hosted(self, mocker):
        """get_subscription_for_org returns None in self-hosted mode."""
        from seer.observability.service import get_subscription_for_org

        org = MagicMock(spec=Organization)
        org.id = 123

        mock_config = MagicMock()
        mock_config.is_self_hosted = True
        mocker.patch("seer.observability.service.config", mock_config)

        subscription = await get_subscription_for_org(org)

        assert subscription is None
