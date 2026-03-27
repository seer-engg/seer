"""
Integration tests for organization-aware usage tracking.

Replaces mock-heavy unit tests with real DB operations.
Tests: tier resolution, effective limits, workflow/run counting.
"""
import pytest

from seer.database import Organization, User
from seer.database.organization_models import OrganizationType
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.database.workflow_models import (
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
)
from seer.observability.models import TierLimits
from seer.services.organization_service import (
    create_personal_organization,
    create_team_organization,
)


# =============================================================================
# Tier Resolution (Real DB)
# =============================================================================


@pytest.mark.integration
class TestResolveOrgTier:
    """Tests for resolve_org_tier with real database subscriptions."""

    @pytest.mark.asyncio
    async def test_pro_subscription_returns_pro(self, db_engine, test_user):
        """Org with active PRO subscription should resolve to PRO tier."""
        from seer.observability.service import resolve_org_tier

        org, _ = await create_team_organization(test_user, "Pro Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
        )

        tier = await resolve_org_tier(org)
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_no_subscription_returns_free(self, db_engine, test_user):
        """Org with deleted subscription should resolve to FREE tier."""
        from seer.observability.service import resolve_org_tier

        org, _ = await create_team_organization(test_user, "No Sub Team")
        # Delete the auto-created subscription
        await BillingSubscription.filter(organization=org).delete()

        tier = await resolve_org_tier(org)
        assert tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_canceled_subscription_returns_free(self, db_engine, test_user):
        """Org with canceled subscription should resolve to FREE tier."""
        from seer.observability.service import resolve_org_tier

        org, _ = await create_team_organization(test_user, "Canceled Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.CANCELED,
        )

        tier = await resolve_org_tier(org)
        assert tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_trialing_subscription_returns_tier(self, db_engine, test_user):
        """Org with trialing subscription should return the subscribed tier."""
        from seer.observability.service import resolve_org_tier

        org, _ = await create_team_organization(test_user, "Trial Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.TRIALING,
        )

        tier = await resolve_org_tier(org)
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_incomplete_subscription_returns_free(self, db_engine, test_user):
        """Org with incomplete subscription should resolve to FREE tier."""
        from seer.observability.service import resolve_org_tier

        org, _ = await create_team_organization(test_user, "Incomplete Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.INCOMPLETE,
        )

        tier = await resolve_org_tier(org)
        assert tier == SubscriptionTier.FREE


# =============================================================================
# Effective Tier (Real DB)
# =============================================================================


@pytest.mark.integration
class TestGetEffectiveTier:
    """Tests for get_effective_tier with real database."""

    @pytest.mark.asyncio
    async def test_team_org_uses_org_tier(self, db_engine, test_user):
        """For team orgs, effective tier should come from org subscription."""
        from seer.observability.service import get_effective_tier

        org, _ = await create_team_organization(test_user, "Tier Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
        )

        tier = await get_effective_tier(test_user, org)
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_personal_org_uses_org_tier(self, db_engine, test_user):
        """For personal orgs, effective tier comes from personal org subscription."""
        from seer.observability.service import get_effective_tier

        org, _ = await create_personal_organization(test_user)

        tier = await get_effective_tier(test_user, org)
        assert tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_no_org_falls_back_to_user_tier(self, db_engine, test_user):
        """Without org context, should resolve from user's personal org."""
        from seer.observability.service import get_effective_tier

        # Create personal org (required for fallback)
        await create_personal_organization(test_user)

        tier = await get_effective_tier(test_user, None)
        assert tier == SubscriptionTier.FREE


# =============================================================================
# Effective Limits (Real DB)
# =============================================================================


@pytest.mark.integration
class TestGetEffectiveLimits:
    """Tests for get_effective_limits with real database."""

    @pytest.mark.asyncio
    async def test_free_tier_returns_free_limits(self, db_engine, test_user):
        """FREE tier should return FREE tier limits."""
        from seer.observability.service import get_effective_limits

        org, _ = await create_personal_organization(test_user)

        limits = await get_effective_limits(test_user, org)
        assert isinstance(limits, TierLimits)
        # FREE tier has defined limits (workflows may be -1 meaning unlimited)
        assert limits.runs_monthly > 0

    @pytest.mark.asyncio
    async def test_pro_tier_returns_higher_limits(self, db_engine, test_user):
        """PRO tier should return higher limits than FREE."""
        from seer.observability.service import get_effective_limits

        org, _ = await create_team_organization(test_user, "Pro Limits Team")
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
        )

        limits = await get_effective_limits(test_user, org)
        assert isinstance(limits, TierLimits)
        # PRO should have higher or unlimited workflows
        assert limits.workflows > 5 or limits.has_unlimited_workflows


# =============================================================================
# Workflow/Run Counting (Real DB)
# =============================================================================


@pytest.mark.integration
class TestUsageCounting:
    """Tests for actual usage counting with real database records."""

    @pytest.mark.asyncio
    async def test_workflow_count(self, db_engine, test_user):
        """Should count real workflows in database."""
        from seer.observability.tracking import get_workflow_count

        # Create some workflows
        for i in range(3):
            await Workflow.create(user=test_user, name=f"Count Test {i}")

        count = await get_workflow_count(test_user)
        assert count >= 3

    @pytest.mark.asyncio
    async def test_org_workflow_count(self, db_engine, test_user):
        """Should count workflows scoped to organization."""
        from seer.observability.tracking import get_org_workflow_count

        org, _ = await create_team_organization(test_user, "Count Org")

        # Create workflows in org
        for i in range(2):
            await Workflow.create(
                user=test_user,
                name=f"Org Count Test {i}",
                organization=org,
            )

        count = await get_org_workflow_count(org)
        assert count == 2

    @pytest.mark.asyncio
    async def test_monthly_run_count(self, db_engine, test_user):
        """Should count workflow runs in current month."""
        from seer.observability.tracking import get_monthly_run_count

        workflow = await Workflow.create(user=test_user, name="Run Count Test")
        for i in range(4):
            await WorkflowRun.create(
                user=test_user,
                workflow=workflow,
                spec={"version": "2"},
                source=WorkflowRunSource.MANUAL,
                status=WorkflowRunStatus.SUCCEEDED,
            )

        count = await get_monthly_run_count(test_user)
        assert count >= 4
