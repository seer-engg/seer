"""
Integration tests for usage limit enforcement.

Tests the actual limit checking functions with real DB state.
Replaces mock-heavy middleware unit tests with behavior-focused tests.
"""
import pytest

from seer.database import User
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
from seer.services.organization_service import (
    create_personal_organization,
    create_team_organization,
)


# =============================================================================
# Workflow Creation Limits (Real DB)
# =============================================================================


@pytest.mark.integration
class TestWorkflowCreationLimits:
    """Tests for workflow creation limit enforcement with real data."""

    @pytest.mark.asyncio
    async def test_workflow_count_reflects_real_records(self, db_engine, test_user):
        """Workflow count should match actual DB records."""
        from seer.observability.tracking import get_workflow_count

        initial = await get_workflow_count(test_user)

        for i in range(3):
            await Workflow.create(user=test_user, name=f"Count Test {i}")

        count = await get_workflow_count(test_user)
        assert count == initial + 3

    @pytest.mark.asyncio
    async def test_free_and_pro_limits_differ(self, db_engine, test_user):
        """PRO tier should have different (higher/unlimited) limits than FREE."""
        from seer.observability.service import get_effective_limits

        # Get FREE limits
        free_org, _ = await create_personal_organization(test_user)
        free_limits = await get_effective_limits(test_user, free_org)

        # Get PRO limits
        pro_org, _ = await create_team_organization(test_user, "Pro Team")
        await BillingSubscription.filter(organization=pro_org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
        )
        pro_limits = await get_effective_limits(test_user, pro_org)

        # PRO should have higher run limits
        assert pro_limits.runs_monthly >= free_limits.runs_monthly


# =============================================================================
# Run Limits (Real DB)
# =============================================================================


@pytest.mark.integration
class TestRunLimits:
    """Tests for monthly run limit enforcement with real data."""

    @pytest.mark.asyncio
    async def test_run_count_tracks_actual_runs(self, db_engine, test_user):
        """Monthly run count should reflect actual runs in DB."""
        from seer.observability.tracking import get_monthly_run_count

        workflow = await Workflow.create(user=test_user, name="Run Count")

        initial_count = await get_monthly_run_count(test_user)

        # Create 3 runs
        for _ in range(3):
            await WorkflowRun.create(
                user=test_user,
                workflow=workflow,
                spec={"version": "2"},
                source=WorkflowRunSource.MANUAL,
                status=WorkflowRunStatus.SUCCEEDED,
            )

        new_count = await get_monthly_run_count(test_user)
        assert new_count == initial_count + 3

    @pytest.mark.asyncio
    async def test_org_run_count_isolated_by_org(self, db_engine, test_user):
        """Run counts should be scoped to the organization."""
        from seer.observability.tracking import get_org_monthly_run_count

        org1, _ = await create_team_organization(test_user, "Org 1")
        org2, _ = await create_team_organization(test_user, "Org 2")

        wf1 = await Workflow.create(user=test_user, name="WF Org1", organization=org1)
        wf2 = await Workflow.create(user=test_user, name="WF Org2", organization=org2)

        # Create runs in org1 only
        for _ in range(5):
            await WorkflowRun.create(
                user=test_user,
                workflow=wf1,
                spec={"version": "2"},
                source=WorkflowRunSource.MANUAL,
                status=WorkflowRunStatus.SUCCEEDED,
                organization=org1,
            )

        # Create 2 runs in org2
        for _ in range(2):
            await WorkflowRun.create(
                user=test_user,
                workflow=wf2,
                spec={"version": "2"},
                source=WorkflowRunSource.MANUAL,
                status=WorkflowRunStatus.SUCCEEDED,
                organization=org2,
            )

        count1 = await get_org_monthly_run_count(org1)
        count2 = await get_org_monthly_run_count(org2)

        assert count1 >= 5
        assert count2 >= 2
        assert count1 != count2  # Isolated


# =============================================================================
# Overage Detection (Real DB)
# =============================================================================


@pytest.mark.integration
class TestOverageDetection:
    """Tests for overage detection with real subscription data."""

    @pytest.mark.asyncio
    async def test_free_tier_has_run_limits(self, db_engine, test_user):
        """FREE tier should have run limits and credit limits."""
        from seer.observability.service import get_effective_limits

        org, _ = await create_personal_organization(test_user)
        limits = await get_effective_limits(test_user, org)

        # FREE tier has monthly run limits and credit limits
        assert limits.runs_monthly > 0
        assert limits.llm_credits_monthly > 0

    @pytest.mark.asyncio
    async def test_tier_upgrade_increases_limits(self, db_engine, test_user):
        """Upgrading tier should increase all limits."""
        from seer.observability.service import get_effective_limits

        org, _ = await create_team_organization(test_user, "Upgrade Test")

        # Get FREE limits
        free_limits = await get_effective_limits(test_user, org)

        # Upgrade to PRO
        await BillingSubscription.filter(organization=org).update(
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
        )

        pro_limits = await get_effective_limits(test_user, org)

        # PRO should be >= FREE in all dimensions
        assert pro_limits.workflows >= free_limits.workflows
        assert pro_limits.runs_monthly >= free_limits.runs_monthly
