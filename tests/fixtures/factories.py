# pylint: disable=too-many-positional-arguments,no-member
# Reason: Factory functions require many parameters; enum members checked at runtime
"""
Model factories for creating test data.

Provides factory functions for creating database models with sensible defaults.
Inspired by factory_boy but simpler and tailored to our needs.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from seer.database.models import User
from seer.database.workflow_models import (
    TriggerSubscription,
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersionStatus,
)
from seer.database.subscription_models import (
    BillingProfile,
    BillingProfileType,
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)


class UserFactory:
    """Factory for creating User instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        first_name: str = "Test",
        last_name: str = "User",
        **kwargs,
    ) -> User:
        """
        Create a User instance.

        Args:
            user_id: User ID (auto-generated if not provided)
            email: User email (auto-generated if not provided)
            first_name: First name (default: "Test")
            last_name: Last name (default: "User")
            **kwargs: Additional User fields

        Returns:
            Created User instance
        """
        cls._counter += 1

        if user_id is None:
            user_id = f"user_{cls._counter}"

        if email is None:
            email = f"user{cls._counter}@example.com"

        return await User.create(
            user_id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            created_at=datetime.now(timezone.utc),
            **kwargs,
        )


class WorkflowFactory:
    """Factory for creating Workflow instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        user: User,
        workflow_id: Optional[str] = None,
        name: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
        status: WorkflowVersionStatus = WorkflowVersionStatus.PUBLISHED,
        **kwargs,
    ) -> Workflow:
        """
        Create a Workflow instance.

        Args:
            user: Workflow owner
            workflow_id: Workflow ID (auto-generated if not provided)
            name: Workflow name (auto-generated if not provided)
            spec: Workflow spec (minimal spec if not provided)
            status: Workflow status (default: PUBLISHED)
            **kwargs: Additional Workflow fields

        Returns:
            Created Workflow instance
        """
        cls._counter += 1

        if workflow_id is None:
            workflow_id = f"wf_{cls._counter}"

        if name is None:
            name = f"Test Workflow {cls._counter}"

        if spec is None:
            # Create minimal valid spec
            spec = {
                "version": "2",
                "triggers": [
                    {
                        "id": "t1",
                        "key": "test.trigger",
                        "label": "Test",
                        "config": {},
                    }
                ],
                "nodes": [
                    {
                        "id": "n1",
                        "type": "tool",
                        "tool": "test.tool",
                        "inputs": {},
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "t1", "target": "n1"}
                ],
            }

        return await Workflow.create(
            user=user,
            workflow_id=workflow_id,
            name=name,
            spec=spec,
            status=status,
            **kwargs,
        )


class WorkflowRunFactory:
    """Factory for creating WorkflowRun instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        workflow: Workflow,
        run_id: Optional[str] = None,
        status: WorkflowRunStatus = WorkflowRunStatus.PENDING,
        source: WorkflowRunSource = WorkflowRunSource.MANUAL,
        initial_state: Optional[Dict[str, Any]] = None,
        final_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> WorkflowRun:
        """
        Create a WorkflowRun instance.

        Args:
            workflow: Associated workflow
            run_id: Run ID (auto-generated if not provided)
            status: Run status (default: PENDING)
            source: Run source (default: MANUAL)
            initial_state: Initial state (empty dict if not provided)
            final_state: Final state (None if not provided)
            **kwargs: Additional WorkflowRun fields

        Returns:
            Created WorkflowRun instance
        """
        cls._counter += 1

        if run_id is None:
            run_id = f"run_{cls._counter}"

        if initial_state is None:
            initial_state = {}

        return await WorkflowRun.create(
            workflow=workflow,
            run_id=run_id,
            status=status,
            source=source,
            initial_state=initial_state,
            final_state=final_state,
            **kwargs,
        )


class TriggerSubscriptionFactory:
    """Factory for creating TriggerSubscription instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        workflow: Workflow,
        subscription_id: Optional[str] = None,
        trigger_id: str = "t1",
        trigger_key: str = "test.trigger",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TriggerSubscription:
        """
        Create a TriggerSubscription instance.

        Args:
            workflow: Associated workflow
            subscription_id: Subscription ID (auto-generated if not provided)
            trigger_id: Trigger ID (default: "t1")
            trigger_key: Trigger key (default: "test.trigger")
            config: Trigger config (empty dict if not provided)
            **kwargs: Additional TriggerSubscription fields

        Returns:
            Created TriggerSubscription instance
        """
        cls._counter += 1

        if subscription_id is None:
            subscription_id = f"sub_{cls._counter}"

        if config is None:
            config = {}

        return await TriggerSubscription.create(
            workflow=workflow,
            subscription_id=subscription_id,
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            config=config,
            **kwargs,
        )


class BillingProfileFactory:
    """Factory for creating BillingProfile instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        user: User,
        profile_type: BillingProfileType = BillingProfileType.STRIPE,
        stripe_customer_id: Optional[str] = None,
        **kwargs,
    ) -> BillingProfile:
        """
        Create a BillingProfile instance.

        Args:
            user: Associated user
            profile_type: Profile type (default: STRIPE)
            stripe_customer_id: Stripe customer ID (auto-generated if not provided)
            **kwargs: Additional BillingProfile fields

        Returns:
            Created BillingProfile instance
        """
        cls._counter += 1

        if stripe_customer_id is None:
            stripe_customer_id = f"cus_test_{cls._counter}"

        return await BillingProfile.create(
            user=user,
            profile_type=profile_type,
            stripe_customer_id=stripe_customer_id,
            **kwargs,
        )


class BillingSubscriptionFactory:
    """Factory for creating BillingSubscription instances."""

    _counter = 0

    @classmethod
    async def create(
        cls,
        billing_profile: BillingProfile,
        subscription_id: Optional[str] = None,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
        current_period_start: Optional[datetime] = None,
        current_period_end: Optional[datetime] = None,
        **kwargs,
    ) -> BillingSubscription:
        """
        Create a BillingSubscription instance.

        Args:
            billing_profile: Associated billing profile
            subscription_id: Subscription ID (auto-generated if not provided)
            tier: Subscription tier (default: PRO)
            status: Subscription status (default: ACTIVE)
            current_period_start: Period start (now if not provided)
            current_period_end: Period end (30 days from start if not provided)
            **kwargs: Additional BillingSubscription fields

        Returns:
            Created BillingSubscription instance
        """
        cls._counter += 1

        if subscription_id is None:
            subscription_id = f"sub_test_{cls._counter}"

        if current_period_start is None:
            current_period_start = datetime.now(timezone.utc)

        if current_period_end is None:
            current_period_end = current_period_start + timedelta(days=30)

        return await BillingSubscription.create(
            billing_profile=billing_profile,
            subscription_id=subscription_id,
            tier=tier,
            status=status,
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            **kwargs,
        )
