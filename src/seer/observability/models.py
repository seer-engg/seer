"""
Data models for usage limits.

Defines the TierLimits dataclass and the tier limits registry that maps
subscription tiers to their specific limits.
"""
from pydantic import BaseModel, Field

from seer.database.subscription_models import SubscriptionTier
from seer.observability.constants import tiered_usage_limits as constants


class TierLimits(BaseModel):
    """
    Comprehensive limit configuration for a subscription tier.

    Attributes:
        workflows: Maximum number of workflows (-1 = unlimited)
        runs_monthly: Maximum workflow runs per month (-1 = unlimited)
        account_day_limit: Maximum days from signup (-1 = unlimited)
        poll_min_interval_seconds: Minimum polling interval in seconds
        llm_credits_monthly: Monthly LLM credit allowance in USD (-1 = unlimited/BYOK)
        llm_credits_5h: 5-hour rolling LLM credit allowance in USD (-1 = unlimited)
        llm_credits_weekly: Weekly rolling LLM credit allowance in USD (-1 = unlimited)
    """

    workflows: int = Field(description="Maximum workflows (-1 = unlimited)")
    runs_monthly: int = Field(description="Maximum runs per month (-1 = unlimited)")
    account_day_limit: int = Field(
        description="Maximum days from signup (-1 = unlimited)"
    )
    poll_min_interval_seconds: int = Field(
        description="Minimum polling interval in seconds"
    )
    llm_credits_monthly: float = Field(
        description="Monthly LLM credits in USD (-1 = unlimited/BYOK)"
    )
    llm_credits_5h: float = Field(
        description="5-hour rolling LLM credits in USD (-1 = unlimited)"
    )
    llm_credits_weekly: float = Field(
        description="Weekly rolling LLM credits in USD (-1 = unlimited)"
    )

    @property
    def has_unlimited_workflows(self) -> bool:
        """Check if workflows are unlimited."""
        return self.workflows == -1

    @property
    def has_unlimited_runs(self) -> bool:
        """Check if runs are unlimited."""
        return self.runs_monthly == -1

    @property
    def has_unlimited_credits(self) -> bool:
        """Check if LLM credits are unlimited (BYOK mode)."""
        return self.llm_credits_monthly == -1

    @property
    def has_unlimited_5h_credits(self) -> bool:
        """Check if 5-hour LLM credits are unlimited."""
        return self.llm_credits_5h == -1

    @property
    def has_unlimited_weekly_credits(self) -> bool:
        """Check if weekly LLM credits are unlimited."""
        return self.llm_credits_weekly == -1

    @property
    def has_time_limit(self) -> bool:
        """Check if account has a time limit (trial period)."""
        return self.account_day_limit > 0


# ============================================================================
# Tier Limits Registry
# ============================================================================

# Tier limits mapped to SubscriptionTier enum
TIER_LIMITS_REGISTRY: dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        workflows=constants.WORKFLOWS_FREE,
        runs_monthly=constants.RUNS_MONTHLY_FREE,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_FREE,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_FREE,
        llm_credits_monthly=constants.LLM_CREDITS_FREE,
        llm_credits_5h=constants.LLM_CREDITS_5H_FREE,
        llm_credits_weekly=constants.LLM_CREDITS_WEEKLY_FREE,
    ),
    SubscriptionTier.PRO: TierLimits(
        workflows=constants.WORKFLOWS_PRO,
        runs_monthly=constants.RUNS_MONTHLY_PRO,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_PRO,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_PRO,
        llm_credits_monthly=constants.LLM_CREDITS_PRO,
        llm_credits_5h=constants.LLM_CREDITS_5H_PRO,
        llm_credits_weekly=constants.LLM_CREDITS_WEEKLY_PRO,
    ),
    SubscriptionTier.PRO_PLUS: TierLimits(
        workflows=constants.WORKFLOWS_PRO_PLUS,
        runs_monthly=constants.RUNS_MONTHLY_PRO_PLUS,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_PRO_PLUS,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_PRO_PLUS,
        llm_credits_monthly=constants.LLM_CREDITS_PRO_PLUS,
        llm_credits_5h=constants.LLM_CREDITS_5H_PRO_PLUS,
        llm_credits_weekly=constants.LLM_CREDITS_WEEKLY_PRO_PLUS,
    ),
}
