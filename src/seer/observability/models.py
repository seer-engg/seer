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
        chat_messages_total: Total chat messages across all workflows (-1 = unlimited, 0 = disabled)
        account_day_limit: Maximum days from signup (-1 = unlimited)
        poll_min_interval_seconds: Minimum polling interval in seconds
        llm_credits_monthly: Monthly LLM credit allowance in USD (-1 = unlimited/BYOK)
    """

    workflows: int = Field(description="Maximum workflows (-1 = unlimited)")
    runs_monthly: int = Field(description="Maximum runs per month (-1 = unlimited)")
    chat_messages_total: int = Field(
        description="Total chat messages across all workflows (-1 = unlimited, 0 = disabled)"
    )
    account_day_limit: int = Field(
        description="Maximum days from signup (-1 = unlimited)"
    )
    poll_min_interval_seconds: int = Field(
        description="Minimum polling interval in seconds"
    )
    llm_credits_monthly: float = Field(
        description="Monthly LLM credits in USD (-1 = unlimited/BYOK)"
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
    def has_unlimited_chat(self) -> bool:
        """Check if chat messages are unlimited."""
        return self.chat_messages_total == -1

    @property
    def is_chat_disabled(self) -> bool:
        """Check if chat is disabled."""
        return self.chat_messages_total == 0

    @property
    def has_unlimited_credits(self) -> bool:
        """Check if LLM credits are unlimited (BYOK mode)."""
        return self.llm_credits_monthly == -1

    @property
    def has_time_limit(self) -> bool:
        """Check if account has a time limit (trial period)."""
        return self.account_day_limit > 0


# ============================================================================
# Tier Limits Registry
# ============================================================================

# Special tier for self-hosted deployments
SELF_HOSTED_LIMITS = TierLimits(
    workflows=constants.WORKFLOWS_SELF_HOSTED,
    runs_monthly=constants.RUNS_MONTHLY_SELF_HOSTED,
    chat_messages_total=constants.CHAT_MESSAGES_TOTAL_SELF_HOSTED,
    account_day_limit=constants.ACCOUNT_DAY_LIMIT_SELF_HOSTED,
    poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_SELF_HOSTED,
    llm_credits_monthly=constants.LLM_CREDITS_SELF_HOSTED,
)

# Cloud tier limits mapped to SubscriptionTier enum
TIER_LIMITS_REGISTRY: dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        workflows=constants.WORKFLOWS_FREE,
        runs_monthly=constants.RUNS_MONTHLY_FREE,
        chat_messages_total=constants.CHAT_MESSAGES_TOTAL_FREE,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_FREE,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_FREE,
        llm_credits_monthly=constants.LLM_CREDITS_FREE,
    ),
    SubscriptionTier.PRO: TierLimits(
        workflows=constants.WORKFLOWS_PRO,
        runs_monthly=constants.RUNS_MONTHLY_PRO,
        chat_messages_total=constants.CHAT_MESSAGES_TOTAL_PRO,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_PRO,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_PRO,
        llm_credits_monthly=constants.LLM_CREDITS_PRO,
    ),
    SubscriptionTier.PRO_PLUS: TierLimits(
        workflows=constants.WORKFLOWS_PRO_PLUS,
        runs_monthly=constants.RUNS_MONTHLY_PRO_PLUS,
        chat_messages_total=constants.CHAT_MESSAGES_TOTAL_PRO_PLUS,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_PRO_PLUS,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_PRO_PLUS,
        llm_credits_monthly=constants.LLM_CREDITS_PRO_PLUS,
    ),
    SubscriptionTier.ULTRA: TierLimits(
        workflows=constants.WORKFLOWS_ULTRA,
        runs_monthly=constants.RUNS_MONTHLY_ULTRA,
        chat_messages_total=constants.CHAT_MESSAGES_TOTAL_ULTRA,
        account_day_limit=constants.ACCOUNT_DAY_LIMIT_ULTRA,
        poll_min_interval_seconds=constants.POLL_MIN_INTERVAL_ULTRA,
        llm_credits_monthly=constants.LLM_CREDITS_ULTRA,
    ),
}
