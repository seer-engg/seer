# pylint: disable=duplicate-code  # Reason: Shares SettingsConfigDict configuration with seer.config to keep env parsing consistent
"""
Numerical constants for usage limits across subscription tiers.

These values define the hard limits for each feature dimension across
Free, Pro, and Pro+ tiers.

Convention:
  - -1 means unlimited
  - 0 means disabled/not allowed
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TieredUsageLimits(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    # ============================================================================
    # Workflow Limits
    # ============================================================================

    WORKFLOWS_FREE: int = Field(default=3)
    WORKFLOWS_PRO: int = Field(default=-1)  # Unlimited
    WORKFLOWS_PRO_PLUS: int = Field(default=-1)  # Unlimited
    # ============================================================================
    # Workflow Run Limits (Monthly)
    # ============================================================================

    RUNS_MONTHLY_FREE: int = Field(default=100)
    RUNS_MONTHLY_PRO: int = Field(default=1_000_000)
    RUNS_MONTHLY_PRO_PLUS: int = Field(default=5_000_000)

    # ============================================================================
    # Account Day Limits
    # ============================================================================

    ACCOUNT_DAY_LIMIT_FREE: int = Field(default=-1)  # No signup-age limit
    ACCOUNT_DAY_LIMIT_PRO: int = Field(default=-1)  # No limit
    ACCOUNT_DAY_LIMIT_PRO_PLUS: int = Field(default=-1)  # No limit

    # ============================================================================
    # Polling Frequency Limits (Minimum Interval in Seconds)
    # ============================================================================

    POLL_MIN_INTERVAL_FREE: int = Field(default=900)  # 15 minutes
    POLL_MIN_INTERVAL_PRO: int = Field(default=60)  # 1 minute
    POLL_MIN_INTERVAL_PRO_PLUS: int = Field(default=30)  # 30 seconds

    # ============================================================================
    # LLM Credit Limits (Monthly, in USD)
    # ============================================================================

    LLM_CREDITS_FREE: float = Field(default=1.00)
    LLM_CREDITS_PRO: float = Field(default=20.00)
    LLM_CREDITS_PRO_PLUS: float = Field(default=50.00)

    # ============================================================================
    # LLM Credit Limits (5-Hour Rolling Window, in USD)
    # ============================================================================

    LLM_CREDITS_5H_FREE: float = Field(default=1.00)
    LLM_CREDITS_5H_PRO: float = Field(default=5.00)
    LLM_CREDITS_5H_PRO_PLUS: float = Field(default=15.00)

    # ============================================================================
    # LLM Credit Limits (Weekly Rolling Window, in USD)
    # ============================================================================

    LLM_CREDITS_WEEKLY_FREE: float = Field(default=1.00)
    LLM_CREDITS_WEEKLY_PRO: float = Field(default=12.00)
    LLM_CREDITS_WEEKLY_PRO_PLUS: float = Field(default=35.00)

    # ============================================================================
    # Credit Thresholds
    # ============================================================================

    # Soft warning threshold (percentage of monthly credits)
    CREDIT_WARNING_THRESHOLD: float = Field(default=0.80)  # Warn at 80% usage

    # Hard block threshold (percentage of monthly credits)
    CREDIT_BLOCK_THRESHOLD: float = Field(default=1.20)  # Block at 120% usage (allow 20% overage)

    # ============================================================================
    # Overage Settings
    # ============================================================================

    # Default margin multiplier for overage pricing (1.30 = 30% margin on LLM cost)
    OVERAGE_DEFAULT_MARGIN_MULTIPLIER: float = Field(default=1.30)

    # Minimum spending cap in cents ($5)
    OVERAGE_MIN_CAP_CENTS: int = Field(default=500)

    # Maximum spending cap in cents ($1000)
    OVERAGE_MAX_CAP_CENTS: int = Field(default=100000)

    # Default spending cap in cents ($50)
    OVERAGE_DEFAULT_CAP_CENTS: int = Field(default=5000)

    # Warning threshold for overage cap (80%)
    OVERAGE_WARNING_THRESHOLD: float = Field(default=0.80)


tiered_usage_limits = TieredUsageLimits()
