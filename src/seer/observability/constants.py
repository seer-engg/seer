"""
Numerical constants for usage limits across subscription tiers.

These values define the hard limits for each feature dimension across
Self-Hosted, Cloud Free, Cloud Pro, Cloud Pro+, and Cloud Ultra tiers.

Convention:
  - -1 means unlimited
  - 0 means disabled/not allowed
"""
from typing import Optional

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

    WORKFLOWS_SELF_HOSTED: int = Field(default=-1)
    WORKFLOWS_FREE: int = Field(default=3)
    WORKFLOWS_PRO: int = Field(default=-1)  # Unlimited
    WORKFLOWS_PRO_PLUS: int = Field(default=-1)  # Unlimited
    WORKFLOWS_ULTRA: int = Field(default=-1)  # Unlimited
    # ============================================================================
    # Workflow Run Limits (Monthly)
    # ============================================================================

    RUNS_MONTHLY_SELF_HOSTED: int = Field(default=-1)  # Unlimited
    RUNS_MONTHLY_FREE: int = Field(default=100)
    RUNS_MONTHLY_PRO: int = Field(default=1_000_000)
    RUNS_MONTHLY_PRO_PLUS: int = Field(default=5_000_000)
    RUNS_MONTHLY_ULTRA: int = Field(default=20_000_000)
    # ============================================================================
    # Chat AI Message Limits (Total per User, across all workflows)
    # ============================================================================

    CHAT_MESSAGES_TOTAL_SELF_HOSTED: int = Field(default=0)  # Disabled
    CHAT_MESSAGES_TOTAL_FREE: int = Field(default=5)  # 50 total messages across all workflows
    CHAT_MESSAGES_TOTAL_PRO: int = Field(default=100)  # Unlimited
    CHAT_MESSAGES_TOTAL_PRO_PLUS: int = Field(default=-1)  # Unlimited
    CHAT_MESSAGES_TOTAL_ULTRA: int = Field(default=-1)  # Unlimited

    # ============================================================================
    # Account Day Limits
    # ============================================================================

    ACCOUNT_DAY_LIMIT_SELF_HOSTED: int = Field(default=-1)  # Unlimited
    ACCOUNT_DAY_LIMIT_FREE: int = Field(default=14)  # 14-day trial
    ACCOUNT_DAY_LIMIT_PRO: int = Field(default=-1)  # No limit
    ACCOUNT_DAY_LIMIT_PRO_PLUS: int = Field(default=-1)  # No limit
    ACCOUNT_DAY_LIMIT_ULTRA: int = Field(default=-1)  # No limit

    # ============================================================================
    # Polling Frequency Limits (Minimum Interval in Seconds)
    # ============================================================================

    POLL_MIN_INTERVAL_SELF_HOSTED: int = Field(default=1)  # 1 second minimum
    POLL_MIN_INTERVAL_FREE: int = Field(default=900)  # 15 minutes
    POLL_MIN_INTERVAL_PRO: int = Field(default=60)  # 1 minute
    POLL_MIN_INTERVAL_PRO_PLUS: int = Field(default=30)  # 30 seconds
    POLL_MIN_INTERVAL_ULTRA: int = Field(default=10)  # 10 seconds

    # ============================================================================
    # LLM Credit Limits (Monthly, in USD)
    # ============================================================================

    LLM_CREDITS_SELF_HOSTED: int = Field(default=-1)  # BYOK (Bring Your Own Key), unlimited
    LLM_CREDITS_FREE: float = Field(default=5.00)
    LLM_CREDITS_PRO: float = Field(default=20.00)
    LLM_CREDITS_PRO_PLUS: float = Field(default=50.00)
    LLM_CREDITS_ULTRA: float = Field(default=100.00)

    # ============================================================================
    # Credit Thresholds
    # ============================================================================

    # Soft warning threshold (percentage of monthly credits)
    CREDIT_WARNING_THRESHOLD: float = Field(default=0.80)  # Warn at 80% usage

    # Hard block threshold (percentage of monthly credits)
    CREDIT_BLOCK_THRESHOLD: float = Field(default=1.20)  # Block at 120% usage (allow 20% overage)


tiered_usage_limits = TieredUsageLimits()
