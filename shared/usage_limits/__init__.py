"""
Usage limits and enforcement system for Seer.

This module provides centralized configuration and tracking for subscription-based
usage limits across different tiers (Self-Hosted, Cloud Free, Cloud Pro/Pro+/Ultra).
"""
from shared.usage_limits.exceptions import (
    ChatDisabledError,
    CreditLimitExceeded,
    MessageLimitExceeded,
    PollingIntervalTooFast,
    RunLimitExceeded,
    TrialExpiredError,
    UsageLimitError,
    WorkflowLimitExceeded,
)
from shared.usage_limits.models import TierLimits
from shared.usage_limits.service import (
    get_account_age_days,
    get_limits_for_tier,
    get_limits_for_user,
    get_subscription_for_user,
    is_trial_expired,
    resolve_user_tier,
)
from shared.usage_limits.tracking import (
    decrement_workflow_count,
    get_chat_message_count,  # Deprecated - use get_total_chat_message_count
    get_monthly_llm_credits_detailed,
    get_monthly_llm_credits_used,
    get_monthly_run_count,
    get_total_chat_message_count,
    get_workflow_count,
    increment_chat_message_count,
    increment_monthly_run_count,
    increment_workflow_count,
    reset_monthly_counters,
    track_llm_usage,
)

__all__ = [
    # Models
    "TierLimits",
    # Service functions
    "get_limits_for_tier",
    "get_limits_for_user",
    "resolve_user_tier",
    "get_account_age_days",
    "is_trial_expired",
    "get_subscription_for_user",
    # Tracking functions
    "increment_workflow_count",
    "decrement_workflow_count",
    "increment_monthly_run_count",
    "increment_chat_message_count",
    "get_workflow_count",
    "get_monthly_run_count",
    "get_chat_message_count",  # Deprecated
    "get_total_chat_message_count",
    "track_llm_usage",
    "get_monthly_llm_credits_used",
    "get_monthly_llm_credits_detailed",
    "reset_monthly_counters",
    # Exceptions
    "UsageLimitError",
    "WorkflowLimitExceeded",
    "RunLimitExceeded",
    "MessageLimitExceeded",
    "TrialExpiredError",
    "CreditLimitExceeded",
    "PollingIntervalTooFast",
    "ChatDisabledError",
]
