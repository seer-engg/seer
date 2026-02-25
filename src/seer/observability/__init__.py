"""
Usage limits and enforcement system for Seer.

This module provides centralized configuration and tracking for subscription-based
usage limits across different tiers (Self-Hosted, Cloud Free, Cloud Pro/Pro+/Ultra).

Also provides Sentry error monitoring utilities for error capture and context enrichment.
"""
from seer.observability.exceptions import (
    CreditLimitExceeded,
    PollingIntervalTooFast,
    RunLimitExceeded,
    TrialExpiredError,
    UsageLimitError,
    WorkflowLimitExceeded,
)
from seer.observability.models import TierLimits
from seer.observability.service import (
    get_account_age_days,
    get_limits_for_tier,
    get_limits_for_user,
    get_subscription_for_user,
    is_trial_expired,
    resolve_user_tier,
)
from seer.observability.tracking import (
    get_5h_llm_credits_used,
    get_llm_usage_by_model,
    get_llm_usage_by_operation,
    get_llm_usage_by_workflow,
    get_llm_usage_daily_trend,
    get_llm_usage_records_paginated,
    get_monthly_llm_credits_detailed,
    get_monthly_llm_credits_used,
    get_monthly_run_count,
    get_weekly_llm_credits_used,
    get_workflow_count,
    reset_monthly_counters,
    track_llm_usage,
)
# Sentry error monitoring utilities
from seer.observability.sentry_client import (
    add_breadcrumb as sentry_add_breadcrumb,
    capture_exception as sentry_capture_exception,
    flush as sentry_flush,
    init_sentry,
    set_context as sentry_set_context,
    set_tag as sentry_set_tag,
    set_user_context as sentry_set_user_context,
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
    "get_workflow_count",
    "get_monthly_run_count",
    "track_llm_usage",
    "get_monthly_llm_credits_used",
    "get_5h_llm_credits_used",
    "get_weekly_llm_credits_used",
    "get_monthly_llm_credits_detailed",
    "reset_monthly_counters",
    # Analytics query functions
    "get_llm_usage_by_model",
    "get_llm_usage_by_operation",
    "get_llm_usage_daily_trend",
    "get_llm_usage_by_workflow",
    "get_llm_usage_records_paginated",
    # Exceptions
    "UsageLimitError",
    "WorkflowLimitExceeded",
    "RunLimitExceeded",
    "TrialExpiredError",
    "CreditLimitExceeded",
    "PollingIntervalTooFast",
    # Sentry error monitoring
    "init_sentry",
    "sentry_capture_exception",
    "sentry_set_user_context",
    "sentry_set_tag",
    "sentry_set_context",
    "sentry_flush",
    "sentry_add_breadcrumb",
]
