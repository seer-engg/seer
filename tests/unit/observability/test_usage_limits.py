# pylint: disable=import-outside-toplevel
# Reason: Test file with lazy imports
"""
Unit tests for usage limits and tracking system.

Tests cover:
- Tier limit configuration and retrieval
- Subscription tier resolution (mocked)

Note: Integration tests requiring database are in tests/integration/observability/
"""
import pytest

from seer.database.subscription_models import SubscriptionTier
from seer.observability import get_limits_for_tier
from seer.observability.models import TierLimits


pytestmark = pytest.mark.unit


def test_all_tiers_have_limits_defined():
    """Test that all subscription tiers have limits defined."""
    for tier in SubscriptionTier:
        limits = get_limits_for_tier(tier)
        assert limits is not None
        assert isinstance(limits, TierLimits)


def test_constants_match_tier_limits():
    """Test that tier limits use constants correctly."""
    from seer.observability.constants import tiered_usage_limits

    free = get_limits_for_tier(SubscriptionTier.FREE)
    assert free.workflows == tiered_usage_limits.WORKFLOWS_FREE
    assert free.runs_monthly == tiered_usage_limits.RUNS_MONTHLY_FREE
    assert free.account_day_limit == tiered_usage_limits.ACCOUNT_DAY_LIMIT_FREE
    assert free.poll_min_interval_seconds == tiered_usage_limits.POLL_MIN_INTERVAL_FREE
    assert free.llm_credits_monthly == tiered_usage_limits.LLM_CREDITS_FREE
