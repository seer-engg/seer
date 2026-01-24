# pylint: disable=unused-import,import-outside-toplevel,unused-argument
# Reason: Test file with fixtures and lazy imports
"""
Unit tests for usage limits and tracking system.

Tests cover:
- Tier limit configuration and retrieval
- Subscription tier resolution
- Usage counter operations (tests require database)
"""
from unittest.mock import patch

import pytest

from seer.database.subscription_models import SubscriptionTier
from seer.observability import get_limits_for_tier
from seer.observability.models import SELF_HOSTED_LIMITS, TierLimits



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


# ============================================================================
# Integration Tests (Require database - marked for integration test suite)
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscription_resolver_no_subscription(db_engine):
    """
    Integration test: Test resolving tier for user without subscription.

    This test requires a database connection and is marked as integration test.
    Run with: pytest -m integration
    """
    from datetime import datetime, timezone
    from seer.database.models import User
    from seer.observability import resolve_user_tier

    # Create test user
    user = await User.create(
        user_id="test_user_123",
        email="test@example.com",
        created_at=datetime.now(timezone.utc),
    )

    try:
        tier = await resolve_user_tier(user)
        assert tier == SubscriptionTier.FREE
    finally:
        await user.delete()



# NOTE: Additional integration tests for subscription resolver and usage tracking
# should be added here. They require:
# 1. Database setup and teardown
# 2. Test isolation between tests
# 3. Proper async test handling
#
# These tests would cover:
# - resolve_user_tier with active/canceled/trialing subscriptions
# - get_limits_for_user in cloud vs self-hosted mode
# - increment_monthly_run_count
# - increment_chat_message_count
# - track_llm_usage
# - is_trial_expired
# - get_subscription_for_user
#
# Run integration tests with: pytest -m integration shared/tests/
