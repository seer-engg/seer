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


# ============================================================================
# Tier Limit Configuration Tests (No DB required)
# ============================================================================


def test_get_limits_for_free_tier():
    """Test that FREE tier limits are correctly configured."""
    limits = get_limits_for_tier(SubscriptionTier.FREE)

    assert isinstance(limits, TierLimits)
    assert limits.workflows == 3
    assert limits.runs_monthly == 100
    assert limits.account_day_limit == 14
    assert limits.poll_min_interval_seconds == 900  # 15 minutes
    assert limits.llm_credits_monthly == 5.00


def test_get_limits_for_pro_tier():
    """Test that PRO tier limits are correctly configured."""
    limits = get_limits_for_tier(SubscriptionTier.PRO)

    assert limits.workflows == -1  # Unlimited
    assert limits.runs_monthly == 1_000_000
    assert limits.account_day_limit == -1  # No time limit
    assert limits.poll_min_interval_seconds == 60  # 1 minute
    assert limits.llm_credits_monthly == 20.00


def test_get_limits_for_pro_plus_tier():
    """Test that PRO_PLUS tier limits are correctly configured."""
    limits = get_limits_for_tier(SubscriptionTier.PRO_PLUS)

    assert limits.workflows == -1
    assert limits.runs_monthly == 5_000_000
    assert limits.poll_min_interval_seconds == 30  # 30 seconds
    assert limits.llm_credits_monthly == 50.00


def test_get_limits_for_ultra_tier():
    """Test that ULTRA tier limits are correctly configured."""
    limits = get_limits_for_tier(SubscriptionTier.ULTRA)

    assert limits.workflows == -1
    assert limits.runs_monthly == 20_000_000
    assert limits.poll_min_interval_seconds == 10  # 10 seconds
    assert limits.llm_credits_monthly == 100.00


def test_self_hosted_limits():
    """Test that self-hosted limits are unlimited/BYOK."""
    limits = SELF_HOSTED_LIMITS

    assert limits.workflows == -1  # Unlimited
    assert limits.runs_monthly == -1  # Unlimited
    assert limits.account_day_limit == -1  # No time limit
    assert limits.poll_min_interval_seconds == 1  # 1 second minimum
    assert limits.llm_credits_monthly == -1  # BYOK


def test_tier_limit_properties():
    """Test TierLimits helper properties."""
    free_limits = get_limits_for_tier(SubscriptionTier.FREE)
    pro_limits = get_limits_for_tier(SubscriptionTier.PRO)
    self_hosted = SELF_HOSTED_LIMITS

    # FREE tier
    assert not free_limits.has_unlimited_workflows
    assert not free_limits.has_unlimited_runs
    assert not free_limits.has_unlimited_chat
    assert not free_limits.is_chat_disabled
    assert free_limits.has_time_limit

    # PRO tier
    assert pro_limits.has_unlimited_workflows
    assert not pro_limits.has_unlimited_runs
    assert pro_limits.has_unlimited_chat
    assert not pro_limits.is_chat_disabled
    assert not pro_limits.has_time_limit

    # Self-hosted
    assert self_hosted.has_unlimited_workflows
    assert self_hosted.has_unlimited_runs
    assert self_hosted.is_chat_disabled
    assert self_hosted.has_unlimited_credits


def test_all_tiers_have_limits_defined():
    """Test that all subscription tiers have limits defined."""
    for tier in SubscriptionTier:
        limits = get_limits_for_tier(tier)
        assert limits is not None
        assert isinstance(limits, TierLimits)


def test_tier_limits_are_logical():
    """Test that tier limits follow logical progression (higher tiers have more/equal limits)."""
    free = get_limits_for_tier(SubscriptionTier.FREE)
    pro = get_limits_for_tier(SubscriptionTier.PRO)
    pro_plus = get_limits_for_tier(SubscriptionTier.PRO_PLUS)
    ultra = get_limits_for_tier(SubscriptionTier.ULTRA)

    # Workflows: FREE is limited, rest are unlimited
    assert free.workflows > 0
    assert pro.workflows == -1
    assert pro_plus.workflows == -1
    assert ultra.workflows == -1

    # Runs: should increase or be unlimited
    assert free.runs_monthly < pro.runs_monthly
    assert pro.runs_monthly < pro_plus.runs_monthly
    assert pro_plus.runs_monthly < ultra.runs_monthly

    # Poll intervals: should decrease (more frequent polling)
    assert free.poll_min_interval_seconds > pro.poll_min_interval_seconds
    assert pro.poll_min_interval_seconds > pro_plus.poll_min_interval_seconds
    assert pro_plus.poll_min_interval_seconds > ultra.poll_min_interval_seconds

    # LLM credits: should increase
    assert free.llm_credits_monthly < pro.llm_credits_monthly
    assert pro.llm_credits_monthly < pro_plus.llm_credits_monthly
    assert pro_plus.llm_credits_monthly < ultra.llm_credits_monthly


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
async def test_subscription_resolver_no_subscription():
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_usage_tracking_workflow_count():
    """
    Integration test: Test workflow counter operations.

    Requires database connection.
    """
    from datetime import datetime, timezone
    from seer.database.models import User
    from seer.observability import (
        get_workflow_count,
    )

    user = await User.create(
        user_id="test_user_workflow",
        email="workflow@example.com",
        created_at=datetime.now(timezone.utc),
    )

    try:
        # Test initial count
        count = await get_workflow_count(user)
        assert count == 0

        # Test increment

        # Verify count increased
        count = await get_workflow_count(user)
        assert count == 1
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
