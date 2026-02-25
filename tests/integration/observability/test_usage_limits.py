# pylint: disable=unused-import,import-outside-toplevel,unused-argument
# Reason: Test file with fixtures and lazy imports
"""
Integration tests for usage limits and tracking system.

These tests require a database connection and cover:
- Subscription tier resolution with real database
- Usage counter operations
"""
from datetime import datetime, timezone

import pytest

from seer.database.subscription_models import SubscriptionTier


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_subscription_resolver_no_subscription(db_engine):
    """
    Integration test: Test resolving tier for user without subscription.

    This test requires a database connection and is marked as integration test.
    Run with: pytest -m integration
    """
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
# - track_llm_usage
# - is_trial_expired
# - get_subscription_for_user
#
# Run integration tests with: pytest -m integration tests/integration/
