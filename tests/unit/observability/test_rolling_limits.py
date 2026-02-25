# pylint: disable=import-outside-toplevel
# Reason: Test file with lazy imports
"""
Unit tests for rolling window LLM credit limits.

Tests cover:
- 5-hour and weekly constants exist for all tiers
- TierLimits model includes new fields and properties
- Rolling window query functions
- Multi-period credit limit checking
- CreditLimitExceeded exception with period field
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database.subscription_models import SubscriptionTier
from seer.observability import get_limits_for_tier
from seer.observability.constants import tiered_usage_limits
from seer.observability.exceptions import CreditLimitExceeded, LimitPeriod
from seer.observability.models import SELF_HOSTED_LIMITS, TIER_LIMITS_REGISTRY, TierLimits


pytestmark = pytest.mark.unit


# =============================================================================
# Constants Tests
# =============================================================================


class TestShortTermLimitConstants:
    """Tests for 5-hour and weekly limit constants."""

    def test_5h_constants_exist_for_all_tiers(self):
        """Test that 5-hour constants exist for all tiers."""
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_SELF_HOSTED")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_FREE")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_PRO")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_PRO_PLUS")

    def test_weekly_constants_exist_for_all_tiers(self):
        """Test that weekly constants exist for all tiers."""
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_SELF_HOSTED")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_FREE")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_PRO")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_PRO_PLUS")

    def test_self_hosted_limits_are_unlimited(self):
        """Test that self-hosted 5h and weekly limits are unlimited (-1)."""
        assert tiered_usage_limits.LLM_CREDITS_5H_SELF_HOSTED == -1
        assert tiered_usage_limits.LLM_CREDITS_WEEKLY_SELF_HOSTED == -1

    def test_5h_limits_are_reasonable_fraction_of_monthly(self):
        """Test that 5h limits are ~25% of monthly (burst protection)."""
        # Free tier: 5h=$1, monthly=$5 -> 20%
        ratio_free = tiered_usage_limits.LLM_CREDITS_5H_FREE / tiered_usage_limits.LLM_CREDITS_FREE
        assert 0.15 <= ratio_free <= 0.35

        # Pro tier: 5h=$5, monthly=$20 -> 25%
        ratio_pro = tiered_usage_limits.LLM_CREDITS_5H_PRO / tiered_usage_limits.LLM_CREDITS_PRO
        assert 0.15 <= ratio_pro <= 0.35

    def test_weekly_limits_are_reasonable_fraction_of_monthly(self):
        """Test that weekly limits are ~60-70% of monthly."""
        # Free tier: weekly=$3, monthly=$5 -> 60%
        ratio_free = tiered_usage_limits.LLM_CREDITS_WEEKLY_FREE / tiered_usage_limits.LLM_CREDITS_FREE
        assert 0.50 <= ratio_free <= 0.80

        # Pro tier: weekly=$12, monthly=$20 -> 60%
        ratio_pro = tiered_usage_limits.LLM_CREDITS_WEEKLY_PRO / tiered_usage_limits.LLM_CREDITS_PRO
        assert 0.50 <= ratio_pro <= 0.80


# =============================================================================
# TierLimits Model Tests
# =============================================================================


class TestTierLimitsModel:
    """Tests for TierLimits model with new fields."""

    def test_tier_limits_has_5h_field(self):
        """Test that TierLimits has llm_credits_5h field."""
        limits = TierLimits(
            workflows=10,
            runs_monthly=1000,
                        account_day_limit=30,
            poll_min_interval_seconds=60,
            llm_credits_monthly=10.0,
            llm_credits_5h=2.0,
            llm_credits_weekly=5.0,
        )
        assert limits.llm_credits_5h == 2.0

    def test_tier_limits_has_weekly_field(self):
        """Test that TierLimits has llm_credits_weekly field."""
        limits = TierLimits(
            workflows=10,
            runs_monthly=1000,
                        account_day_limit=30,
            poll_min_interval_seconds=60,
            llm_credits_monthly=10.0,
            llm_credits_5h=2.0,
            llm_credits_weekly=5.0,
        )
        assert limits.llm_credits_weekly == 5.0

    def test_has_unlimited_5h_credits_property(self):
        """Test has_unlimited_5h_credits property."""
        limited = TierLimits(
            workflows=10,
            runs_monthly=1000,
                        account_day_limit=30,
            poll_min_interval_seconds=60,
            llm_credits_monthly=10.0,
            llm_credits_5h=2.0,
            llm_credits_weekly=5.0,
        )
        assert limited.has_unlimited_5h_credits is False

        unlimited = TierLimits(
            workflows=-1,
            runs_monthly=-1,
                        account_day_limit=-1,
            poll_min_interval_seconds=1,
            llm_credits_monthly=-1,
            llm_credits_5h=-1,
            llm_credits_weekly=-1,
        )
        assert unlimited.has_unlimited_5h_credits is True

    def test_has_unlimited_weekly_credits_property(self):
        """Test has_unlimited_weekly_credits property."""
        limited = TierLimits(
            workflows=10,
            runs_monthly=1000,
                        account_day_limit=30,
            poll_min_interval_seconds=60,
            llm_credits_monthly=10.0,
            llm_credits_5h=2.0,
            llm_credits_weekly=5.0,
        )
        assert limited.has_unlimited_weekly_credits is False

        unlimited = TierLimits(
            workflows=-1,
            runs_monthly=-1,
                        account_day_limit=-1,
            poll_min_interval_seconds=1,
            llm_credits_monthly=-1,
            llm_credits_5h=-1,
            llm_credits_weekly=-1,
        )
        assert unlimited.has_unlimited_weekly_credits is True


class TestTierRegistry:
    """Tests for tier registry with new fields."""

    def test_all_tiers_have_short_term_limits(self):
        """Test that all subscription tiers define 5h and weekly limits."""
        for tier in SubscriptionTier:
            limits = get_limits_for_tier(tier)
            assert limits is not None
            assert hasattr(limits, "llm_credits_5h")
            assert hasattr(limits, "llm_credits_weekly")

    def test_self_hosted_limits_include_short_term(self):
        """Test that SELF_HOSTED_LIMITS includes 5h and weekly limits."""
        assert hasattr(SELF_HOSTED_LIMITS, "llm_credits_5h")
        assert hasattr(SELF_HOSTED_LIMITS, "llm_credits_weekly")
        assert SELF_HOSTED_LIMITS.llm_credits_5h == -1
        assert SELF_HOSTED_LIMITS.llm_credits_weekly == -1

    def test_tier_registry_uses_correct_constants(self):
        """Test that tier registry uses the correct constant values."""
        free_limits = TIER_LIMITS_REGISTRY[SubscriptionTier.FREE]
        assert free_limits.llm_credits_5h == tiered_usage_limits.LLM_CREDITS_5H_FREE
        assert free_limits.llm_credits_weekly == tiered_usage_limits.LLM_CREDITS_WEEKLY_FREE

        pro_limits = TIER_LIMITS_REGISTRY[SubscriptionTier.PRO]
        assert pro_limits.llm_credits_5h == tiered_usage_limits.LLM_CREDITS_5H_PRO
        assert pro_limits.llm_credits_weekly == tiered_usage_limits.LLM_CREDITS_WEEKLY_PRO


# =============================================================================
# LimitPeriod Enum Tests
# =============================================================================


class TestLimitPeriodEnum:
    """Tests for LimitPeriod enum."""

    def test_limit_period_values(self):
        """Test LimitPeriod enum values."""
        assert LimitPeriod.FIVE_HOUR.value == "5_hour"
        assert LimitPeriod.WEEKLY.value == "weekly"
        assert LimitPeriod.MONTHLY.value == "monthly"

    def test_limit_period_is_string_enum(self):
        """Test that LimitPeriod is a string enum."""
        assert isinstance(LimitPeriod.FIVE_HOUR, str)
        assert LimitPeriod.FIVE_HOUR == "5_hour"


# =============================================================================
# CreditLimitExceeded Exception Tests
# =============================================================================


class TestCreditLimitExceededException:
    """Tests for CreditLimitExceeded with period support."""

    def test_default_period_is_monthly(self):
        """Test that default period is MONTHLY for backwards compatibility."""
        exc = CreditLimitExceeded(
            limit=10.0,
            current=12.0,
            tier=SubscriptionTier.FREE,
        )
        assert exc.period == LimitPeriod.MONTHLY

    def test_5h_period_in_message(self):
        """Test that 5-hour period appears in error message."""
        exc = CreditLimitExceeded(
            limit=1.0,
            current=1.2,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.FIVE_HOUR,
        )
        assert "5-hour" in exc.message

    def test_weekly_period_in_message(self):
        """Test that weekly period appears in error message."""
        exc = CreditLimitExceeded(
            limit=3.0,
            current=3.6,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.WEEKLY,
        )
        assert "weekly" in exc.message

    def test_monthly_period_in_message(self):
        """Test that monthly period appears in error message."""
        exc = CreditLimitExceeded(
            limit=5.0,
            current=6.0,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.MONTHLY,
        )
        assert "monthly" in exc.message

    def test_to_dict_includes_period(self):
        """Test that to_dict includes period field."""
        exc = CreditLimitExceeded(
            limit=1.0,
            current=1.2,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.FIVE_HOUR,
        )
        data = exc.to_dict()
        assert "period" in data
        assert data["period"] == "5_hour"

    def test_soft_limit_message_includes_period(self):
        """Test that soft limit message includes period."""
        exc = CreditLimitExceeded(
            limit=1.0,
            current=0.9,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.FIVE_HOUR,
            is_soft_limit=True,
        )
        assert "5-hour" in exc.message
        assert "Warning" in exc.message


# =============================================================================
# Credit Gate Tests
# =============================================================================


@pytest.mark.asyncio
class TestCreditGate:
    """Tests for check_credit_limit with multi-period checking."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        from seer.database import User

        user = MagicMock(spec=User)
        user.id = 1
        user.user_id = "user_123"
        return user

    async def test_unlimited_credits_skips_all_checks(self, mock_user):
        """Test that unlimited credits (self-hosted) skips all limit checks."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            # Simulate self-hosted with unlimited credits
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = True
            mock_get_limits.return_value = mock_limits

            await check_credit_limit(mock_user)

            # None of the usage functions should be called
            mock_5h.assert_not_called()
            mock_weekly.assert_not_called()
            mock_monthly.assert_not_called()

    async def test_5h_limit_checked_before_weekly(self, mock_user):
        """Test that 5-hour limit is checked before weekly."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            # Setup limits (not unlimited)
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # 5h limit exceeded (at 120%)
            mock_5h.return_value = Decimal("1.20")
            mock_weekly.return_value = Decimal("0.50")  # Under limit
            mock_monthly.return_value = Decimal("0.50")  # Under limit

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            # Should fail on 5-hour limit
            assert exc_info.value.period == LimitPeriod.FIVE_HOUR

            # Weekly and monthly should not have been checked
            mock_weekly.assert_not_called()
            mock_monthly.assert_not_called()

    async def test_weekly_limit_checked_when_5h_passes(self, mock_user):
        """Test that weekly limit is checked when 5-hour passes."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # 5h under limit, weekly exceeded
            mock_5h.return_value = Decimal("0.50")  # Under limit
            mock_weekly.return_value = Decimal("3.60")  # At 120%
            mock_monthly.return_value = Decimal("3.60")  # Under limit

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.period == LimitPeriod.WEEKLY

    async def test_monthly_limit_checked_when_short_term_pass(self, mock_user):
        """Test that monthly limit is checked when 5h and weekly pass."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # 5h and weekly under limit, monthly exceeded
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("2.00")
            mock_monthly.return_value = Decimal("6.00")  # At 120%

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.period == LimitPeriod.MONTHLY

    async def test_all_limits_pass(self, mock_user):
        """Test that no exception is raised when all limits pass."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # All under limits
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("1.50")
            mock_monthly.return_value = Decimal("2.50")

            # Should not raise
            await check_credit_limit(mock_user)

    async def test_skips_5h_check_when_unlimited(self, mock_user):
        """Test that 5h check is skipped when has_unlimited_5h_credits is True."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = True  # Unlimited 5h
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = -1
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.PRO

            mock_weekly.return_value = Decimal("1.00")
            mock_monthly.return_value = Decimal("2.00")

            await check_credit_limit(mock_user)

            # 5h should not be called
            mock_5h.assert_not_called()
            # Weekly and monthly should be called
            mock_weekly.assert_called_once()
            mock_monthly.assert_called_once()

    async def test_warning_logged_at_80_percent(self, mock_user, caplog):
        """Test that warning is logged when usage reaches 80%."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_limits_for_user") as mock_get_limits,
            patch("seer.observability.credit_gate.resolve_user_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # At 80% of 5h limit (should warn but not block)
            mock_5h.return_value = Decimal("0.80")
            mock_weekly.return_value = Decimal("1.00")
            mock_monthly.return_value = Decimal("2.00")

            import logging
            with caplog.at_level(logging.WARNING):
                await check_credit_limit(mock_user)

            # Should have logged a warning
            assert "approaching" in caplog.text.lower() or "5_hour" in caplog.text
