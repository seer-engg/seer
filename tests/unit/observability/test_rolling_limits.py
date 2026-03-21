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
from seer.observability.models import TIER_LIMITS_REGISTRY, TierLimits


pytestmark = pytest.mark.unit


# =============================================================================
# Constants Tests
# =============================================================================


class TestShortTermLimitConstants:
    """Tests for 5-hour and weekly limit constants."""

    def test_5h_constants_exist_for_all_tiers(self):
        """Test that 5-hour constants exist for all tiers."""
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_FREE")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_PRO")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_5H_PRO_PLUS")

    def test_weekly_constants_exist_for_all_tiers(self):
        """Test that weekly constants exist for all tiers."""
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_FREE")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_PRO")
        assert hasattr(tiered_usage_limits, "LLM_CREDITS_WEEKLY_PRO_PLUS")

    def test_5h_limits_match_monthly_for_free_tier(self):
        """Test that free-tier 5h credits match the monthly allowance."""
        assert tiered_usage_limits.LLM_CREDITS_5H_FREE == tiered_usage_limits.LLM_CREDITS_FREE

    def test_weekly_limits_match_monthly_for_free_tier(self):
        """Test that free-tier weekly credits match the monthly allowance."""
        assert tiered_usage_limits.LLM_CREDITS_WEEKLY_FREE == tiered_usage_limits.LLM_CREDITS_FREE


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
            limit=1.0,
            current=1.2,
            tier=SubscriptionTier.FREE,
            period=LimitPeriod.WEEKLY,
        )
        assert "weekly" in exc.message

    def test_monthly_period_in_message(self):
        """Test that monthly period appears in error message."""
        exc = CreditLimitExceeded(
            limit=1.0,
            current=1.2,
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
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
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
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
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
            mock_limits.llm_credits_weekly = 1.0
            mock_limits.llm_credits_monthly = 1.0
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
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 1.0
            mock_limits.llm_credits_monthly = 1.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # 5h under limit, weekly exceeded
            mock_5h.return_value = Decimal("0.50")  # Under limit
            mock_weekly.return_value = Decimal("1.21")  # Above 120%
            mock_monthly.return_value = Decimal("0.90")  # Under limit

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.period == LimitPeriod.WEEKLY

    async def test_monthly_limit_checked_when_short_term_pass(self, mock_user):
        """Test that monthly limit is checked when 5h and weekly pass."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 1.0
            mock_limits.llm_credits_monthly = 1.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # 5h and weekly under limit, monthly exceeded
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("0.90")
            mock_monthly.return_value = Decimal("1.21")  # Above 120%

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.period == LimitPeriod.MONTHLY

    async def test_all_limits_pass(self, mock_user):
        """Test that no exception is raised when all limits pass."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
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
            mock_weekly.return_value = Decimal("0.70")
            mock_monthly.return_value = Decimal("0.70")

            # Should not raise
            await check_credit_limit(mock_user)

    async def test_skips_5h_check_when_unlimited(self, mock_user):
        """Test that 5h check is skipped when has_unlimited_5h_credits is True."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
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
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
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
            mock_weekly.return_value = Decimal("0.70")
            mock_monthly.return_value = Decimal("0.70")

            import logging
            with caplog.at_level(logging.WARNING):
                await check_credit_limit(mock_user)

            # Should have logged a warning
            assert "approaching" in caplog.text.lower() or "5_hour" in caplog.text


# =============================================================================
# Credit Gate Overage Allowance Tests
# =============================================================================


@pytest.mark.asyncio
class TestCreditGateOverageAllowance:
    """Tests for credit gate overage allowance flow."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        from seer.database import User

        user = MagicMock(spec=User)
        user.id = 1
        user.user_id = "user_123"
        return user

    async def test_monthly_limit_allows_overage_when_enabled(self, mock_user):
        """Test that monthly limit allows usage when overage is enabled with remaining cap."""
        from seer.observability.credit_gate import check_credit_limit

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
            patch("seer.observability.credit_gate._check_overage_allowance") as mock_overage,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.PRO

            # Under short-term limits but over monthly (at 120%)
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("2.00")
            mock_monthly.return_value = Decimal("6.00")  # At 120% of monthly

            # Overage is allowed
            from seer.observability.credit_gate import OverageCheckResult
            mock_overage.return_value = OverageCheckResult(
                allowed=True,
                overage_enabled=True,
                overage_cap_reached=False,
                remaining_cap_cents=5000,
            )

            # Should NOT raise because overage is allowed
            await check_credit_limit(mock_user)

    async def test_monthly_limit_blocks_when_overage_cap_reached(self, mock_user):
        """Test that monthly limit blocks when overage cap is reached."""
        from seer.observability.credit_gate import check_credit_limit
        from seer.observability.exceptions import CreditLimitExceeded

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
            patch("seer.observability.credit_gate._check_overage_allowance") as mock_overage,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 3.0
            mock_limits.llm_credits_monthly = 5.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.PRO

            # Over monthly limit
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("2.00")
            mock_monthly.return_value = Decimal("6.50")

            # Overage cap reached
            from seer.observability.credit_gate import OverageCheckResult
            mock_overage.return_value = OverageCheckResult(
                allowed=False,
                overage_enabled=True,
                overage_cap_reached=True,
                remaining_cap_cents=0,
            )

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.overage_cap_reached is True

    async def test_monthly_limit_blocks_when_overage_not_enabled(self, mock_user):
        """Test that monthly limit blocks when overage is not enabled."""
        from seer.observability.credit_gate import check_credit_limit
        from seer.observability.exceptions import CreditLimitExceeded

        with (
            patch("seer.observability.credit_gate.get_effective_limits") as mock_get_limits,
            patch("seer.observability.credit_gate.get_effective_tier") as mock_resolve_tier,
            patch("seer.observability.credit_gate.get_5h_llm_credits_used") as mock_5h,
            patch("seer.observability.credit_gate.get_weekly_llm_credits_used") as mock_weekly,
            patch("seer.observability.credit_gate.get_monthly_llm_credits_used") as mock_monthly,
            patch("seer.observability.credit_gate._check_overage_allowance") as mock_overage,
        ):
            mock_limits = MagicMock()
            mock_limits.has_unlimited_credits = False
            mock_limits.has_unlimited_5h_credits = False
            mock_limits.has_unlimited_weekly_credits = False
            mock_limits.llm_credits_5h = 1.0
            mock_limits.llm_credits_weekly = 1.0
            mock_limits.llm_credits_monthly = 1.0
            mock_get_limits.return_value = mock_limits

            mock_resolve_tier.return_value = SubscriptionTier.FREE

            # Over monthly limit
            mock_5h.return_value = Decimal("0.50")
            mock_weekly.return_value = Decimal("0.90")
            mock_monthly.return_value = Decimal("1.20")

            # Overage not enabled
            from seer.observability.credit_gate import OverageCheckResult
            mock_overage.return_value = OverageCheckResult(
                allowed=False,
                overage_enabled=False,
                overage_cap_reached=False,
                remaining_cap_cents=0,
            )

            with pytest.raises(CreditLimitExceeded) as exc_info:
                await check_credit_limit(mock_user)

            assert exc_info.value.overage_enabled is False


@pytest.mark.asyncio
class TestCheckOverageAllowance:
    """Tests for _check_overage_allowance internal function."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        from seer.database import User

        user = MagicMock(spec=User)
        user.id = 1
        user.user_id = "user_123"
        return user

    async def test_returns_not_allowed_when_no_overage_settings(self, mock_user):
        """Test returns not allowed when user has no overage settings."""
        from seer.observability.credit_gate import _check_overage_allowance

        with patch(
            "seer.observability.credit_gate._get_effective_overage_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = None

            result = await _check_overage_allowance(
                user=mock_user,
                credits_used=Decimal("1.20"),
                subscription_limit=1.0,
            )

            assert result.allowed is False
            assert result.overage_enabled is False
            assert result.overage_cap_reached is False

    async def test_returns_allowed_when_under_spending_cap(self, mock_user):
        """Test returns allowed when under spending cap."""
        from seer.observability.credit_gate import _check_overage_allowance

        mock_settings = MagicMock()
        mock_settings.remaining_cap_cents = 5000  # $50 remaining

        with patch(
            "seer.observability.credit_gate._get_effective_overage_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = await _check_overage_allowance(
                user=mock_user,
                credits_used=Decimal("1.20"),
                subscription_limit=1.0,
            )

            assert result.allowed is True
            assert result.overage_enabled is True
            assert result.overage_cap_reached is False
            assert result.remaining_cap_cents == 5000

    async def test_returns_not_allowed_when_cap_reached(self, mock_user):
        """Test returns not allowed when spending cap is reached."""
        from seer.observability.credit_gate import _check_overage_allowance

        mock_settings = MagicMock()
        mock_settings.remaining_cap_cents = 0  # Cap exhausted

        with patch(
            "seer.observability.credit_gate._get_effective_overage_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = await _check_overage_allowance(
                user=mock_user,
                credits_used=Decimal("55.00"),
                subscription_limit=5.0,
            )

            assert result.allowed is False
            assert result.overage_enabled is True
            assert result.overage_cap_reached is True
            assert result.remaining_cap_cents == 0


# =============================================================================
# CreditLimitExceeded Overage Fields Tests
# =============================================================================


class TestCreditLimitExceededOverageFields:
    """Tests for CreditLimitExceeded overage-specific fields."""

    def test_overage_enabled_in_to_dict(self):
        """Test overage_enabled is included in to_dict output."""
        exc = CreditLimitExceeded(
            limit=5.0,
            current=7.0,
            tier=SubscriptionTier.PRO,
            period=LimitPeriod.MONTHLY,
            overage_enabled=True,
            overage_cap_reached=False,
        )
        data = exc.to_dict()

        assert "overage_enabled" in data
        assert data["overage_enabled"] is True
        assert data["overage_cap_reached"] is False

    def test_overage_cap_reached_message(self):
        """Test error message when overage cap is reached."""
        exc = CreditLimitExceeded(
            limit=5.0,
            current=55.0,
            tier=SubscriptionTier.PRO,
            period=LimitPeriod.MONTHLY,
            overage_enabled=True,
            overage_cap_reached=True,
        )

        assert "spending cap" in exc.message.lower()

    def test_overage_not_enabled_message(self):
        """Test error message suggests enabling overage pricing."""
        exc = CreditLimitExceeded(
            limit=5.0,
            current=7.0,
            tier=SubscriptionTier.PRO,
            period=LimitPeriod.MONTHLY,
            overage_enabled=False,
            overage_cap_reached=False,
        )

        assert "usage-based pricing" in exc.message.lower() or "upgrade" in exc.message.lower()
