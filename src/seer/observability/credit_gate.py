"""
Credit gate for checking LLM usage limits before execution.

Checks three rolling window limits in order:
1. 5-hour limit (burst protection)
2. Weekly limit (prevent front-loading)
3. Monthly limit (billing period cap)

For paid tier users with overage enabled, usage beyond 100% of monthly
limit is allowed up to the spending cap.
"""
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from seer.database.models import User
from seer.database.overage_models import OverageSettings
from seer.database.subscription_models import BillingProfile, SubscriptionTier
from seer.observability.constants import tiered_usage_limits
from seer.observability.exceptions import CreditLimitExceeded, LimitPeriod
from seer.observability.service import get_limits_for_user, resolve_user_tier
from seer.observability.tracking import (
    get_5h_llm_credits_used,
    get_monthly_llm_credits_used,
    get_weekly_llm_credits_used,
)

logger = logging.getLogger(__name__)


@dataclass
class OverageCheckResult:
    """Result of checking overage allowance."""

    allowed: bool
    overage_enabled: bool
    overage_cap_reached: bool
    remaining_cap_cents: int


async def _get_user_overage_settings(user: User) -> Optional[OverageSettings]:
    """
    Get overage settings for a user if they exist and are enabled.

    Args:
        user: The user to check.

    Returns:
        OverageSettings if enabled, None otherwise.
    """
    try:
        billing_profile = await BillingProfile.get_or_none(owner_user=user)
        if not billing_profile:
            return None

        overage_settings = await OverageSettings.get_or_none(
            billing_profile=billing_profile,
            enabled=True,
        )
        return overage_settings
    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


async def _check_overage_allowance(
    user: User,
    credits_used: Decimal,
    subscription_limit: float,
) -> OverageCheckResult:
    """
    Check if overage is enabled and has remaining capacity.

    Called when monthly usage exceeds the subscription limit.

    Args:
        user: The user to check.
        credits_used: Current credits used in USD.
        subscription_limit: Monthly subscription credit limit in USD.

    Returns:
        OverageCheckResult with allowance status.
    """
    overage_settings = await _get_user_overage_settings(user)

    if not overage_settings:
        return OverageCheckResult(
            allowed=False,
            overage_enabled=False,
            overage_cap_reached=False,
            remaining_cap_cents=0,
        )

    # Calculate how much is in overage territory (for future use in detailed logging)
    _ = float(credits_used) - subscription_limit  # overage_amount, reserved for future use

    # Check if adding this would exceed the spending cap
    remaining = overage_settings.remaining_cap_cents

    if remaining <= 0:
        return OverageCheckResult(
            allowed=False,
            overage_enabled=True,
            overage_cap_reached=True,
            remaining_cap_cents=0,
        )

    return OverageCheckResult(
        allowed=True,
        overage_enabled=True,
        overage_cap_reached=False,
        remaining_cap_cents=remaining,
    )


async def _check_single_limit(
    user: User,
    limit: float,
    period: LimitPeriod,
    tier: SubscriptionTier,
    get_usage_fn: Callable[[User], Awaitable[Decimal]],
) -> None:
    """
    Check a single credit limit and raise/warn as appropriate.

    Args:
        user: The user to check
        limit: The credit limit for this period
        period: The time period (5-hour, weekly, or monthly)
        tier: The user's subscription tier
        get_usage_fn: Async function to retrieve current usage for this period

    Raises:
        CreditLimitExceeded: If user is at or over the hard threshold (120%)
    """
    credits_used = await get_usage_fn(user)
    limit_decimal = Decimal(str(limit))

    soft_threshold = limit_decimal * Decimal(str(tiered_usage_limits.CREDIT_WARNING_THRESHOLD))  # 80%
    hard_threshold = limit_decimal * Decimal(str(tiered_usage_limits.CREDIT_BLOCK_THRESHOLD))  # 120%

    # Hard block at 120%
    if credits_used >= hard_threshold:
        raise CreditLimitExceeded(
            limit=float(limit),
            current=float(credits_used),
            tier=tier,
            period=period,
            is_soft_limit=False,
        )

    # Soft warning at 80%
    if credits_used >= soft_threshold:
        percentage = (credits_used / limit_decimal) * Decimal("100")
        logger.warning(
            "User %s approaching %s LLM credit limit: $%.2f / $%.2f (%.1f%%)",
            user.user_id,
            period.value,
            credits_used,
            limit,
            percentage,
            extra={
                "user_id": user.user_id,
                "period": period.value,
                "credits_used": float(credits_used),
                "limit": float(limit),
                "percentage": float(percentage),
            },
        )


async def _check_monthly_limit_with_overage(
    user: User,
    limit: float,
    tier: SubscriptionTier,
) -> None:
    """
    Check monthly credit limit with overage support for paid tier users.

    Flow:
    1. Check if under 100% of subscription limit → allow
    2. Check if overage enabled + under spending cap → allow
    3. Otherwise → raise CreditLimitExceeded

    Args:
        user: The user to check
        limit: The monthly credit limit
        tier: The user's subscription tier

    Raises:
        CreditLimitExceeded: If limit exceeded and overage not available/exhausted
    """
    credits_used = await get_monthly_llm_credits_used(user)
    limit_decimal = Decimal(str(limit))

    # Under 100% of limit: always allow (existing behavior with buffer up to 120%)
    hard_threshold = limit_decimal * Decimal(str(tiered_usage_limits.CREDIT_BLOCK_THRESHOLD))  # 120%

    if credits_used < hard_threshold:
        # Warn at 80%
        soft_threshold = limit_decimal * Decimal(str(tiered_usage_limits.CREDIT_WARNING_THRESHOLD))
        if credits_used >= soft_threshold:
            percentage = (credits_used / limit_decimal) * Decimal("100")
            logger.warning(
                "User %s approaching monthly LLM credit limit: $%.2f / $%.2f (%.1f%%)",
                user.user_id,
                credits_used,
                limit,
                percentage,
            )
        return

    # At or over 120%: check if overage is available
    overage_result = await _check_overage_allowance(user, credits_used, limit)

    if overage_result.allowed:
        # Overage enabled and has remaining capacity
        logger.info(
            "User %s using overage credits: $%.2f used, $%.2f remaining in cap",
            user.user_id,
            credits_used,
            overage_result.remaining_cap_cents / 100,
        )
        return

    # Overage not available or exhausted
    raise CreditLimitExceeded(
        limit=float(limit),
        current=float(credits_used),
        tier=tier,
        period=LimitPeriod.MONTHLY,
        is_soft_limit=False,
        overage_enabled=overage_result.overage_enabled,
        overage_cap_reached=overage_result.overage_cap_reached,
    )


async def check_credit_limit(user: User) -> None:
    """
    Check if user has sufficient LLM credits before execution.

    Checks limits from most restrictive to least restrictive:
    1. 5-hour rolling window (burst protection)
    2. Weekly rolling window (prevent front-loading)
    3. Monthly billing period (with overage support for paid tiers)

    This ordering ensures users get the most specific error message
    about which limit they've exceeded.

    For monthly limits, paid tier users with overage enabled can exceed
    their subscription limit up to their spending cap.

    Raises:
        CreditLimitExceeded: If user is at or over 120% of any limit
                            (or overage cap reached for monthly)

    Logs warning if user is at or over 80% of any limit.
    """
    limits = await get_limits_for_user(user)

    # Skip all checks if unlimited credits (self-hosted or BYOK)
    if limits.has_unlimited_credits:
        return

    tier = await resolve_user_tier(user)

    # Check 5-hour limit (burst protection) - no overage support
    if not limits.has_unlimited_5h_credits:
        await _check_single_limit(
            user=user,
            limit=limits.llm_credits_5h,
            period=LimitPeriod.FIVE_HOUR,
            tier=tier,
            get_usage_fn=get_5h_llm_credits_used,
        )

    # Check weekly limit - no overage support
    if not limits.has_unlimited_weekly_credits:
        await _check_single_limit(
            user=user,
            limit=limits.llm_credits_weekly,
            period=LimitPeriod.WEEKLY,
            tier=tier,
            get_usage_fn=get_weekly_llm_credits_used,
        )

    # Check monthly limit with overage support for paid tiers
    await _check_monthly_limit_with_overage(
        user=user,
        limit=limits.llm_credits_monthly,
        tier=tier,
    )
