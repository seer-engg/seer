"""
Credit gate for checking LLM usage limits before execution.
"""
import logging
from decimal import Decimal

from seer.database.models import User
from seer.observability.constants import tiered_usage_limits
from seer.observability.exceptions import CreditLimitExceeded
from seer.observability.service import get_limits_for_user, resolve_user_tier
from seer.observability.tracking import get_monthly_llm_credits_used

logger = logging.getLogger(__name__)


async def check_credit_limit(user: User) -> None:
    """
    Check if user has sufficient LLM credits before execution.

    Raises:
        CreditLimitExceeded: If user is at or over 120% of monthly limit

    Logs warning if user is at or over 80% of monthly limit.
    """
    # Get user's tier limits
    limits = await get_limits_for_user(user)

    # Skip check if unlimited credits (self-hosted or BYOK)
    if limits.has_unlimited_credits:
        return

    # Get current usage
    credits_used = await get_monthly_llm_credits_used(user)
    monthly_limit = Decimal(str(limits.llm_credits_monthly))

    # Calculate thresholds
    soft_limit = monthly_limit * Decimal(str(tiered_usage_limits.CREDIT_WARNING_THRESHOLD))  # 80%
    hard_limit = monthly_limit * Decimal(str(tiered_usage_limits.CREDIT_BLOCK_THRESHOLD))  # 120%

    # Hard block at 120%
    if credits_used >= hard_limit:
        tier = await resolve_user_tier(user)
        raise CreditLimitExceeded(
            limit=float(monthly_limit),
            current=float(credits_used),
            tier=tier,
            is_soft_limit=False,
        )

    # Soft warning at 80%
    if credits_used >= soft_limit:
        percentage = (credits_used / monthly_limit) * Decimal("100")
        logger.warning(
            "User %s approaching LLM credit limit: $%.2f / $%.2f (%.1f%%)",
            user.user_id,
            credits_used,
            monthly_limit,
            percentage,
            extra={
                "user_id": user.user_id,
                "credits_used": float(credits_used),
                "monthly_limit": float(monthly_limit),
                "percentage": float(percentage),
            },
        )
