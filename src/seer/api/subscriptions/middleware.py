"""
Subscription guard middleware for protecting premium features.

Provides decorators to enforce minimum subscription tier requirements
on API endpoints.
"""
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request

from seer.database.subscription_models import SubscriptionStatus, SubscriptionTier

# Tier ordering for comparison
TIER_ORDER = {
    SubscriptionTier.FREE: 0,
    SubscriptionTier.PRO: 1,
    SubscriptionTier.PRO_PLUS: 2,
    SubscriptionTier.ULTRA: 3,
}


def require_subscription(min_tier: SubscriptionTier = SubscriptionTier.PRO) -> Callable:
    """
    Decorator to require minimum subscription tier for an endpoint.

    Usage:
        @router.post("/advanced-feature")
        @require_subscription(min_tier=SubscriptionTier.PRO_PLUS)
        async def advanced_feature(request: Request):
            # Only Pro+ and Ultra users can access this
            pass

    Args:
        min_tier: Minimum subscription tier required (default: PRO)

    Returns:
        Decorator function

    Raises:
        HTTPException 401: If user is not authenticated
        HTTPException 403: If subscription is not active or tier is insufficient
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, "db_user", None)
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Import here to avoid circular imports
            from seer.api.subscriptions.stripe_service import (  # pylint: disable=import-outside-toplevel
                get_user_subscription,  # Reason: Avoids circular import with stripe_service
            )

            subscription = await get_user_subscription(user)

            if subscription.status != SubscriptionStatus.ACTIVE:
                raise HTTPException(
                    status_code=403,
                    detail="Subscription is not active"
                )

            if TIER_ORDER[subscription.tier] < TIER_ORDER[min_tier]:
                raise HTTPException(
                    status_code=403,
                    detail=f"This feature requires {min_tier.value} subscription or higher"
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


async def check_subscription_tier(request: Request, min_tier: SubscriptionTier) -> bool:
    """
    Check if user has at least the specified subscription tier.

    For use in endpoint logic when you need conditional behavior
    rather than outright blocking.

    Args:
        request: FastAPI request with db_user in state
        min_tier: Minimum subscription tier to check for

    Returns:
        True if user meets tier requirement, False otherwise
    """
    user = getattr(request.state, "db_user", None)
    if not user:
        return False

    from seer.api.subscriptions.stripe_service import (  # pylint: disable=import-outside-toplevel
        get_user_subscription,  # Reason: Avoids circular import with stripe_service
    )

    subscription = await get_user_subscription(user)

    if subscription.status != SubscriptionStatus.ACTIVE:
        return False

    return TIER_ORDER[subscription.tier] >= TIER_ORDER[min_tier]
