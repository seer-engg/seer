"""
Usage limit service for resolving and retrieving tier-based limits.

This module provides functions to:
- Get limits for a specific subscription tier
- Get limits for a user (handles self-hosted vs cloud, subscription lookup)
- Resolve subscription tier for a user
- Compute billing periods for anniversary-based usage windows
"""
from calendar import monthrange
from datetime import datetime, timezone
from typing import Optional

from seer.config import config
from seer.database.models import User
from seer.database.subscription_models import (
    BillingProfile,
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.observability.models import (
    SELF_HOSTED_LIMITS,
    TIER_LIMITS_REGISTRY,
    TierLimits,
)


def get_limits_for_tier(tier: SubscriptionTier) -> TierLimits:
    """
    Get usage limits for a specific subscription tier.

    Args:
        tier: The subscription tier to get limits for

    Returns:
        TierLimits object with all limit dimensions

    Raises:
        KeyError: If tier is not found in registry
    """
    return TIER_LIMITS_REGISTRY[tier]


def _ensure_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetimes are timezone-aware in UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime while clamping the day to month length."""
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


async def get_billing_period_for_user(
    user: User,
    subscription: BillingSubscription | None = None,
    *,
    reference_now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return the current billing period window for a user.

    Priority:
    1) Use Stripe subscription current_period_start/end when present.
    2) Fall back to a signup-anniversary monthly window (for free users or missing Stripe data).
    3) As a safety net, use the current calendar month.
    """
    now = reference_now or datetime.now(timezone.utc)

    # Paid/Stripe-backed subscriptions: use Stripe period dates when valid.
    if not config.is_self_hosted:
        subscription = subscription or await BillingSubscription.get_or_none(
            billing_profile__owner_user=user
        )
        if subscription:
            start = _ensure_aware(subscription.current_period_start)
            end = _ensure_aware(subscription.current_period_end)
            if start and end and start <= now < end:
                return start, end

    # Free or missing Stripe period: align to signup anniversary month.
    created_at = _ensure_aware(user.created_at)
    if created_at:
        months_since_start = (now.year - created_at.year) * 12 + (now.month - created_at.month)
        if now.day < created_at.day:
            months_since_start -= 1
        months_since_start = max(months_since_start, 0)
        period_start = _add_months(created_at, months_since_start)
        period_end = _add_months(period_start, 1)
        return period_start, period_end

    # Fallback: calendar month
    period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if now.month == 12:
        period_end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        period_end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    return period_start, period_end


async def get_limits_for_user(user: User) -> TierLimits:
    """
    Get effective usage limits for a user.

    Handles:
    - Self-hosted mode: Returns unlimited/BYOK limits
    - Cloud mode: Looks up user's subscription tier and returns appropriate limits
    - Defaults to FREE tier if no subscription exists

    Args:
        user: The user to get limits for

    Returns:
        TierLimits object with effective limits for this user
    """
    # Self-hosted mode: return unlimited limits
    if config.is_self_hosted:
        return SELF_HOSTED_LIMITS

    # Cloud mode: resolve subscription tier
    tier = await resolve_user_tier(user)
    return get_limits_for_tier(tier)


async def resolve_user_tier(user: User) -> SubscriptionTier:
    """
    Resolve the active subscription tier for a user.

    Logic:
    1. Look up BillingProfile for user
    2. Get associated BillingSubscription
    3. Check subscription status (active, trialing, past_due)
    4. Return tier (defaults to FREE if no subscription)

    Args:
        user: The user to resolve tier for

    Returns:
        SubscriptionTier enum value
    """
    try:
        # Fetch billing profile with related subscription
        billing_profile = await BillingProfile.get_or_none(
            owner_user=user
        ).prefetch_related("subscription")

        if not billing_profile:
            # No billing profile -> FREE tier
            return SubscriptionTier.FREE

        # Get subscription (one-to-one relationship)
        subscription = await BillingSubscription.get_or_none(
            billing_profile=billing_profile
        )

        if not subscription:
            # Profile exists but no subscription -> FREE tier
            return SubscriptionTier.FREE

        # Check subscription status
        if subscription.status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
        ]:
            # Active or trialing subscriptions use their tier
            return subscription.tier

        if subscription.status == SubscriptionStatus.PAST_DUE:
            # Past due: allow grace period, still use their paid tier
            # (Could add stricter enforcement here if needed)
            return subscription.tier

        # Canceled or incomplete -> fall back to FREE
        return SubscriptionTier.FREE

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation on DB errors
        # If any error occurs during lookup, default to FREE tier for safety
        return SubscriptionTier.FREE


async def get_account_age_days(user: User) -> int:
    """
    Calculate the number of days since user account creation.

    Args:
        user: The user to calculate age for

    Returns:
        Number of days since account creation (rounded down)
    """
    now = datetime.now(timezone.utc)

    # Ensure created_at is timezone-aware
    created_at = user.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    delta = now - created_at
    return delta.days


async def is_trial_expired(user: User) -> bool:
    """
    Check if a user's trial period has expired.

    Only applies to Cloud FREE tier users. Returns False for:
    - Self-hosted mode
    - Paid tier users
    - Users within trial period

    Args:
        user: The user to check

    Returns:
        True if trial is expired, False otherwise
    """
    # Self-hosted: no trial limits
    if config.is_self_hosted:
        return False

    # Check user's tier
    tier = await resolve_user_tier(user)

    # Only FREE tier has trial limits
    if tier != SubscriptionTier.FREE:
        return False

    # Check account age against limit
    limits = get_limits_for_tier(tier)
    account_age = await get_account_age_days(user)

    return account_age > limits.account_day_limit


async def get_subscription_for_user(user: User) -> Optional[BillingSubscription]:
    """
    Get the active billing subscription for a user.

    Returns None if:
    - Self-hosted mode
    - No billing profile exists
    - No subscription exists

    Args:
        user: The user to get subscription for

    Returns:
        BillingSubscription if found, None otherwise
    """
    if config.is_self_hosted:
        return None

    try:
        billing_profile = await BillingProfile.get_or_none(
            owner_user=user
        ).prefetch_related("subscription")

        if not billing_profile:
            return None

        subscription = await BillingSubscription.get_or_none(
            billing_profile=billing_profile
        )

        return subscription

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None
