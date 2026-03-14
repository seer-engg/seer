"""
Usage limit service for resolving and retrieving tier-based limits.

This module provides functions to:
- Get limits for a specific subscription tier
- Get limits for a user (subscription lookup)
- Resolve subscription tier for a user
- Compute billing periods for anniversary-based usage windows
"""
# pylint: disable=too-many-lines  # Service module consolidates all tier/limit resolution logic
from calendar import monthrange
from datetime import datetime, timezone
from typing import Optional

from seer.database.models import User
from seer.database.organization_models import Organization, OrganizationType
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.observability.models import (
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
    # Get user's personal org subscription
    if not subscription:
        personal_org = await Organization.get_or_none(owner=user, type=OrganizationType.PERSONAL)
        if personal_org:
            subscription = await BillingSubscription.get_or_none(organization=personal_org)
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

    Looks up user's subscription tier and returns appropriate limits.
    Defaults to FREE tier if no subscription exists.

    Args:
        user: The user to get limits for

    Returns:
        TierLimits object with effective limits for this user
    """
    tier = await resolve_user_tier(user)
    return get_limits_for_tier(tier)


async def resolve_user_tier(user: User) -> SubscriptionTier:
    """
    Resolve the active subscription tier for a user via their personal org.

    Logic:
    1. Find user's personal organization
    2. Get organization's billing subscription
    3. Check subscription status (active, trialing, past_due)
    4. Return tier (defaults to FREE if no subscription)

    Args:
        user: The user to resolve tier for

    Returns:
        SubscriptionTier enum value
    """
    try:
        # Find user's personal organization
        personal_org = await Organization.get_or_none(
            owner=user,
            type=OrganizationType.PERSONAL
        )

        if not personal_org:
            return SubscriptionTier.FREE

        # Get organization's subscription
        subscription = await BillingSubscription.get_or_none(
            organization=personal_org
        )

        if not subscription:
            return SubscriptionTier.FREE

        # Check subscription status
        if subscription.status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
        ]:
            return subscription.tier

        if subscription.status == SubscriptionStatus.PAST_DUE:
            # Past due: allow grace period, still use their paid tier
            return subscription.tier

        # Canceled or incomplete -> fall back to FREE
        return SubscriptionTier.FREE

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation on DB errors
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

    Only applies to FREE tier users. Returns False for:
    - Paid tier users
    - Users within trial period

    Args:
        user: The user to check

    Returns:
        True if trial is expired, False otherwise
    """
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
    Get the active billing subscription for a user via their personal org.

    Returns None if:
    - No personal organization exists
    - No subscription exists

    Args:
        user: The user to get subscription for

    Returns:
        BillingSubscription if found, None otherwise
    """
    try:
        # Find user's personal organization
        personal_org = await Organization.get_or_none(
            owner=user,
            type=OrganizationType.PERSONAL
        )

        if not personal_org:
            return None

        subscription = await BillingSubscription.get_or_none(
            organization=personal_org
        )

        return subscription

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


# =============================================================================
# Organization-Scoped Functions
# =============================================================================


async def resolve_org_tier(organization: Organization) -> SubscriptionTier:
    """
    Resolve the active subscription tier for an organization.

    Uses the organization's direct billing_subscription FK.
    No fallback to owner's personal subscription - org's tier is authoritative.

    Args:
        organization: The organization to resolve tier for

    Returns:
        SubscriptionTier enum value
    """
    try:
        # Use Organization.billing_subscription FK directly
        subscription = await BillingSubscription.get_or_none(organization=organization)

        if not subscription:
            return SubscriptionTier.FREE

        # Check subscription status
        if subscription.status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
        ]:
            return subscription.tier

        if subscription.status == SubscriptionStatus.PAST_DUE:
            # Past due: allow grace period, still use their paid tier
            return subscription.tier

        # Canceled or incomplete -> fall back to FREE
        return SubscriptionTier.FREE

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation on DB errors
        return SubscriptionTier.FREE


async def get_limits_for_org(organization: Organization) -> TierLimits:
    """
    Get effective usage limits for an organization.

    Looks up organization's subscription tier and returns appropriate limits.
    Defaults to FREE tier if no subscription exists.

    Args:
        organization: The organization to get limits for

    Returns:
        TierLimits object with effective limits for this organization
    """
    tier = await resolve_org_tier(organization)
    return get_limits_for_tier(tier)


async def get_billing_period_for_org(
    organization: Organization,
    subscription: BillingSubscription | None = None,
    *,
    reference_now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return the current billing period window for an organization.

    Priority:
    1) Use Stripe subscription current_period_start/end when present.
    2) Fall back to a signup-anniversary monthly window (for free orgs or missing Stripe data).
    3) As a safety net, use the current calendar month.

    Args:
        organization: The organization to get billing period for
        subscription: Optional pre-fetched subscription
        reference_now: Optional reference datetime for testing

    Returns:
        Tuple of (period_start, period_end) datetimes
    """
    now = reference_now or datetime.now(timezone.utc)

    # Paid/Stripe-backed subscriptions: use Stripe period dates when valid.
    subscription = subscription or await get_subscription_for_org(organization)
    if subscription:
        start = _ensure_aware(subscription.current_period_start)
        end = _ensure_aware(subscription.current_period_end)
        if start and end and start <= now < end:
            return start, end

    # Free or missing Stripe period: align to org creation anniversary month.
    created_at = _ensure_aware(organization.created_at)
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


async def get_subscription_for_org(organization: Organization) -> Optional[BillingSubscription]:
    """
    Get the active billing subscription for an organization.

    Creates a default FREE tier subscription if none exists (lazy initialization).
    Uses the organization's direct billing_subscription FK.

    Args:
        organization: The organization to get subscription for

    Returns:
        BillingSubscription (existing or newly created)
    """
    try:
        # Get or create subscription directly via organization FK
        subscription, created = await BillingSubscription.get_or_create(
            organization=organization,
            defaults={
                "tier": SubscriptionTier.FREE,
                "status": SubscriptionStatus.ACTIVE,
            }
        )

        if created:
            from seer.logger import get_logger  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
            logger = get_logger(__name__)
            logger.info("Created default FREE subscription for organization %s", organization.id)

        return subscription

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


# =============================================================================
# Unified "Effective" Functions
# =============================================================================
# These determine whether to use org or user context based on org type.


async def get_effective_tier(user: User, organization: Optional[Organization]) -> SubscriptionTier:
    """
    Get the effective subscription tier considering organization context.

    Logic:
    - If organization is a team org: use organization's tier
    - Otherwise: use user's personal tier

    Args:
        user: The user to resolve tier for
        organization: Optional organization context

    Returns:
        SubscriptionTier enum value
    """
    if organization and organization.type == OrganizationType.TEAM:
        return await resolve_org_tier(organization)
    return await resolve_user_tier(user)


async def get_effective_limits(user: User, organization: Optional[Organization]) -> TierLimits:
    """
    Get the effective usage limits considering organization context.

    Logic:
    - If organization is a team org: use organization's limits
    - Otherwise: use user's personal limits

    Args:
        user: The user to get limits for
        organization: Optional organization context

    Returns:
        TierLimits object with effective limits
    """
    if organization and organization.type == OrganizationType.TEAM:
        return await get_limits_for_org(organization)
    return await get_limits_for_user(user)


async def get_effective_billing_period(
    user: User,
    organization: Optional[Organization],
    subscription: BillingSubscription | None = None,
    *,
    reference_now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """
    Get the effective billing period considering organization context.

    Logic:
    - If organization is a team org: use organization's billing period
    - Otherwise: use user's personal billing period

    Args:
        user: The user context
        organization: Optional organization context
        subscription: Optional pre-fetched subscription
        reference_now: Optional reference datetime for testing

    Returns:
        Tuple of (period_start, period_end) datetimes
    """
    if organization and organization.type == OrganizationType.TEAM:
        return await get_billing_period_for_org(organization, subscription, reference_now=reference_now)
    return await get_billing_period_for_user(user, subscription, reference_now=reference_now)


async def get_effective_subscription(
    user: User,
    organization: Optional[Organization],
) -> Optional[BillingSubscription]:
    """
    Get the effective subscription considering organization context.

    Logic:
    - If organization is a team org: use organization's subscription
    - Otherwise: use user's personal subscription

    Args:
        user: The user context
        organization: Optional organization context

    Returns:
        BillingSubscription if found, None otherwise
    """
    if organization and organization.type == OrganizationType.TEAM:
        return await get_subscription_for_org(organization)
    return await get_subscription_for_user(user)


# =============================================================================
# V2: Organization-Centric Functions (Post-Migration)
# =============================================================================
# These use the new Organization.billing_subscription FK directly,
# These use the Organization.billing_subscription FK directly.


async def resolve_org_tier_v2(organization: Organization) -> SubscriptionTier:
    """
    Resolve the active subscription tier for an organization using V2 model.

    V2 model uses Organization.billing_subscription FK directly.
    No fallback to owner's personal subscription - org's tier is authoritative.

    Args:
        organization: The organization to resolve tier for

    Returns:
        SubscriptionTier enum value
    """
    try:
        # Use Organization.billing_subscription FK directly
        subscription = await BillingSubscription.get_or_none(organization=organization)

        if not subscription:
            return SubscriptionTier.FREE

        # Check subscription status
        if subscription.status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
        ]:
            return subscription.tier

        if subscription.status == SubscriptionStatus.PAST_DUE:
            # Past due: allow grace period, still use their paid tier
            return subscription.tier

        # Canceled or incomplete -> fall back to FREE
        return SubscriptionTier.FREE

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation on DB errors
        return SubscriptionTier.FREE


async def get_subscription_for_org_v2(organization: Organization) -> Optional[BillingSubscription]:
    """
    Get the active billing subscription for an organization using V2 model.

    V2 model uses Organization.billing_subscription FK directly.
    Creates a default FREE tier subscription if none exists.

    Args:
        organization: The organization to get subscription for

    Returns:
        BillingSubscription (existing or newly created)
    """
    try:
        # Get or create subscription directly via organization FK
        subscription, created = await BillingSubscription.get_or_create(
            organization=organization,
            defaults={
                "tier": SubscriptionTier.FREE,
                "status": SubscriptionStatus.ACTIVE,
            }
        )

        if created:
            from seer.logger import get_logger  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
            logger = get_logger(__name__)
            logger.info("Created default FREE subscription for organization %s (V2)", organization.id)

        return subscription

    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


async def get_effective_tier_v2(organization: Organization) -> SubscriptionTier:
    """
    Get the effective subscription tier for an organization using V2 model.

    V2 model: All billing goes through Organization directly.
    Works for both PERSONAL and TEAM organizations.

    Args:
        organization: The organization to resolve tier for

    Returns:
        SubscriptionTier enum value
    """
    return await resolve_org_tier_v2(organization)


async def get_effective_limits_v2(organization: Organization) -> TierLimits:
    """
    Get the effective usage limits for an organization using V2 model.

    V2 model: All billing goes through Organization directly.

    Args:
        organization: The organization to get limits for

    Returns:
        TierLimits object with effective limits
    """
    tier = await resolve_org_tier_v2(organization)
    return get_limits_for_tier(tier)
