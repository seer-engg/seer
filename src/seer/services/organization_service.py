"""
Organization service for creating and managing organizations.

This module provides business logic for organization operations,
separated from the API layer for reusability.
"""
import re
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

from tortoise.backends.base.client import BaseDBAsyncClient

from seer.database import Organization, OrganizationMembership, User
from seer.database.organization_models import (
    MembershipStatus,
    OrganizationRole,
    OrganizationType,
)
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.logger import get_logger

logger = get_logger(__name__)


def generate_slug(name: str, suffix: Optional[str] = None) -> str:
    """
    Generate a URL-friendly slug from a name.

    Args:
        name: The name to convert to a slug
        suffix: Optional suffix to append (e.g., random string for uniqueness)

    Returns:
        A lowercase, hyphenated slug
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = name.lower().strip()
    # Replace any non-alphanumeric characters (except hyphens) with hyphens
    slug = re.sub(r"[^a-z0-9-]", "-", slug)
    # Replace multiple hyphens with single hyphen
    slug = re.sub(r"-+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")

    if suffix:
        slug = f"{slug}-{suffix}"

    return slug[:255]  # Respect max length


def generate_unique_suffix() -> str:
    """Generate a short random suffix for slug uniqueness."""
    return secrets.token_hex(4)  # 8 hex characters


async def create_personal_organization(
    user: User,
) -> Tuple[Organization, OrganizationMembership]:
    """
    Create a personal organization for a user.

    This should be called when a new user signs up. Every user
    has exactly one personal organization.

    Args:
        user: The user to create the personal organization for

    Returns:
        Tuple of (Organization, OrganizationMembership)
    """
    # Generate org name from user info
    name_parts = []
    if user.first_name:
        name_parts.append(user.first_name)
    if not name_parts and user.email:
        # Use email prefix as fallback
        name_parts.append(user.email.split("@")[0])
    if not name_parts:
        name_parts.append("Personal")

    org_name = f"{' '.join(name_parts)}'s Workspace"

    # Generate unique slug
    slug = generate_slug(f"personal-{user.user_id}")

    # Create the organization
    organization = await Organization.create(
        name=org_name,
        slug=slug,
        type=OrganizationType.PERSONAL,
        owner=user,
        settings={},
    )

    # Create owner membership
    membership = await OrganizationMembership.create(
        organization=organization,
        user=user,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
        joined_at=datetime.now(timezone.utc),
    )

    # Create FREE subscription directly on organization
    await BillingSubscription.get_or_create(
        organization=organization,
        defaults={
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
        },
    )

    logger.info(
        "Created personal organization %s for user %s",
        organization.id,
        user.user_id,
    )

    return organization, membership


async def create_team_organization(
    owner: User,
    name: str,
    slug: Optional[str] = None,
    conn: BaseDBAsyncClient | None = None,
) -> Tuple[Organization, OrganizationMembership]:
    """
    Create a team organization.

    The owner's subscription will be transferred to the team.

    Args:
        owner: The user creating the team (becomes owner)
        name: Team name
        slug: Optional custom slug (generated from name if not provided)

    Returns:
        Tuple of (Organization, OrganizationMembership)

    Raises:
        ValueError: If slug is not unique
    """
    # Generate slug if not provided
    if not slug:
        base_slug = generate_slug(name)
        slug = base_slug

        # Check uniqueness and append suffix if needed
        existing_query = Organization.filter(slug=slug)
        if conn is not None:
            existing_query = existing_query.using_db(conn)
        existing = await existing_query.exists()
        if existing:
            slug = f"{base_slug}-{generate_unique_suffix()}"

    # Verify slug uniqueness
    slug_query = Organization.filter(slug=slug)
    if conn is not None:
        slug_query = slug_query.using_db(conn)
    if await slug_query.exists():
        raise ValueError(f"Organization slug already exists: {slug}")

    # Create the organization
    organization = await Organization.create(
        name=name,
        slug=slug,
        type=OrganizationType.TEAM,
        owner=owner,
        settings={},
        using_db=conn,
    )

    # Create owner membership
    membership = await OrganizationMembership.create(
        organization=organization,
        user=owner,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
        joined_at=datetime.now(timezone.utc),
        using_db=conn,
    )

    # V2: Create FREE subscription for team (checkout required for paid tier)
    # When transfer_subscription is used, this will be upgraded via transfer
    await BillingSubscription.get_or_create(
        organization=organization,
        defaults={
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
        },
        using_db=conn,
    )

    logger.info(
        "Created team organization %s (%s) for owner %s",
        organization.id,
        name,
        owner.user_id,
    )

    return organization, membership


async def convert_personal_to_team(
    organization: Organization,
    new_name: str,
) -> Organization:
    """
    Convert a personal organization to a team.

    This allows users to make their personal workspace collaborative.

    Args:
        organization: The personal organization to convert
        new_name: New name for the team

    Returns:
        Updated Organization

    Raises:
        ValueError: If organization is already a team
    """
    if organization.type == OrganizationType.TEAM:
        raise ValueError("Organization is already a team")

    organization.type = OrganizationType.TEAM
    organization.name = new_name
    await organization.save()

    logger.info(
        "Converted organization %s to team: %s",
        organization.id,
        new_name,
    )

    return organization


async def switch_user_organization(
    user: User,
    org_id: int,
) -> OrganizationMembership:
    """
    Switch a user to a different organization.

    Persists the active organization in the database so subsequent
    requests resolve org context without requiring a token refresh.

    Args:
        user: The user switching organizations
        org_id: ID of the organization to switch to

    Returns:
        The user's membership in the new organization

    Raises:
        ValueError: If user is not a member of the organization
    """
    # Validate membership
    membership = await OrganizationMembership.get_or_none(
        organization_id=org_id,
        user=user,
        status=MembershipStatus.ACTIVE,
    )

    if not membership:
        raise ValueError(f"User is not an active member of organization {org_id}")

    user.active_organization_id = org_id
    await user.save(update_fields=["active_organization_id"])

    await membership.fetch_related("organization")

    logger.info(
        "User %s switched to organization %s",
        user.user_id,
        org_id,
    )

    return membership


async def get_user_organizations(
    user: User,
) -> list[Tuple[Organization, OrganizationMembership]]:
    """
    Get all organizations a user is a member of.

    Args:
        user: The user to get organizations for

    Returns:
        List of (Organization, OrganizationMembership) tuples
    """
    memberships = await OrganizationMembership.filter(
        user=user,
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("organization")

    return [(m.organization, m) for m in memberships]


async def ensure_personal_organization(user: User) -> Tuple[Organization, OrganizationMembership]:
    """
    Ensure a user has a personal organization.

    If the personal org exists, returns it. Otherwise creates one.
    This is idempotent and safe to call multiple times.

    Args:
        user: The user to ensure has a personal org

    Returns:
        Tuple of (Organization, OrganizationMembership)
    """
    # Try to find existing personal org
    personal_org = await Organization.get_or_none(
        owner=user,
        type=OrganizationType.PERSONAL,
    )

    if personal_org:
        membership = await OrganizationMembership.get_or_none(
            organization=personal_org,
            user=user,
        )
        if membership:
            return personal_org, membership

        # Membership missing (shouldn't happen, but fix it)
        membership = await OrganizationMembership.create(
            organization=personal_org,
            user=user,
            role=OrganizationRole.OWNER,
            status=MembershipStatus.ACTIVE,
            joined_at=datetime.now(timezone.utc),
        )
        return personal_org, membership

    # Create personal org
    return await create_personal_organization(user)
