from datetime import datetime, timezone

import pytest

from seer.database.organization_models import (
    MembershipStatus,
    Organization,
    OrganizationMembership,
    OrganizationRole,
    OrganizationType,
)
from seer.services.organization_service import switch_user_organization


@pytest.mark.integration
@pytest.mark.asyncio
async def test_switch_user_organization_persists_active_org(db_engine, test_user):
    """Switching organizations should persist the user's active organization."""
    organization = await Organization.create(
        name="Team Org",
        slug="team-org-switch-user",
        type=OrganizationType.TEAM,
        owner=test_user,
        settings={},
    )
    await OrganizationMembership.create(
        organization=organization,
        user=test_user,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
        joined_at=datetime.now(timezone.utc),
    )

    membership = await switch_user_organization(test_user, organization.id)
    await test_user.refresh_from_db()

    assert membership.organization.id == organization.id
    assert test_user.active_organization_id == organization.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_switch_user_organization_rejects_non_member(db_engine, test_user):
    """Switching should fail when the user is not an active member."""
    organization = await Organization.create(
        name="Forbidden Org",
        slug="forbidden-org-switch-user",
        type=OrganizationType.TEAM,
        owner=test_user,
        settings={},
    )

    with pytest.raises(ValueError, match="not an active member"):
        await switch_user_organization(test_user, organization.id)
