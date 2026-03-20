"""Tests for consultant workflow read visibility."""

import pytest

from seer.api.core.permissions import can_view_workflow
from seer.api.workflows.services.lifecycle import list_workflows
from seer.api.workflows.services.shared import _can_view_workflow
from seer.database import Organization, OrganizationMembership, User, Workflow
from seer.database.organization_models import MembershipStatus, OrganizationRole, OrganizationType, WorkflowAssignment
from seer.database.workflow_models import WorkflowVisibility


async def _create_team_org(owner: User, slug: str) -> Organization:
    return await Organization.create(
        name=f"{slug} Org",
        slug=slug,
        type=OrganizationType.TEAM,
        owner=owner,
        settings={},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consultant_can_view_team_visible_workflow(db_engine):
    owner = await User.create(user_id="owner_user", email="owner@example.com", first_name="Owner", last_name="User")
    consultant = await User.create(user_id="consultant_user", email="consultant@example.com", first_name="Consultant", last_name="User")
    org = await _create_team_org(owner, "consultant-team-visible")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )
    workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Shared Workflow",
        visibility=WorkflowVisibility.TEAM,
    )

    assert await _can_view_workflow(consultant, workflow, membership) is True
    assert await can_view_workflow(consultant, workflow, membership) is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consultant_can_view_own_private_workflow(db_engine):
    consultant = await User.create(user_id="consultant_private_owner", email="consultant-private@example.com", first_name="Consultant", last_name="Owner")
    org = await _create_team_org(consultant, "consultant-own-private")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )
    workflow = await Workflow.create(
        user=consultant,
        organization=org,
        name="Private Workflow",
        visibility=WorkflowVisibility.PRIVATE,
    )

    assert await _can_view_workflow(consultant, workflow, membership) is True
    assert await can_view_workflow(consultant, workflow, membership) is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consultant_cannot_view_other_users_private_workflow(db_engine):
    owner = await User.create(user_id="private_owner", email="private-owner@example.com", first_name="Private", last_name="Owner")
    consultant = await User.create(user_id="private_consultant", email="private-consultant@example.com", first_name="Private", last_name="Consultant")
    org = await _create_team_org(owner, "consultant-private-blocked")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )
    workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Other Private Workflow",
        visibility=WorkflowVisibility.PRIVATE,
    )

    assert await _can_view_workflow(consultant, workflow, membership) is False
    assert await can_view_workflow(consultant, workflow, membership) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consultant_list_workflows_uses_org_visibility_rules(db_engine):
    owner = await User.create(user_id="list_owner", email="list-owner@example.com", first_name="List", last_name="Owner")
    consultant = await User.create(
        user_id="list_consultant",
        email="list-consultant@example.com",
        first_name="List",
        last_name="Consultant",
    )
    org = await _create_team_org(owner, "consultant-list-visibility")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )

    team_workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Team Workflow",
        visibility=WorkflowVisibility.TEAM,
    )
    own_private_workflow = await Workflow.create(
        user=consultant,
        organization=org,
        name="Own Private Workflow",
        visibility=WorkflowVisibility.PRIVATE,
    )
    blocked_private_workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Blocked Private Workflow",
        visibility=WorkflowVisibility.PRIVATE,
    )
    assigned_workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Assigned Workflow",
        visibility=WorkflowVisibility.ASSIGNED,
    )
    await WorkflowAssignment.create(
        workflow=assigned_workflow,
        user=consultant,
        organization=org,
        assigned_by=owner,
    )

    response = await list_workflows(consultant, organization=org, membership=membership)

    workflow_names = {item.name for item in response.items}
    assert workflow_names == {team_workflow.name, own_private_workflow.name, assigned_workflow.name}
    assert blocked_private_workflow.name not in workflow_names
