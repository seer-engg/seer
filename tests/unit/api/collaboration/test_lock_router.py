"""Tests for workflow collaboration lock endpoints."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from seer.api.collaboration.router import router
from seer.database import Organization, OrganizationMembership, User, Workflow
from seer.database.organization_models import MembershipStatus, OrganizationRole, OrganizationType, WorkflowAssignment
from seer.database.workflow_models import WorkflowVisibility
from seer.services.collaboration.models import WorkflowLockMetadata


async def _create_team_org(owner: User, slug: str) -> Organization:
    return await Organization.create(
        name=f"{slug} Org",
        slug=slug,
        type=OrganizationType.TEAM,
        owner=owner,
        settings={},
    )


@pytest.mark.asyncio
async def test_acquire_lock_returns_403_for_read_only_visible_workflow(db_engine, mock_app, api_client):
    owner = await User.create(user_id="owner_user", email="owner@example.com", first_name="Owner", last_name="User")
    consultant = await User.create(
        user_id="consultant_user",
        email="consultant@example.com",
        first_name="Consultant",
        last_name="User",
    )
    org = await _create_team_org(owner, "collab-lock-consultant")
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

    @mock_app.middleware("http")
    async def inject_request_state(request, call_next):
        request.state.db_user = consultant
        request.state.organization = org
        request.state.membership = membership
        request.state.user = SimpleNamespace(claims={"sub": consultant.user_id})
        return await call_next(request)

    mock_app.include_router(router)

    response = await api_client.post(f"/collaboration/workflows/{workflow.workflow_id}/lock", json={"tab_id": "tab-1"})

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have permission to edit this workflow"


@pytest.mark.asyncio
async def test_acquire_lock_keeps_404_for_non_viewable_workflow(db_engine, mock_app, api_client):
    owner = await User.create(user_id="private_owner", email="owner@example.com", first_name="Private", last_name="Owner")
    consultant = await User.create(
        user_id="private_consultant",
        email="consultant@example.com",
        first_name="Private",
        last_name="Consultant",
    )
    org = await _create_team_org(owner, "collab-lock-private")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )
    workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Private Workflow",
        visibility=WorkflowVisibility.PRIVATE,
    )

    @mock_app.middleware("http")
    async def inject_request_state(request, call_next):
        request.state.db_user = consultant
        request.state.organization = org
        request.state.membership = membership
        request.state.user = SimpleNamespace(claims={"sub": consultant.user_id})
        return await call_next(request)

    mock_app.include_router(router)

    response = await api_client.post(f"/collaboration/workflows/{workflow.workflow_id}/lock", json={"tab_id": "tab-1"})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_acquire_lock_allows_consultant_for_assigned_workflow(db_engine, mock_app, api_client):
    owner = await User.create(user_id="assigned_owner", email="owner@example.com", first_name="Assigned", last_name="Owner")
    consultant = await User.create(
        user_id="assigned_consultant",
        email="consultant@example.com",
        first_name="Assigned",
        last_name="Consultant",
    )
    org = await _create_team_org(owner, "collab-lock-assigned")
    membership = await OrganizationMembership.create(
        organization=org,
        user=consultant,
        role=OrganizationRole.CONSULTANT,
        status=MembershipStatus.ACTIVE,
    )
    workflow = await Workflow.create(
        user=owner,
        organization=org,
        name="Assigned Workflow",
        visibility=WorkflowVisibility.ASSIGNED,
    )
    await WorkflowAssignment.create(
        workflow=workflow,
        user=consultant,
        organization=org,
        assigned_by=owner,
    )

    @mock_app.middleware("http")
    async def inject_request_state(request, call_next):
        request.state.db_user = consultant
        request.state.organization = org
        request.state.membership = membership
        request.state.user = SimpleNamespace(claims={"sub": consultant.user_id})
        request.state.correlation_id = "corr-1"
        return await call_next(request)

    mock_app.include_router(router)

    lock = WorkflowLockMetadata(
        organization_id=org.id,
        workflow_id=workflow.workflow_id,
        holder_clerk_user_id=consultant.user_id,
        holder_db_user_id=consultant.id,
        holder_name="Assigned Consultant",
        acquired_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        tab_id="tab-1",
    )

    with patch("seer.api.collaboration.router.WorkflowLockService.acquire_lock", new=AsyncMock(return_value=(True, lock))) as acquire_mock:
        with patch("seer.api.collaboration.router.WorkflowLockService.close", new=AsyncMock()):
            with patch("seer.api.collaboration.router.publish_collaboration_event", new=AsyncMock()):
                response = await api_client.post(
                    f"/collaboration/workflows/{workflow.workflow_id}/lock",
                    json={"tab_id": "tab-1"},
                )

    assert response.status_code == 200
    assert response.json()["lock"]["holder_db_user_id"] == consultant.id
    acquire_mock.assert_awaited_once()
