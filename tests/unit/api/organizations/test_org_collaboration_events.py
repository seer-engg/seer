"""Tests for organization collaboration event hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from seer.api.organizations.router import router
from seer.database import Organization, OrganizationMembership, User
from seer.database.organization_models import MembershipStatus, OrganizationRole, OrganizationType
from seer.services.collaboration.models import CollaborationEventType


@pytest.mark.asyncio
async def test_create_invitation_publishes_collaboration_event(db_engine, mock_app, api_client):
    user = await User.create(
        user_id="org_user",
        email="owner@example.com",
        first_name="Owner",
        last_name="User",
    )
    org = await Organization.create(
        name="Org",
        slug="org-slug",
        type=OrganizationType.TEAM,
        owner=user,
        settings={},
    )
    membership = await OrganizationMembership.create(
        organization=org,
        user=user,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
    )

    @mock_app.middleware("http")
    async def inject_request_state(request, call_next):
        request.state.db_user = user
        request.state.organization = org
        request.state.membership = membership
        request.state.correlation_id = "corr-1"
        request.state.user = SimpleNamespace(claims={"sub": user.user_id})
        return await call_next(request)

    mock_app.include_router(router)

    with patch("seer.api.organizations.router.send_invitation_email", new=AsyncMock()):
        with patch("seer.api.organizations.router.publish_collaboration_event", new=AsyncMock()) as publish_mock:
            response = await api_client.post(
                f"/organizations/{org.id}/invitations",
                json={"email": "invitee@example.com", "role": "user"},
            )

    assert response.status_code == 201
    publish_mock.assert_awaited_once()
    assert publish_mock.await_args.kwargs["organization_id"] == org.id
    assert publish_mock.await_args.kwargs["event_type"] == CollaborationEventType.INVITATION_CREATED
