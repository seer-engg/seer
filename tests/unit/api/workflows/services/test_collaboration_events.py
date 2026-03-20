"""Tests for workflow collaboration event hooks."""

from unittest.mock import AsyncMock, patch

import pytest

from seer.api.workflows import models as api_models
from seer.api.workflows.services.lifecycle import update_workflow
from seer.database import Organization, OrganizationMembership, User, Workflow, WorkflowVersion, WorkflowVersionStatus
from seer.database.organization_models import MembershipStatus, OrganizationRole, OrganizationType
from seer.services.collaboration.models import CollaborationEventType


@pytest.mark.asyncio
async def test_update_workflow_publishes_collaboration_event(db_engine, sample_workflow_spec):
    user = await User.create(
        user_id="workflow_user",
        email="workflow@example.com",
        first_name="Workflow",
        last_name="User",
    )
    org = await Organization.create(
        name="Workflow Org",
        slug="workflow-org",
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
    workflow = await Workflow.create(user=user, organization=org, name="Original")
    await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=sample_workflow_spec,
        created_by=user,
        updated_by=user,
        spec_hash="hash",
        version_number=0,
    )

    with patch("seer.api.workflows.services.lifecycle.publish_collaboration_event", new=AsyncMock()) as publish_mock:
        response = await update_workflow(
            user,
            workflow.workflow_id,
            api_models.WorkflowUpdateRequest(name="Updated Workflow"),
            organization=org,
            membership=membership,
        )

    assert response.name == "Updated Workflow"
    publish_mock.assert_awaited_once()
    assert publish_mock.await_args.kwargs["organization_id"] == org.id
    assert publish_mock.await_args.kwargs["event_type"] == CollaborationEventType.WORKFLOW_UPDATED
    assert publish_mock.await_args.kwargs["resource_id"] == workflow.workflow_id
