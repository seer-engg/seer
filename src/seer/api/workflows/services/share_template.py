"""Service for sharing a workflow as a community template."""

import re
from typing import Optional

from pydantic import BaseModel

from seer.api.core.errors import VALIDATION_PROBLEM, raise_problem
from seer.database.models import User
from seer.database.profile_models import UserProfile
from seer.database.template_models import TemplateCategory, TemplateSource, WorkflowTemplate
from seer.database.workflow_models import Workflow, WorkflowVersion, WorkflowVersionStatus, parse_workflow_public_id
from seer.database.organization_models import Organization, OrganizationMembership, MembershipStatus


class ShareAsTemplateRequest(BaseModel):
    name: str
    description: str
    category: TemplateCategory
    tags: list[str] = []
    icon: Optional[str] = None
    visibility: str = "private"  # "private" | "public"


class ShareAsTemplateResponse(BaseModel):
    slug: str
    name: str
    public_url: str


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:80]


async def share_workflow_as_template(user: User, workflow_id: str, payload: ShareAsTemplateRequest) -> ShareAsTemplateResponse:
    # 1. Verify user has a profile with username
    profile = await UserProfile.filter(user=user).first()
    if not profile or not profile.username:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Profile required",
            detail="You must set up a profile with a username before sharing templates",
            status=400,
        )

    # 2. Get workflow owned by user
    internal_id = parse_workflow_public_id(workflow_id)
    workflow = await Workflow.filter(id=internal_id, user=user).first()
    if not workflow:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail="Workflow not found or you do not own it",
            status=404,
        )

    # 3. Get RELEASED version
    released_version = await WorkflowVersion.filter(
        workflow=workflow, status=WorkflowVersionStatus.RELEASED
    ).first()
    if not released_version:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not published",
            detail="You must publish your workflow before sharing it as a template",
            status=400,
        )

    # 4. Generate unique slug
    base_slug = _slugify(payload.name)
    slug = base_slug
    counter = 1
    while await WorkflowTemplate.filter(slug=slug).exists():
        slug = f"{base_slug}-{counter}"
        counter += 1

    # 5. Get user's organization
    membership = await OrganizationMembership.filter(user=user, status=MembershipStatus.ACTIVE).first()
    org = None
    if membership:
        org = await Organization.filter(id=membership.organization_id).first()

    # 6. Create template
    template = await WorkflowTemplate.create(
        slug=slug,
        name=payload.name,
        description=payload.description,
        category=payload.category,
        tags=payload.tags,
        icon=payload.icon,
        source=TemplateSource.COMMUNITY,
        created_by=user,
        organization=org,
        visibility=payload.visibility,
        spec=released_version.spec,
        is_published=payload.visibility == "public",
        source_workflow_id=workflow.id,
    )

    # 7. Return response
    return ShareAsTemplateResponse(
        slug=template.slug,
        name=template.name,
        public_url=f"https://getseer.dev/u/{profile.username}/{template.slug}",
    )
