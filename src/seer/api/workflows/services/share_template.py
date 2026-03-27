"""Service for sharing a workflow as a community template."""

from __future__ import annotations

import json
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


class ShareAsTemplateResponse(BaseModel):
    slug: str
    name: str
    public_url: str
    is_update: bool = False


class WorkflowTemplateMetaResponse(BaseModel):
    slug: str
    name: str
    description: str
    category: str
    tags: list[str]
    icon: Optional[str]
    public_url: str


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:80]


async def get_workflow_template_meta(user: User, workflow_id: str) -> Optional[WorkflowTemplateMetaResponse]:
    """Get existing template metadata for a workflow, if any."""
    internal_id = parse_workflow_public_id(workflow_id)
    workflow = await Workflow.filter(id=internal_id, user=user).first()
    if not workflow:
        return None

    template = await WorkflowTemplate.filter(source_workflow_id=workflow.id, created_by=user).first()
    if not template:
        return None

    profile = await UserProfile.filter(user=user).first()
    username = profile.username if profile else "unknown"

    return WorkflowTemplateMetaResponse(
        slug=template.slug,
        name=template.name,
        description=template.description,
        category=template.category.value if hasattr(template.category, "value") else template.category,
        tags=template.tags or [],
        icon=template.icon,
        public_url=f"https://getseer.dev/u/{username}/{template.slug}",
    )


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

    # 3. Get spec: prefer DRAFT (current canvas), fall back to RELEASED
    draft_version = await WorkflowVersion.filter(
        workflow=workflow, status=WorkflowVersionStatus.DRAFT
    ).first()
    released_version = await WorkflowVersion.filter(
        workflow=workflow, status=WorkflowVersionStatus.RELEASED
    ).first()
    if not draft_version and not released_version:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not published",
            detail="You must publish your workflow before sharing it as a template",
            status=400,
        )
    spec_version = draft_version or released_version

    # 4. Auto-detect required integrations
    # pylint: disable-next=import-outside-toplevel # circular import avoidance
    from seer.api.templates.admin_services import _detect_required_integrations
    required_integrations = _detect_required_integrations(spec_version.spec)

    # 5. Check for existing template from this workflow
    existing = await WorkflowTemplate.filter(source_workflow_id=workflow.id, created_by=user).first()

    if existing:
        # UPDATE existing template
        existing.name = payload.name
        existing.description = payload.description
        existing.category = payload.category
        existing.tags = payload.tags
        existing.icon = payload.icon
        existing.spec = spec_version.spec
        existing.required_integrations = required_integrations
        await existing.save(update_fields=[
            "name", "description", "category", "tags", "icon",
            "spec", "required_integrations",
        ])
        template = existing
        is_update = True
    else:
        # CREATE new template
        base_slug = _slugify(payload.name)
        slug = base_slug
        counter = 1
        while await WorkflowTemplate.filter(slug=slug).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1

        membership = await OrganizationMembership.filter(user=user, status=MembershipStatus.ACTIVE).first()
        org = None
        if membership:
            org = await Organization.filter(id=membership.organization_id).first()

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
            spec=spec_version.spec,
            required_integrations=required_integrations,
            source_workflow_id=workflow.id,
        )
        is_update = False

    return ShareAsTemplateResponse(
        slug=template.slug,
        name=template.name,
        public_url=f"https://getseer.dev/u/{profile.username}/{template.slug}",
        is_update=is_update,
    )


class GenerateTemplateDescriptionResponse(BaseModel):
    name: str
    description: str
    category: str
    tags: list[str]


def _summarize_node(node: dict) -> str:
    """Build a one-line summary of a workflow node for the LLM prompt."""
    data = node.get("data", {})
    parts = [f"type: {data.get('type', 'unknown')}"]
    if data.get("toolId"):
        parts.append(f"tool: {data['toolId']}")
    if data.get("triggerId"):
        parts.append(f"trigger: {data['triggerId']}")
    label = data.get("label", node.get("id", ""))
    return f"- {label} ({', '.join(parts)})"


_TEMPLATE_GEN_SYSTEM_PROMPT = """You are a workflow automation expert writing template descriptions for a marketplace.

Given a workflow's structure (nodes, tools, triggers), generate:
1. A clean template name (2-6 words, title case)
2. A compelling description (1-3 sentences) explaining what the workflow does and who it's for
3. A category from the allowed list
4. 2-5 relevant tags (lowercase, hyphenated)

Respond with JSON only:
{
  "name": "...",
  "description": "...",
  "category": "...",
  "tags": ["..."]
}"""


async def generate_template_description(user: User, workflow_id: str) -> GenerateTemplateDescriptionResponse:
    """Use LLM to generate template metadata from the workflow spec."""
    # pylint: disable=import-outside-toplevel,too-many-locals # same pattern as generate_schema_metadata
    from langchain_core.messages import HumanMessage, SystemMessage

    from seer.llm import get_llm
    from seer.logger import get_logger

    logger = get_logger("api.workflows.template_description")

    internal_id = parse_workflow_public_id(workflow_id)
    workflow = await Workflow.filter(id=internal_id, user=user).first()
    if not workflow:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Workflow not found",
                      detail="Workflow not found or you do not own it", status=404)

    draft = await WorkflowVersion.filter(workflow=workflow, status=WorkflowVersionStatus.DRAFT).first()
    released = await WorkflowVersion.filter(workflow=workflow, status=WorkflowVersionStatus.RELEASED).first()
    spec_version = draft or released
    if not spec_version:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="No workflow spec",
                      detail="Workflow has no draft or released version", status=400)

    nodes = spec_version.spec.get("graph", {}).get("nodes", [])
    node_lines = "\n".join(_summarize_node(n) for n in nodes) or "(empty workflow)"
    categories = ", ".join(c.value for c in TemplateCategory)

    user_prompt = (
        f"Workflow name: {workflow.name}\nNode count: {len(nodes)}\n"
        f"Nodes:\n{node_lines}\n\nAllowed categories: {categories}\n\nGenerate template metadata:"
    )

    try:
        llm = get_llm(model="qwen/qwen3-30b-a3b", temperature=0.3)
        response = await llm.ainvoke([
            SystemMessage(content=_TEMPLATE_GEN_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # LLM may wrap JSON in markdown code fences — strip them and retry
            if "```" in raw:
                start = raw.find("```json")
                if start != -1:
                    start += 7
                else:
                    start = raw.find("```") + 3
                end = raw.find("```", start)
                if end > start:
                    raw = raw[start:end].strip()
            result = json.loads(raw)
        return GenerateTemplateDescriptionResponse(
            name=result.get("name", workflow.name),
            description=result.get("description", ""),
            category=result.get("category", "other"),
            tags=result.get("tags", []),
        )
    except Exception:  # pylint: disable=broad-exception-caught # fallback on any LLM failure
        logger.exception("Template description generation failed")
        return GenerateTemplateDescriptionResponse(
            name=workflow.name, description="", category="other", tags=[],
        )
