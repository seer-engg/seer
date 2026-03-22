"""Public API endpoints — no authentication required."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from tortoise.expressions import Q

from seer.database.profile_models import UserProfile, validate_username
from seer.database.template_models import TemplateSource, WorkflowTemplate

router = APIRouter(prefix="/public", tags=["public"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PublicCreatorSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str
    display_name: str | None
    bio: str | None
    avatar_url: str | None
    tags: list[str]
    template_count: int


class PublicTemplateSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    name: str
    description: str
    category: str
    tags: list[str]
    icon: str | None
    usage_count: int
    creator_username: str | None


class PublicCreatorDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str
    display_name: str | None
    bio: str | None
    avatar_url: str | None
    tags: list[str]
    template_count: int
    social_links: dict
    templates: list[PublicTemplateSummary]


class PublicTemplateDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    name: str
    description: str
    category: str
    tags: list[str]
    icon: str | None
    usage_count: int
    creator_username: str | None
    required_integrations: list
    creator: PublicCreatorSummary | None


class PreviewNode(BaseModel):
    id: str
    type: str
    label: str
    position: dict | None = None


class PreviewEdge(BaseModel):
    source: str
    target: str
    type: str | None = None


class PreviewTrigger(BaseModel):
    id: str
    key: str
    mode: str
    ui_meta: dict | None = None


class PreviewMetadata(BaseModel):
    name: str
    description: str
    icon: str | None = None


class PublicWorkflowPreview(BaseModel):
    nodes: list[PreviewNode]
    edges: list[PreviewEdge]
    triggers: list[PreviewTrigger]
    metadata: PreviewMetadata


class CreatorListResponse(BaseModel):
    creators: list[PublicCreatorSummary]
    total: int
    limit: int
    offset: int


class TemplateListResponse(BaseModel):
    templates: list[PublicTemplateSummary]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _template_count_for_profile(profile: UserProfile) -> int:
    return await WorkflowTemplate.filter(
        created_by_id=profile.user_id,
        source=TemplateSource.COMMUNITY,

    ).count()


async def _creator_summary(profile: UserProfile) -> PublicCreatorSummary:
    return PublicCreatorSummary(
        username=profile.username,
        display_name=profile.display_name,
        bio=profile.bio,
        avatar_url=profile.avatar_url,
        tags=profile.tags or [],
        template_count=await _template_count_for_profile(profile),
    )


async def _template_summary(t: WorkflowTemplate, username_map: dict[int, str] | None = None) -> PublicTemplateSummary:
    creator_username: str | None = None
    if username_map is not None and t.created_by_id:
        creator_username = username_map.get(t.created_by_id)
    elif t.created_by_id:
        profile = await UserProfile.filter(user_id=t.created_by_id).first()
        creator_username = profile.username if profile else None
    return PublicTemplateSummary(
        slug=t.slug,
        name=t.name,
        description=t.description,
        category=t.category.value if hasattr(t.category, "value") else str(t.category),
        tags=t.tags or [],
        icon=t.icon,
        usage_count=t.usage_count,
        creator_username=creator_username,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/username-available/{username}")
async def check_username_available(username: str) -> dict:
    try:
        validated = validate_username(username)
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}
    exists = await UserProfile.filter(username=validated).exists()
    if exists:
        return {"available": False, "reason": "Username is already taken"}
    return {"available": True, "reason": None}


@router.get("/creators", response_model=CreatorListResponse)
async def list_creators(
    tag: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> CreatorListResponse:
    qs = UserProfile.filter(is_public=True)
    if search:
        qs = qs.filter(Q(username__icontains=search) | Q(display_name__icontains=search))

    all_profiles = await qs.offset(offset).limit(limit)

    # Tag filtering on JSONField — filter in Python
    if tag:
        all_profiles = [p for p in all_profiles if tag in (p.tags or [])]

    total_qs = UserProfile.filter(is_public=True)
    if search:
        total_qs = total_qs.filter(Q(username__icontains=search) | Q(display_name__icontains=search))
    total = await total_qs.count()

    creators = [await _creator_summary(p) for p in all_profiles]
    return CreatorListResponse(creators=creators, total=total, limit=limit, offset=offset)


@router.get("/creators/{username}", response_model=PublicCreatorDetail)
async def get_creator(username: str) -> PublicCreatorDetail:
    profile = await UserProfile.filter(username=username, is_public=True).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Creator not found")

    templates = await WorkflowTemplate.filter(
        created_by_id=profile.user_id,
        source=TemplateSource.COMMUNITY,

    )
    template_summaries = [await _template_summary(t, {profile.user_id: profile.username}) for t in templates]

    return PublicCreatorDetail(
        username=profile.username,
        display_name=profile.display_name,
        bio=profile.bio,
        avatar_url=profile.avatar_url,
        tags=profile.tags or [],
        template_count=len(template_summaries),
        social_links=profile.social_links or {},
        templates=template_summaries,
    )


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TemplateListResponse:
    qs = WorkflowTemplate.filter(source=TemplateSource.COMMUNITY)
    if category:
        qs = qs.filter(category=category)
    if search:
        qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))

    total = await qs.count()
    templates = await qs.offset(offset).limit(limit)

    # Build username map
    user_ids = {t.created_by_id for t in templates if t.created_by_id}
    username_map: dict[int, str] = {}
    if user_ids:
        profiles = await UserProfile.filter(user_id__in=list(user_ids))
        username_map = {p.user_id: p.username for p in profiles}

    summaries = [await _template_summary(t, username_map) for t in templates]
    return TemplateListResponse(templates=summaries, total=total, limit=limit, offset=offset)


@router.get("/templates/{slug}", response_model=PublicTemplateDetail)
async def get_template(slug: str) -> PublicTemplateDetail:
    t = await WorkflowTemplate.filter(slug=slug, source=TemplateSource.COMMUNITY).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")

    creator: PublicCreatorSummary | None = None
    creator_username: str | None = None
    if t.created_by_id:
        profile = await UserProfile.filter(user_id=t.created_by_id).first()
        if profile:
            creator_username = profile.username
            creator = await _creator_summary(profile)

    return PublicTemplateDetail(
        slug=t.slug,
        name=t.name,
        description=t.description,
        category=t.category.value if hasattr(t.category, "value") else str(t.category),
        tags=t.tags or [],
        icon=t.icon,
        usage_count=t.usage_count,
        creator_username=creator_username,
        required_integrations=t.required_integrations or [],
        creator=creator,
    )


def _sanitize_spec(spec: dict, name: str, description: str, icon: str | None) -> PublicWorkflowPreview:
    """Strip sensitive data from a workflow spec for public preview."""
    nodes = []
    for n in spec.get("nodes", []):
        pos = n.get("ui", {}).get("position") if isinstance(n.get("ui"), dict) else None
        label = n.get("id", "")
        # Use tool name as label for tool nodes
        if n.get("type") == "tool" and n.get("tool"):
            label = n["tool"]
        nodes.append(PreviewNode(id=n.get("id", ""), type=n.get("type", ""), label=label, position=pos))

    edges = []
    for e in spec.get("edges", []):
        edges.append(PreviewEdge(source=e["source"], target=e["target"], type=e.get("type")))

    triggers = []
    for t in spec.get("triggers", []):
        triggers.append(PreviewTrigger(
            id=t.get("id", ""),
            key=t.get("key", ""),
            mode=t.get("mode", ""),
            ui_meta={"position": t["ui_meta"]["position"]} if t.get("ui_meta", {}).get("position") else None,
        ))

    return PublicWorkflowPreview(
        nodes=nodes,
        edges=edges,
        triggers=triggers,
        metadata=PreviewMetadata(name=name, description=description, icon=icon),
    )


@router.get("/templates/{slug}/preview", response_model=PublicWorkflowPreview)
async def get_template_preview(slug: str) -> PublicWorkflowPreview:
    t = await WorkflowTemplate.filter(slug=slug, source=TemplateSource.COMMUNITY).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    return _sanitize_spec(t.spec or {}, t.name, t.description, t.icon)
