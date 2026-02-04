"""Admin services for workflow template management."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from tortoise.exceptions import IntegrityError

from seer.api.templates import models as api_models
from seer.api.templates.services import extract_config_fields
from seer.database import (
    User,
    Workflow,
    WorkflowVersion,
    WorkflowVersionStatus,
    WorkflowTemplate,
    TemplateCategory,
    TemplateSource,
    make_template_public_id,
    parse_workflow_public_id,
)


def _raise_not_found(slug: str) -> None:
    """Raise a 404 error for template not found."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Template '{slug}' not found",
    )


def _raise_conflict(message: str) -> None:
    """Raise a 409 conflict error."""
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=message,
    )


def _raise_bad_request(message: str) -> None:
    """Raise a 400 bad request error."""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def _validate_category(category: str) -> TemplateCategory:
    """Validate and convert category string to enum."""
    try:
        return TemplateCategory(category)
    except ValueError:
        valid = [c.value for c in TemplateCategory]
        _raise_bad_request(f"Invalid category '{category}'. Valid values: {valid}")
        raise  # Unreachable but satisfies type checker


# Provider mapping (integration_type -> provider)
INTEGRATION_PROVIDER_MAP = {
    "gmail": "google",
    "google_calendar": "google",
    "google_drive": "google",
    "google_sheets": "google",
    "slack": "slack",
    "github": "github",
    "notion": "notion",
    "linear": "linear",
    "hubspot": "hubspot",
    "salesforce": "salesforce",
}


def _map_integration_to_provider(integration_type: str) -> str:
    """Map integration type to provider name."""
    return INTEGRATION_PROVIDER_MAP.get(integration_type, integration_type)


async def _get_workflow_for_template(workflow_id: str) -> Workflow:
    """Get workflow by public ID (admin can access any user's workflow)."""
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow_id format",
        ) from exc

    workflow = await Workflow.filter(id=pk).first()
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )

    return workflow


async def _get_workflow_spec(workflow: Workflow) -> Dict[str, Any]:
    """Get spec from RELEASED version if exists, otherwise DRAFT."""
    # Try RELEASED first
    version = await WorkflowVersion.filter(
        workflow=workflow,
        status=WorkflowVersionStatus.RELEASED,
    ).first()

    # Fall back to DRAFT
    if not version:
        version = await WorkflowVersion.filter(
            workflow=workflow,
            status=WorkflowVersionStatus.DRAFT,
        ).first()

    if not version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow has no spec (no draft or released version)",
        )

    return version.spec


def _detect_required_integrations(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """Auto-detect required integrations from workflow triggers and tool nodes."""
    integrations = []
    seen_providers = set()

    # Check triggers
    for trigger in spec.get("triggers", []):
        trigger_key = trigger.get("key", "")  # e.g., "gmail.new_email"
        if "." in trigger_key:
            integration_type = trigger_key.split(".")[0]  # "gmail"
            provider = _map_integration_to_provider(integration_type)  # "google"

            if provider not in seen_providers:
                seen_providers.add(provider)
                integrations.append({
                    "provider": provider,
                    "integration_type": integration_type,
                    "reason": f"Required for {trigger_key} trigger",
                })

    # Check tool nodes for provider-specific tools
    for node in spec.get("nodes", []):
        if node.get("type") == "tool":
            tool_key = node.get("tool", "")  # e.g., "gmail.send_email"
            if "." in tool_key:
                integration_type = tool_key.split(".")[0]
                provider = _map_integration_to_provider(integration_type)

                if provider not in seen_providers:
                    seen_providers.add(provider)
                    integrations.append({
                        "provider": provider,
                        "integration_type": integration_type,
                        "reason": f"Required for {tool_key} tool",
                    })

    return integrations


async def list_all_templates(
    *,
    category: Optional[str] = None,
    include_unpublished: bool = True,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.TemplateAdminListResponse:
    """List all templates (including unpublished) for admin."""
    limit = max(1, min(limit, 100))

    query = WorkflowTemplate.all()

    if category:
        query = query.filter(category=category)

    if not include_unpublished:
        query = query.filter(is_published=True)

    if cursor:
        try:
            cursor_id = int(cursor)
            query = query.filter(id__lt=cursor_id)
        except ValueError:
            pass

    templates = await query.order_by("-id").limit(limit + 1)

    items = [_to_admin_response(t) for t in templates[:limit]]
    next_cursor = str(templates[-1].id) if len(templates) > limit else None

    total = await WorkflowTemplate.all().count()

    return api_models.TemplateAdminListResponse(
        items=items,
        total=total,
        next_cursor=next_cursor,
    )


async def get_template_admin(slug: str) -> api_models.TemplateAdminResponse:
    """Get template details by slug (admin, includes unpublished)."""
    template = await WorkflowTemplate.filter(slug=slug).first()
    if not template:
        _raise_not_found(slug)

    return _to_admin_response(template)


async def create_template(
    user: User,
    payload: api_models.TemplateCreateRequest,
) -> api_models.TemplateAdminResponse:
    """Create a new template from an existing workflow."""
    category = _validate_category(payload.category)

    # 1. Fetch workflow (any user's workflow - admin privilege)
    workflow = await _get_workflow_for_template(payload.workflow_id)

    # 2. Get spec from RELEASED version (if exists) or DRAFT
    spec = await _get_workflow_spec(workflow)

    # 3. Auto-detect required integrations from triggers (if not provided)
    if payload.required_integrations is None:
        required_integrations = _detect_required_integrations(spec)
    else:
        required_integrations = [
            {
                "provider": ri.provider,
                "integration_type": ri.integration_type,
                "reason": ri.reason,
            }
            for ri in payload.required_integrations
        ]

    # 4. Create template with extracted spec
    try:
        template = await WorkflowTemplate.create(
            slug=payload.slug,
            name=payload.name,
            description=payload.description,
            category=category,
            tags=payload.tags,
            source=TemplateSource.SYSTEM,
            created_by=user,
            spec=spec,
            required_integrations=required_integrations,
            source_workflow_id=workflow.id,  # Track lineage
            icon=payload.icon,
            preview_image_url=payload.preview_image_url,
            is_published=payload.is_published,
            is_featured=payload.is_featured,
        )
    except IntegrityError as e:
        if "slug" in str(e).lower() or "unique" in str(e).lower():
            _raise_conflict(f"Template with slug '{payload.slug}' already exists")
        raise

    return _to_admin_response(template)


def _build_template_update_data(
    payload: api_models.TemplateUpdateRequest,
) -> Dict[str, Any]:
    """Build update dictionary from payload."""
    update_data = {}

    # Simple field mappings (field_name, value, transformer)
    field_mappings = [
        ("slug", payload.slug, None),
        ("name", payload.name, None),
        ("description", payload.description, None),
        ("category", payload.category, _validate_category),
        ("tags", payload.tags, None),
        ("spec", payload.spec, None),
        ("icon", payload.icon, None),
        ("preview_image_url", payload.preview_image_url, None),
        ("is_featured", payload.is_featured, None),
    ]

    for field_name, value, transformer in field_mappings:
        if value is not None:
            update_data[field_name] = transformer(value) if transformer else value

    # Handle required_integrations separately (needs list comprehension)
    if payload.required_integrations is not None:
        update_data["required_integrations"] = [
            {
                "provider": ri.provider,
                "integration_type": ri.integration_type,
                "reason": ri.reason,
            }
            for ri in payload.required_integrations
        ]

    return update_data


async def update_template(
    slug: str,
    payload: api_models.TemplateUpdateRequest,
) -> api_models.TemplateAdminResponse:
    """Update an existing template."""
    template = await WorkflowTemplate.filter(slug=slug).first()
    if not template:
        _raise_not_found(slug)

    update_data = _build_template_update_data(payload)

    if update_data:
        try:
            await WorkflowTemplate.filter(id=template.id).update(**update_data)
        except IntegrityError as e:
            if "slug" in str(e).lower() or "unique" in str(e).lower():
                _raise_conflict(f"Template with slug '{payload.slug}' already exists")
            raise

    # Refetch to get updated values
    template = await WorkflowTemplate.get(id=template.id)
    return _to_admin_response(template)


async def delete_template(slug: str) -> None:
    """Delete a template."""
    template = await WorkflowTemplate.filter(slug=slug).first()
    if not template:
        _raise_not_found(slug)

    await template.delete()


async def toggle_publish(
    slug: str,
    payload: api_models.TemplatePublishRequest,
) -> api_models.TemplateAdminResponse:
    """Publish or unpublish a template."""
    template = await WorkflowTemplate.filter(slug=slug).first()
    if not template:
        _raise_not_found(slug)

    await WorkflowTemplate.filter(id=template.id).update(is_published=payload.is_published)

    # Refetch to get updated values
    template = await WorkflowTemplate.get(id=template.id)
    return _to_admin_response(template)


def _to_admin_response(template: WorkflowTemplate) -> api_models.TemplateAdminResponse:
    """Convert a template to an admin response."""
    config_fields = extract_config_fields(template.spec or {})

    return api_models.TemplateAdminResponse(
        template_id=make_template_public_id(template.id),
        slug=template.slug,
        name=template.name,
        description=template.description,
        category=template.category.value if isinstance(template.category, TemplateCategory) else template.category,
        tags=template.tags or [],
        icon=template.icon,
        is_featured=template.is_featured,
        is_published=template.is_published,
        usage_count=template.usage_count,
        required_integrations=[
            api_models.RequiredIntegration(
                provider=r.get("provider", ""),
                integration_type=r.get("integration_type", ""),
                reason=r.get("reason", ""),
            )
            for r in (template.required_integrations or [])
        ],
        spec=template.spec or {},
        config_fields=config_fields,
        preview_image_url=template.preview_image_url,
        source=template.source.value if hasattr(template.source, "value") else template.source,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


__all__ = [
    "list_all_templates",
    "get_template_admin",
    "create_template",
    "update_template",
    "delete_template",
    "toggle_publish",
    # Helper functions for testing
    "_get_workflow_for_template",
    "_get_workflow_spec",
    "_detect_required_integrations",
    "_map_integration_to_provider",
]
