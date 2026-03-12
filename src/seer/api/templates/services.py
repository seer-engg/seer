"""Public services for workflow templates."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

import pytz
from fastapi import HTTPException, status

from seer.api.templates import models as api_models
from seer.api.workflows import models as workflow_models
from seer.api.workflows.services.lifecycle import create_workflow as create_workflow_from_spec
from seer.database import (
    OAuthConnection,
    Organization,
    User,
    WorkflowTemplate,
    TemplateCategory,
    make_template_public_id,
)


# Config placeholder pattern: ${config.field_name}
CONFIG_PLACEHOLDER_PATTERN = re.compile(r"\$\{config\.([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _raise_not_found(slug: str) -> None:
    """Raise a 404 error for template not found."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Template '{slug}' not found",
    )


def _build_scope_query(scope: Optional[str], user: Optional[User]):
    """Build base QuerySet filtered by scope and user."""
    if scope == "mine" and user:
        return WorkflowTemplate.filter(created_by=user)
    if scope == "community" and user:
        return WorkflowTemplate.filter(is_published=True, visibility="public").exclude(created_by=user)
    return WorkflowTemplate.filter(is_published=True)


async def list_templates(  # pylint: disable=too-many-arguments  # legitimate filter params
    *,
    user: Optional[User] = None,
    scope: Optional[str] = None,  # "mine" | "community" | None (all)
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    search: Optional[str] = None,
    featured_only: bool = False,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.TemplateListResponse:
    """List templates with visibility-aware filtering."""
    limit = max(1, min(limit, 100))

    query = _build_scope_query(scope, user)

    # Apply filters
    if category:
        query = query.filter(category=category)

    if featured_only:
        query = query.filter(is_featured=True)

    if search:
        query = query.filter(name__icontains=search) | query.filter(description__icontains=search)

    # Cursor-based pagination
    if cursor:
        try:
            cursor_id = int(cursor)
            query = query.filter(id__lt=cursor_id)
        except ValueError:
            pass

    # Order by featured first, then by usage count, then by id
    templates = await query.order_by("-is_featured", "-usage_count", "-id").limit(limit + 1)

    # Filter by tags (post-query since JSONField)
    if tags:
        templates = [t for t in templates if any(tag in (t.tags or []) for tag in tags)]

    # Build response
    items = [_to_summary(t) for t in templates[:limit]]
    next_cursor = str(templates[-1].id) if len(templates) > limit else None

    # Get total count for current scope
    count_query = _build_scope_query(scope, user)
    if category:
        count_query = count_query.filter(category=category)
    if featured_only:
        count_query = count_query.filter(is_featured=True)
    total = await count_query.count()

    return api_models.TemplateListResponse(
        items=items,
        total=total,
        next_cursor=next_cursor,
    )


async def get_template(slug: str) -> api_models.TemplateDetailResponse:
    """Get template details by slug."""
    template = await WorkflowTemplate.filter(slug=slug, is_published=True).first()
    if not template:
        _raise_not_found(slug)

    return _to_detail(template)


async def get_template_categories() -> api_models.TemplateCategoriesResponse:
    """List available template categories with counts."""
    categories = []

    for cat in TemplateCategory:
        count = await WorkflowTemplate.filter(category=cat, is_published=True).count()
        categories.append(
            api_models.TemplateCategoryInfo(
                key=cat.value,
                label=api_models.CATEGORY_LABELS.get(cat, cat.value.replace("_", " ").title()),
                template_count=count,
            )
        )

    return api_models.TemplateCategoriesResponse(categories=categories)


async def check_requirements(
    user: User,
    slug: str,
) -> api_models.TemplateRequirementsResponse:
    """Check if user has required integrations for a template."""
    template = await WorkflowTemplate.filter(slug=slug, is_published=True).first()
    if not template:
        _raise_not_found(slug)

    # Get user's OAuth connections
    connections = await OAuthConnection.filter(user=user, status="active").all()
    connection_map: Dict[str, OAuthConnection] = {}
    for conn in connections:
        # Map by provider (e.g., "google" -> connection)
        if conn.provider not in connection_map:
            connection_map[conn.provider] = conn

    # Check each required integration
    integration_statuses = []
    all_connected = True

    for req in template.required_integrations or []:
        provider = req.get("provider", "")
        integration_type = req.get("integration_type", "")
        reason = req.get("reason", "")

        conn = connection_map.get(provider)
        connected = conn is not None

        if not connected:
            all_connected = False

        integration_statuses.append(
            api_models.IntegrationStatus(
                provider=provider,
                integration_type=integration_type,
                reason=reason,
                connected=connected,
                connection_id=conn.id if conn else None,
                connection_name=conn.provider_account_id if conn else None,
            )
        )

    return api_models.TemplateRequirementsResponse(
        template_id=make_template_public_id(template.id),
        all_connected=all_connected,
        integrations=integration_statuses,
    )


async def instantiate_template(
    user: User,
    slug: str,
    payload: api_models.TemplateInstantiateRequest,
    organization: Optional[Organization] = None,
) -> api_models.TemplateInstantiateResponse:
    """Create a workflow from a template."""
    template = await WorkflowTemplate.filter(slug=slug, is_published=True).first()
    if not template:
        _raise_not_found(slug)

    # Deep copy the spec
    spec = copy.deepcopy(template.spec)

    # Resolve config placeholders
    spec = _resolve_placeholders(spec, payload.config)

    # Validate resolved timezones in trigger configs
    _validate_resolved_timezones(spec)

    # Map provider connections to triggers
    if payload.provider_connections:
        spec = _apply_provider_connections(spec, payload.provider_connections)

    # Check for missing integrations
    missing_integrations = await _check_missing_integrations(user, template.required_integrations or [])

    # Create the workflow using existing service
    workflow_request = workflow_models.WorkflowCreateRequest(
        name=payload.name,
        spec=spec,
    )

    workflow_response = await create_workflow_from_spec(user, workflow_request, organization=organization)

    # Increment usage count
    await WorkflowTemplate.filter(id=template.id).update(usage_count=template.usage_count + 1)

    return api_models.TemplateInstantiateResponse(
        workflow_id=workflow_response.workflow_id,
        name=workflow_response.name,
        missing_integrations=[
            api_models.RequiredIntegration(
                provider=m.get("provider", ""),
                integration_type=m.get("integration_type", ""),
                reason=m.get("reason", ""),
            )
            for m in missing_integrations
        ],
    )


def extract_config_fields(spec: Dict[str, Any]) -> List[api_models.TemplateConfigField]:
    """Extract ${config.xxx} placeholders from a spec."""
    found_fields: Dict[str, api_models.TemplateConfigField] = {}

    def scan_value(value: Any) -> None:
        if isinstance(value, str):
            for match in CONFIG_PLACEHOLDER_PATTERN.finditer(value):
                field_name = match.group(1)
                if field_name not in found_fields:
                    # Generate a human-readable label
                    label = field_name.replace("_", " ").title()
                    found_fields[field_name] = api_models.TemplateConfigField(
                        name=field_name,
                        label=label,
                        type="string",
                        required=True,
                    )
        elif isinstance(value, dict):
            for v in value.values():
                scan_value(v)
        elif isinstance(value, list):
            for item in value:
                scan_value(item)

    scan_value(spec)
    return list(found_fields.values())


# ===== Internal Helper Functions =====


def _build_template_fields(template: WorkflowTemplate) -> Dict[str, Any]:
    """Build common template field mappings for API responses."""
    config_fields = extract_config_fields(template.spec or {})
    return {
        "template_id": make_template_public_id(template.id),
        "slug": template.slug,
        "name": template.name,
        "description": template.description,
        "category": template.category.value if isinstance(template.category, TemplateCategory) else template.category,
        "tags": template.tags or [],
        "icon": template.icon,
        "is_featured": template.is_featured,
        "usage_count": template.usage_count,
        "visibility": template.visibility,
        "required_integrations": [
            api_models.RequiredIntegration(
                provider=r.get("provider", ""),
                integration_type=r.get("integration_type", ""),
                reason=r.get("reason", ""),
            )
            for r in (template.required_integrations or [])
        ],
        "spec": template.spec or {},
        "config_fields": config_fields,
        "preview_image_url": template.preview_image_url,
        "source": template.source.value if hasattr(template.source, "value") else template.source,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
    }


def _to_summary(template: WorkflowTemplate) -> api_models.TemplateSummary:
    """Convert a template to a summary response."""
    return api_models.TemplateSummary(
        template_id=make_template_public_id(template.id),
        slug=template.slug,
        name=template.name,
        description=template.description,
        category=template.category.value if isinstance(template.category, TemplateCategory) else template.category,
        tags=template.tags or [],
        icon=template.icon,
        is_featured=template.is_featured,
        usage_count=template.usage_count,
        visibility=template.visibility,
        required_integrations=[
            api_models.RequiredIntegration(
                provider=r.get("provider", ""),
                integration_type=r.get("integration_type", ""),
                reason=r.get("reason", ""),
            )
            for r in (template.required_integrations or [])
        ],
    )


def _to_detail(template: WorkflowTemplate) -> api_models.TemplateDetailResponse:
    """Convert a template to a detail response."""
    return api_models.TemplateDetailResponse(**_build_template_fields(template))


def _validate_resolved_timezones(spec: Dict[str, Any]) -> None:
    """Validate that timezone values in trigger provider_configs are valid IANA names."""
    for trigger in spec.get("triggers", []):
        provider_config = trigger.get("provider_config", {})
        if not provider_config or "timezone" not in provider_config:
            continue
        tz_val = provider_config["timezone"]
        if not isinstance(tz_val, str):
            continue
        tz_val = tz_val.strip()
        try:
            pytz.timezone(tz_val)
        except pytz.exceptions.UnknownTimeZoneError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timezone '{provider_config['timezone']}'. Use IANA timezone names (e.g., America/Chicago, UTC)",
            ) from exc


def _resolve_placeholders(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve ${config.xxx} placeholders in a spec with provided config values.

    Returns a new spec with placeholders replaced.
    """

    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            # Replace all ${config.xxx} patterns
            def replacer(match: re.Match) -> str:
                field_name = match.group(1)
                if field_name in config:
                    return str(config[field_name])
                # Keep the placeholder if no value provided
                return match.group(0)

            return CONFIG_PLACEHOLDER_PATTERN.sub(replacer, value)
        if isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value

    return resolve_value(spec)


def _apply_provider_connections(
    spec: Dict[str, Any],
    provider_connections: Dict[str, int],
) -> Dict[str, Any]:
    """
    Apply provider connection IDs to triggers in the spec.

    Maps integration_type -> connection_id to trigger provider_config.
    """
    triggers = spec.get("triggers", [])
    if not triggers:
        return spec

    for trigger in triggers:
        # Check if this trigger uses an integration
        trigger_key = trigger.get("trigger", "")
        if "." in trigger_key:
            provider = trigger_key.split(".")[0]
            if provider in provider_connections:
                if "provider_config" not in trigger:
                    trigger["provider_config"] = {}
                trigger["provider_config"]["connection_id"] = provider_connections[provider]

    return spec


async def _check_missing_integrations(
    user: User,
    required_integrations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Check which required integrations the user is missing."""
    if not required_integrations:
        return []

    connections = await OAuthConnection.filter(user=user, status="active").all()
    connected_providers = {conn.provider for conn in connections}

    missing = []
    for req in required_integrations:
        provider = req.get("provider", "")
        if provider and provider not in connected_providers:
            missing.append(req)

    return missing


__all__ = [
    "list_templates",
    "get_template",
    "get_template_categories",
    "check_requirements",
    "instantiate_template",
    "extract_config_fields",
    "_build_template_fields",
]
