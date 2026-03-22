"""FastAPI router for workflow templates."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from seer.api.templates import models as api_models
from seer.api.templates import services
from seer.api.templates import admin_services
from seer.api.core.middleware.organization import get_organization
from seer.database import Organization, User

router = APIRouter(prefix="/v1", tags=["templates"])


def _require_user(request: Request) -> User:
    """Require authenticated user."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


# ===== Public Endpoints =====


@router.get("/templates/categories", response_model=api_models.TemplateCategoriesResponse)
async def get_template_categories(request: Request):
    """List available template categories with counts."""
    _require_user(request)
    return await services.get_template_categories()


@router.get("/templates", response_model=api_models.TemplateListResponse)
async def list_templates(  # pylint: disable=too-many-arguments  # FastAPI query params
    request: Request,
    *,  # Force keyword-only arguments
    category: Optional[str] = Query(None, description="Filter by category"),
    scope: Optional[str] = Query(None, description="Filter scope: 'mine' or 'community'"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    featured: bool = Query(False, description="Only show featured templates"),
    limit: int = Query(50, ge=1, le=100, description="Max items to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
):
    """List published workflow templates."""
    user = _require_user(request)
    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    return await services.list_templates(
        user=user,
        scope=scope,
        category=category,
        tags=tag_list,
        search=search,
        featured_only=featured,
        limit=limit,
        cursor=cursor,
    )


@router.get("/templates/{slug}", response_model=api_models.TemplateDetailResponse)
async def get_template(request: Request, slug: str):
    """Get template details by slug."""
    _require_user(request)
    return await services.get_template(slug)


@router.get("/templates/{slug}/requirements", response_model=api_models.TemplateRequirementsResponse)
async def check_template_requirements(request: Request, slug: str):
    """Check if user has required integrations for a template."""
    user = _require_user(request)
    return await services.check_requirements(user, slug)


@router.post(
    "/templates/{slug}/instantiate",
    response_model=api_models.TemplateInstantiateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def instantiate_template(
    request: Request,
    slug: str,
    payload: api_models.TemplateInstantiateRequest,
):
    """Create a workflow from a template."""
    user = _require_user(request)
    org: Optional[Organization] = None
    try:
        org = get_organization(request)
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: fallback to None if org context not available
        pass
    return await services.instantiate_template(user, slug, payload, organization=org)


# ===== Admin Endpoints =====
# Note: In production, these should have additional admin authorization checks


@router.get("/admin/templates", response_model=api_models.TemplateAdminListResponse)
async def admin_list_templates(
    request: Request,
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=100, description="Max items to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
):
    """List all templates (admin view)."""
    _require_user(request)
    return await admin_services.list_all_templates(
        category=category,
        limit=limit,
        cursor=cursor,
    )


@router.get("/admin/templates/{slug}", response_model=api_models.TemplateAdminResponse)
async def admin_get_template(request: Request, slug: str):
    """Get template details (admin view, includes unpublished)."""
    _require_user(request)
    return await admin_services.get_template_admin(slug)


@router.put("/admin/templates/{slug}", response_model=api_models.TemplateAdminResponse)
async def admin_update_template(
    request: Request,
    slug: str,
    payload: api_models.TemplateUpdateRequest,
):
    """Update an existing template."""
    _require_user(request)
    return await admin_services.update_template(slug, payload)


@router.delete("/admin/templates/{slug}", status_code=status.HTTP_200_OK)
async def admin_delete_template(request: Request, slug: str):
    """Delete a template."""
    _require_user(request)
    await admin_services.delete_template(slug)
    return {"ok": True}


__all__ = ["router"]
