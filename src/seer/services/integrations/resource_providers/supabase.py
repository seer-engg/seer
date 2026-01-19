from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.services.integrations.constants import (
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from seer.services.integrations.providers import get_integration_provider
from seer.services.integrations.resource_providers.base import ResourceProvider
from seer.logger import get_logger

logger = get_logger("api.integrations.resource_providers.supabase")


def _transform_supabase_project(project: Dict) -> Dict[str, Any]:
    """
    Normalize Supabase project API response to resource schema.

    Handles field variations (ref/project_ref, id/project_id) and builds
    fallback chains for display_name.

    Args:
        project: Raw project dict from Supabase API

    Returns:
        Normalized resource dict matching schema
    """
    ref = project.get("ref") or project.get("project_ref")
    project_id = project.get("id") or project.get("project_id")
    return {
        "id": ref or str(project_id),
        "name": project.get("name") or ref or project_id,
        "display_name": project.get("name") or ref or project_id,
        "type": "project",
        "project_id": project_id,
        "project_ref": ref,
        "organization_id": project.get("organization_id") or project.get("org_id"),
        "region": project.get("region"),
        "status": project.get("status"),
        "rest_url": project.get("api_url") or project.get("restUrl"),
    }


def _filter_projects_by_query(projects: list[Dict], query: Optional[str]) -> list[Dict]:
    """
    Filter projects by case-insensitive substring match on name/ref.

    Args:
        projects: List of project dicts from Supabase API
        query: Search string (optional)

    Returns:
        Filtered list (or original if no query)
    """
    if not query:
        return projects

    query_lower = query.lower()
    return [
        project
        for project in projects
        if query_lower in (project.get("name") or "").lower()
        or query_lower in (project.get("ref") or "").lower()
    ]


def _paginate_results(
    items: list[Dict],
    page_token: Optional[str],
    page_size: int
) -> tuple[list[Dict], Optional[str], int]:
    """
    Apply offset-based pagination with defensive token parsing.

    Args:
        items: Full list to paginate
        page_token: String-encoded integer offset (optional)
        page_size: Items per page

    Returns:
        Tuple of (sliced_items, next_page_token, total_count)
    """
    total = len(items)
    start_index = 0

    if page_token:
        try:
            start_index = int(page_token)
        except ValueError:
            start_index = 0

    end_index = start_index + page_size
    sliced = items[start_index:end_index]
    next_page = str(end_index) if end_index < total else None

    return sliced, next_page, total


async def _fetch_supabase_projects_safe(
    access_token: str,
) -> tuple[Optional[list[Dict]], Optional[Dict[str, Any]]]:
    """
    Fetch Supabase projects with error handling.

    Returns:
        Tuple of (projects_list, error_response)
        If error_response is not None, caller should return it immediately
    """
    provider = get_integration_provider(SUPABASE_RESOURCE_PROVIDER)
    if not provider:
        return None, {
            "items": [],
            "error": "Supabase provider unavailable",
            "next_page_token": None
        }

    try:
        projects = await provider.list_remote_resources(
            access_token=access_token,
            resource_type=SUPABASE_RESOURCE_TYPE_PROJECT,
        )
        return projects, None
    except HTTPException as exc:
        return None, {"items": [], "error": exc.detail, "next_page_token": None}
    except Exception as exc:
        logger.exception("Error listing Supabase projects: %s", exc)
        return None, {"items": [], "error": str(exc), "next_page_token": None}


class SupabaseResourceProvider(ResourceProvider):
    provider = SUPABASE_RESOURCE_PROVIDER
    aliases = ["supabase", "supabase_mgmt"]
    resource_configs: Dict[str, Dict[str, Any]] = {
        "supabase_project": {
            "display_field": "name",
            "value_field": "ref",
            "supports_hierarchy": False,
            "supports_search": True,
        },
    }

    async def list_resources(
        self,
        *,
        access_token: str,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """List Supabase projects with search and pagination."""
        if resource_type != "supabase_project":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported Supabase resource type '{resource_type}'"
            )

        # Fetch projects with error handling
        projects, error = await _fetch_supabase_projects_safe(access_token)
        if error:
            return error

        # Filter by query
        filtered = _filter_projects_by_query(projects, query)

        # Paginate
        sliced, next_token, total = _paginate_results(filtered, page_token, page_size)

        # Transform items
        items = [_transform_supabase_project(p) for p in sliced]

        return {
            "items": items,
            "next_page_token": next_token,
            "supports_hierarchy": False,
            "supports_search": True,
            "total_count": total,
        }
