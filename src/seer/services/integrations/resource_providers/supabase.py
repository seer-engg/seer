# pylint: disable=broad-exception-caught,too-many-arguments
# Reason: Resource provider needs broad exception handling for API errors; list_resources has many filter params
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.database import IntegrationResource, IntegrationSecret
from seer.services.integrations.constants import (
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from seer.services.integrations.providers import get_integration_provider
from seer.services.integrations.resource_providers.base import ResourceContext, ResourceProvider
from seer.services.integrations.resource_providers.utils import filter_entries, paginate_items, parse_offset, resolve_resource_id
from seer.tools.supabase.common import _resolve_rest_url, _service_headers
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


async def _call_supabase_rpc(
    *,
    rest_url: str,
    service_role_key: str,
    function: str,
    payload: dict,
) -> list[dict]:
    """
    Call Supabase RPC function.

    Args:
        rest_url: Supabase REST API URL
        service_role_key: Service role key for authentication
        function: RPC function name
        payload: Function parameters

    Returns:
        List of result dictionaries

    Raises:
        HTTPException: If RPC call fails
    """
    url = f"{rest_url.rstrip('/')}/rpc/{function}"
    headers = _service_headers(service_role_key, {"Content-Type": "application/json"})
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return []
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Supabase RPC call failed",
            extra={
                "url": url,
                "status": exc.response.status_code,
                "body": exc.response.text[:200],
                "payload": payload,
            },
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=(
                f"Supabase RPC '{function}' failed or is missing. "
                "Please create the function in your project and grant execute to service_role."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Supabase RPC call failed", extra={"url": url})
        raise HTTPException(status_code=500, detail="Failed to fetch Supabase metadata") from exc


async def _get_supabase_rest_context(user: Any, integration_resource_id: int) -> tuple[IntegrationResource, str, str]:
    """
    Get Supabase REST API context for a resource.

    Args:
        user: User object
        integration_resource_id: Supabase project resource ID

    Returns:
        Tuple of (resource, service_role_key, rest_url)

    Raises:
        HTTPException: If resource not found or missing credentials
    """
    resource = await IntegrationResource.get_or_none(
        id=integration_resource_id,
        user=user,
        provider=SUPABASE_RESOURCE_PROVIDER,
        status="active",
    )
    if not resource:
        raise HTTPException(status_code=404, detail=f"Supabase resource {integration_resource_id} not found")

    service_key = await IntegrationSecret.get_or_none(
        user=user,
        provider=SUPABASE_RESOURCE_PROVIDER,
        resource=resource,
        name="supabase_service_role_key",
        status="active",
    )
    if not service_key:
        raise HTTPException(
            status_code=400,
            detail="Supabase project is missing service role key. Please re-bind the project.",
        )

    rest_url = _resolve_rest_url(resource)
    if not rest_url:
        raise HTTPException(
            status_code=400,
            detail="Supabase project metadata is missing rest_url. Please re-bind the project.",
        )

    return resource, service_key.value_enc, rest_url


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
            "source": "api",
        },
        "schema": {
            "display_field": "name",
            "value_field": "id",
            "supports_hierarchy": False,
            "supports_search": True,
            "depends_on": "integration_resource_id",
            "source": "rpc",
        },
        "table": {
            "display_field": "name",
            "value_field": "id",
            "supports_hierarchy": False,
            "supports_search": True,
            "depends_on": ["integration_resource_id", "schema"],
            "source": "rpc",
        },
    }

    async def list_resources(  # pylint: disable=too-many-locals  # Multiple resource types require different context variables
        self,
        *,
        access_token: Optional[str] = None,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
        context: Optional[ResourceContext] = None,
    ) -> Dict[str, Any]:
        """
        List Supabase resources (projects, schemas, or tables).

        Args:
            access_token: OAuth access token (for projects)
            resource_type: "supabase_project", "schema", or "table"
            query: Optional search query
            parent_id: Not used
            page_token: Pagination token
            page_size: Number of results per page
            filter_params: Not used
            depends_on_values: For schema/table, must contain integration_resource_id
            context: ResourceContext with user and auth information

        Returns:
            Standard resource response with items, next_page_token, metadata

        Raises:
            HTTPException: If resource type unsupported or required params missing
        """
        if resource_type == "supabase_project":
            # Use access_token from either parameter or context
            token = access_token or (context.access_token if context else None)
            if not token:
                raise HTTPException(status_code=400, detail="Access token required for listing Supabase projects")

            # Fetch projects with error handling
            projects, error = await _fetch_supabase_projects_safe(token)
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

        if resource_type == "schema":
            if not context or not context.user:
                raise HTTPException(status_code=400, detail="User context required for listing schemas")

            # Resolve integration_resource_id
            resource_id = resolve_resource_id(None, depends_on_values or {})

            # Get Supabase REST context
            _, service_role_key, rest_url = await _get_supabase_rest_context(context.user, resource_id)

            # Call RPC to list schemas
            offset = parse_offset(page_token)
            raw_schemas = await _call_supabase_rpc(
                rest_url=rest_url,
                service_role_key=service_role_key,
                function="list_schemas",
                payload={},
            )

            # Filter and paginate
            names = filter_entries(raw_schemas, name_keys=("schema_name", "name"), query=query, skip_system=True)
            return paginate_items(names, page_size=page_size, offset=offset, item_type="schema")

        if resource_type == "table":
            if not context or not context.user:
                raise HTTPException(status_code=400, detail="User context required for listing tables")

            # Resolve integration_resource_id and schema
            resource_id = resolve_resource_id(None, depends_on_values or {})
            schema_name = (depends_on_values or {}).get("schema", "public")
            if isinstance(schema_name, str) and schema_name.strip():
                schema_name = schema_name.strip()
            else:
                schema_name = "public"

            # Get Supabase REST context
            _, service_role_key, rest_url = await _get_supabase_rest_context(context.user, resource_id)

            # Call RPC to list tables
            offset = parse_offset(page_token)
            raw_tables = await _call_supabase_rpc(
                rest_url=rest_url,
                service_role_key=service_role_key,
                function="list_tables",
                payload={"_schema": schema_name},
            )

            # Filter and paginate
            names = filter_entries(raw_tables, name_keys=("table_name", "name"), query=query, skip_system=False)
            return paginate_items(names, page_size=page_size, offset=offset, item_type="table", description=schema_name)

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Supabase resource type '{resource_type}'"
        )
