"""Integration resource routes."""
# pylint: disable=duplicate-code,too-many-lines
# Reason: Supabase REST validation mirrors supabase database tool usage; this module aggregates many integration endpoints and helpers.
import json
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.integrations.models import SupabaseBindRequest, SupabaseManualBindRequest
from seer.api.integrations.services import (
    bind_supabase_project,
    bind_supabase_project_manual,
    deactivate_integration_resource,
    get_valid_access_token,
    list_resource_secrets,
    serialize_integration_resource,
    serialize_integration_secret,
)
from seer.config import config
from seer.database import IntegrationResource, IntegrationSecret, User
from seer.logger import get_logger
from seer.services.integrations.constants import SUPABASE_RESOURCE_PROVIDER
from seer.services.integrations.resource_browser import ResourceBrowser, ResourceListOptions
from seer.services.integrations.resource_providers.base import ResourceContext
from seer.services.integrations.resource_providers.utils import filter_entries, paginate_items, parse_depends_on, parse_offset, resolve_resource_id
from seer.tools.oauth_manager import get_oauth_token
from seer.tools.supabase.common import _resolve_rest_url, _service_headers

logger = get_logger(__name__)


router = APIRouter(tags=["integrations.resources"])

@router.get("/resources/{resource_id}/secrets")
async def list_resource_secret_bindings(request: Request, resource_id: int):
    user: User = request.state.db_user
    secrets = await list_resource_secrets(user, resource_id)
    return {"items": [serialize_integration_secret(s) for s in secrets]}


@router.get("/resources/{resource_id}/status")
async def get_resource_status(request: Request, resource_id: int):
    """Check if a resource exists and its status."""
    user: User = request.state.db_user
    resource = await IntegrationResource.get_or_none(id=resource_id, user=user)
    if not resource:
        return {"exists": False, "status": None}
    return {"exists": True, "status": resource.status}


@router.delete("/resources/{resource_id}")
async def delete_resource_binding(request: Request, resource_id: int):
    user: User = request.state.db_user
    resource = await deactivate_integration_resource(user, resource_id)
    return {"resource": serialize_integration_resource(resource)}


@router.post("/supabase/projects/bind")
async def bind_supabase_project_route(request: Request, payload: SupabaseBindRequest):
    """
    Persist a Supabase project resource and sync its API keys into the vault.
    """

    user: User = request.state.db_user
    resource = await bind_supabase_project(user, payload.project_ref, payload.connection_id)
    secrets = await list_resource_secrets(user, resource.id)
    return {
        "resource": serialize_integration_resource(resource),
        "secrets": [serialize_integration_secret(s) for s in secrets],
    }


@router.post("/supabase/projects/manual-bind")
async def bind_supabase_project_manual_route(request: Request, payload: SupabaseManualBindRequest):
    """
    Persist a Supabase project using user-supplied secrets instead of OAuth.
    Falls back to the OAuth binding flow when connection_id is provided.
    """

    user: User = request.state.db_user

    if payload.connection_id:
        resource = await bind_supabase_project(user, payload.project_ref, payload.connection_id)
    else:
        if payload.service_role_key is None:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Missing service_role_key",
                detail="service_role_key is required when connection_id is not provided",
                status=400
            )
        assert payload.service_role_key is not None
        service_role_key = payload.service_role_key
        resource = await bind_supabase_project_manual(
            user,
            project_ref=payload.project_ref,
            service_role_key=service_role_key,
            project_name=payload.project_name,
            anon_key=payload.anon_key,
        )

    secrets = await list_resource_secrets(user, resource.id)
    return {
        "resource": serialize_integration_resource(resource),
        "secrets": [serialize_integration_secret(s) for s in secrets],
    }


async def _get_supabase_rest_context(user: User, integration_resource_id: int) -> tuple[IntegrationResource, str, str]:
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


async def _execute_supabase_sql(
    *,
    access_token: str,
    project_ref: str,
    sql: str,
) -> None:
    api_base = config.supabase_management_api_base or "https://api.supabase.com"
    url = f"{api_base.rstrip('/')}/v1/projects/{project_ref}/database/query"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {"query": sql}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Supabase SQL execution failed",
            extra={"status": exc.response.status_code, "body": exc.response.text[:200], "project_ref": project_ref},
        )
        raise HTTPException(status_code=exc.response.status_code, detail="Failed to provision Supabase RPC functions") from exc
    except Exception as exc:
        logger.exception("Supabase SQL execution failed", extra={"project_ref": project_ref})
        raise HTTPException(status_code=500, detail="Failed to provision Supabase RPC functions") from exc


async def _ensure_supabase_metadata_functions(resource: IntegrationResource) -> None:
    """
    Best-effort creation of metadata RPC helpers (list_schemas, list_tables).
    Uses Supabase management API when OAuth connection is available.
    """
    oauth_connection_rel = resource.oauth_connection
    if oauth_connection_rel is None:
        logger.info("Skipping metadata function provisioning: no OAuth connection on resource %s", resource.id)
        return
    oauth_connection = await oauth_connection_rel

    user = await resource.user
    _, access_token = await get_oauth_token(user, connection_id=str(oauth_connection.id), provider="supabase_mgmt")
    project_ref = resource.resource_key or (resource.resource_metadata or {}).get("project_ref")
    if not project_ref:
        logger.info("Skipping metadata function provisioning: missing project_ref on resource %s", resource.id)
        return

    sql = """
create or replace function public.list_schemas()
returns table(schema_name text)
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select n.nspname as schema_name
  from pg_namespace n
  where n.nspname not like 'pg_%'
    and n.nspname <> 'information_schema'
  order by n.nspname;
$$;
grant execute on function public.list_schemas() to service_role;

create or replace function public.list_tables(_schema text)
returns table(table_name text)
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select t.table_name
  from information_schema.tables t
  where t.table_schema = _schema
    and t.table_type = 'BASE TABLE'
  order by t.table_name;
$$;
grant execute on function public.list_tables(text) to service_role;
"""
    await _execute_supabase_sql(access_token=access_token, project_ref=project_ref, sql=sql)


async def _call_supabase_rpc(
    *,
    rest_url: str,
    service_role_key: str,
    function: str,
    payload: dict,
) -> list[dict]:
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


async def _browse_resources_unified(  # pylint: disable=too-many-arguments  # Keeps `browse_resources` small while staying explicit at call sites.
    *,
    user: User,
    provider: str,
    resource_type: str,
    q: Optional[str],
    parent_id: Optional[str],
    page_token: Optional[str],
    page_size: int,
    depends_on: Optional[str],
) -> dict[str, Any]:
    """
    Browse resources using ResourceContext and ResourceBrowser.

    Supports all providers (OAuth-backed, database-backed, and API-backed with bot tokens).
    Uses ResourceContext to handle different authentication types.

    This keeps `browse_resources` small enough to satisfy pylint complexity limits.
    """
    # Get access token for OAuth providers (optional - not all providers need it)
    access_token = await get_valid_access_token(user, provider)

    # For OAuth providers that require a token, validate it's present
    # Discord and Supabase schema/table resources don't require OAuth token
    oauth_providers = ["google", "github", "supabase_mgmt"]
    if provider in oauth_providers and access_token is None:
        msg = (
            f"No active {provider} connection. "
            f"Please connect your {provider} account first."
        )
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="No active connection",
            detail=msg,
            status=401,
        )

    # Parse depends_on values
    depends_on_values = None
    if depends_on:
        try:
            depends_on_values = json.loads(depends_on)
        except json.JSONDecodeError:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid JSON",
                detail="Invalid depends_on JSON",
                status=400
            )

    # Create ResourceContext with user and authentication info
    context = ResourceContext(
        user=user,
        access_token=access_token,
        bot_token=config.discord_bot_token if provider == "discord" else None,
    )

    # Create browser with context
    browser = ResourceBrowser(context=context, provider=provider)

    try:
        result = await browser.list_resources(
            resource_type=resource_type,
            options=ResourceListOptions(
                query=q,
                parent_id=parent_id,
                page_token=page_token,
                page_size=page_size,
                depends_on_values=depends_on_values,
            ),
        )

        if "error" in result and result["error"]:
            logger.error("Resource browser error: %s", result["error"])
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Resource browser error",
                detail=result["error"],
                status=500
            )

        return result

    except ValueError as exc:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid request",
            detail=str(exc),
            status=400
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught  # provider implementations may raise varied errors
        logger.exception("Error browsing resources: %s", exc)
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Resource browsing failed",
            detail=f"Error browsing resources: {str(exc)}",
            status=500
        )
    raise AssertionError("Unreachable")  # pragma: no cover


# Legacy function for backward compatibility - now delegates to unified handler
async def _browse_oauth_resources(  # pylint: disable=too-many-arguments  # Legacy compatibility function maintains original signature
    *,
    user: User,
    provider: str,
    resource_type: str,
    q: Optional[str],
    parent_id: Optional[str],
    page_token: Optional[str],
    page_size: int,
    depends_on: Optional[str],
) -> dict[str, Any]:
    """Deprecated: Use _browse_resources_unified instead."""
    return await _browse_resources_unified(
        user=user,
        provider=provider,
        resource_type=resource_type,
        q=q,
        parent_id=parent_id,
        page_token=page_token,
        page_size=page_size,
        depends_on=depends_on,
    )




async def _attempt_metadata_provision(resource: IntegrationResource) -> None:
    try:
        await _ensure_supabase_metadata_functions(resource)
    except HTTPException as exc:
        logger.info("Proceeding without auto-provisioning Supabase metadata functions: %s", exc.detail)


@router.get("/supabase/resources/schemas")
async def list_supabase_schemas(
    request: Request,
    *,
    integration_resource_id: Optional[int] = Query(
        None, description="Persisted Supabase project resource ID", ge=1
    ),
    depends_on: Optional[str] = Query(None, description="Dependent parameters (JSON)"),
    q: Optional[str] = Query(None, description="Search schema name"),
    page_token: Optional[str] = Query(None, description="Offset-based pagination token"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page (max 100)"),
):
    user: User = request.state.db_user

    depends_on_values = parse_depends_on(depends_on, error_detail="Invalid depends_on JSON for Supabase schemas")
    resource_id = resolve_resource_id(integration_resource_id, depends_on_values)

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    await _attempt_metadata_provision(resource)

    offset = parse_offset(page_token)
    raw_schemas = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_schemas",
        payload={},
    )

    names = filter_entries(raw_schemas, name_keys=("schema_name", "name"), query=q, skip_system=True)
    return paginate_items(names, page_size=page_size, offset=offset, item_type="schema")


@router.get("/supabase/resources/tables")
async def list_supabase_tables(
    request: Request,
    *,
    integration_resource_id: Optional[int] = Query(
        None, description="Persisted Supabase project resource ID", ge=1
    ),
    schema: Optional[str] = Query("public", description="Schema to list tables from"),
    q: Optional[str] = Query(None, description="Search table name"),
    depends_on: Optional[str] = Query(None, description="Dependent parameters (JSON)"),
    page_token: Optional[str] = Query(None, description="Offset-based pagination token"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page (max 100)"),
):
    user: User = request.state.db_user

    depends_on_values = parse_depends_on(depends_on, error_detail="Invalid depends_on JSON for Supabase tables")
    resource_id = resolve_resource_id(integration_resource_id, depends_on_values)
    schema_override = depends_on_values.get("schema")
    schema_name = (
        schema_override.strip()
        if isinstance(schema_override, str) and schema_override.strip()
        else (schema or "public").strip() or "public"
    )

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    await _attempt_metadata_provision(resource)

    offset = parse_offset(page_token)
    raw_tables = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_tables",
        payload={"_schema": schema_name},
    )

    names = filter_entries(raw_tables, name_keys=("table_name", "name"), query=q, skip_system=False)
    return paginate_items(names, page_size=page_size, offset=offset, item_type="table", description=schema_name)


# =============================================================================
# RESOURCE BROWSER ROUTES - For browsing integration resources
# =============================================================================

@router.get("/resources/types")
async def list_resource_types(_request: Request):
    """
    List all supported resource types across all providers.

    Returns configuration info for each resource type including
    whether it supports hierarchy, search, and dependencies.
    """
    all_types = {}
    # Include all registered providers
    for provider in ["google", "github", "discord", "supabase_mgmt"]:
        types = ResourceBrowser.get_supported_resource_types(provider)
        for rt in types:
            info = ResourceBrowser.get_resource_type_info(rt)
            if info:
                info["provider"] = provider
                all_types[rt] = info

    return {"resource_types": all_types}


@router.get("/resources/{provider}/types")
async def list_provider_resource_types(_request: Request, provider: str):
    """
    List supported resource types for a specific provider.

    Args:
        provider: OAuth provider (google, github, etc.)
    """
    types = ResourceBrowser.get_supported_resource_types(provider)
    result = {}
    for rt in types:
        info = ResourceBrowser.get_resource_type_info(rt)
        if info:
            result[rt] = info

    return {"provider": provider, "resource_types": result}


@router.get("/resources/{provider}/{resource_type}")
# pylint: disable=too-many-arguments,too-many-positional-arguments
# Reason: FastAPI endpoint signature matches REST API contract required by ResourcePicker UI
async def browse_resources(
    request: Request,
    provider: str,
    resource_type: str,
    *,
    q: Optional[str] = Query(None, description="Search query"),
    parent_id: Optional[str] = Query(
        None, description="Parent folder ID for hierarchy navigation"
    ),
    page_token: Optional[str] = Query(None, description="Pagination token"),
    page_size: int = Query(
        50, ge=1, le=100, description="Number of items per page"
    ),
    depends_on: Optional[str] = Query(
        None, description="JSON object of dependent parameter values"
    ),
):
    """
    Browse resources of a specific type.

    This endpoint powers the ResourcePicker UI component, allowing users
    to browse and select resources (files, spreadsheets, repos, etc.)
    instead of manually entering IDs.

    Args:
        provider: OAuth provider (google, github)
        resource_type: Type of resource to browse (google_spreadsheet, github_repo, etc.)
        q: Optional search query
        parent_id: Parent folder ID for hierarchical navigation (Google Drive)
        page_token: Token for pagination
        page_size: Number of results per page (max 100)
        depends_on: JSON object with values for dependent parameters

    Returns:
        List of resources with metadata for display
    """
    user: User = request.state.db_user

    # Use unified resource browsing for all providers (including Discord, Supabase, etc.)
    return await _browse_resources_unified(
        user=user,
        provider=provider,
        resource_type=resource_type,
        q=q,
        parent_id=parent_id,
        page_token=page_token,
        page_size=page_size,
        depends_on=depends_on,
    )
