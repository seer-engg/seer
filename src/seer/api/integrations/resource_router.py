"""Integration resource routes."""
# pylint: disable=duplicate-code,too-many-lines
# Reason: Supabase REST validation mirrors supabase database tool usage; this module aggregates many integration endpoints and helpers.
import json
from typing import Any, Dict, Iterable, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.integrations.models import SupabaseBindRequest, SupabaseManualBindRequest
from seer.api.integrations.services import (
    bind_supabase_project,
    bind_supabase_project_manual,
    deactivate_integration_resource,
    get_valid_access_token,
    list_integration_resources,
    list_resource_secrets,
    serialize_integration_resource,
    serialize_integration_secret,
)
from seer.config import config
from seer.database import IntegrationResource, IntegrationSecret, User
from seer.logger import get_logger
from seer.services.integrations.constants import SUPABASE_RESOURCE_PROVIDER
from seer.services.integrations.providers.discord import DiscordProvider
from seer.services.integrations.resource_browser import ResourceBrowser, ResourceListOptions
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


def _parse_depends_on(depends_on: Optional[str], *, error_detail: str) -> dict[str, Any]:
    if not depends_on:
        return {}
    try:
        parsed = json.loads(depends_on)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=error_detail) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=error_detail)
    return parsed


def _resolve_resource_id(
    integration_resource_id: Optional[int],
    depends_on_values: dict[str, Any],
) -> int:
    if integration_resource_id is not None:
        return integration_resource_id
    candidate = depends_on_values.get("integration_resource_id")
    if candidate is None:
        raise HTTPException(status_code=400, detail="integration_resource_id is required")
    try:
        return int(candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="integration_resource_id is required") from exc


async def _browse_discord_channels(
    user: User,
    guild_id: str,
    q: Optional[str],
    page_token: Optional[str],
    page_size: int,
) -> Dict[str, Any]:
    """
    Browse Discord channels for a specific guild.

    Args:
        user: Current user
        guild_id: Discord guild ID
        q: Optional search query
        page_token: Pagination token
        page_size: Number of results per page

    Returns:
        Dictionary with items, next_page_token, and metadata
    """
    # Verify user has access to this guild
    guild_resource = await IntegrationResource.get_or_none(
        user=user,
        provider="discord",
        resource_type="guild",
        resource_id=guild_id,
        status="active"
    )
    if not guild_resource:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Guild not found",
            detail=f"Discord guild {guild_id} not found or you don't have access to it",
            status=404
        )

    bot_token = config.discord_bot_token
    if not bot_token:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Configuration error",
            detail="Discord bot token not configured",
            status=500
        )
    assert bot_token is not None

    provider_impl = DiscordProvider()
    try:
        channels = await provider_impl.fetch_guild_channels(
            guild_id=guild_id,
            bot_token=bot_token
        )
    except HTTPException as exc:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Failed to fetch channels",
            detail=exc.detail,
            status=exc.status_code
        )

    # Filter channels (only text channels that bot can send messages to)
    # Channel types: 0=GUILD_TEXT, 2=GUILD_VOICE, 4=GUILD_CATEGORY, 5=GUILD_NEWS, 15=GUILD_FORUM
    # We'll include text channels (0), news channels (5), and forum channels (15)
    text_channels = [
        ch for ch in channels
        if ch.get("type") in [0, 5, 15]  # GUILD_TEXT, GUILD_NEWS, GUILD_FORUM
    ]

    # Apply search filter if provided
    filtered_channels = text_channels
    if q:
        q_lower = q.lower()
        filtered_channels = [
            ch for ch in text_channels
            if q_lower in (ch.get("name") or "").lower()
        ]

    # Pagination
    offset = _parse_offset(page_token)
    paged_channels = filtered_channels[offset:offset + page_size]

    items = [
        {
            "id": str(ch.get("id", "")),
            "name": ch.get("name") or f"Channel {ch.get('id', '')}",
            "display_name": ch.get("name") or f"Channel {ch.get('id', '')}",
            "type": "channel",
            "metadata": {
                "channel_id": str(ch.get("id", "")),
                "channel_name": ch.get("name"),
                "channel_type": ch.get("type"),
                "guild_id": guild_id,
            },
        }
        for ch in paged_channels
    ]

    next_page_token = str(offset + page_size) if offset + page_size < len(filtered_channels) else None

    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


async def _browse_discord_guilds(
    user: User,
    q: Optional[str],
    page_token: Optional[str],
    page_size: int,
) -> Dict[str, Any]:
    """
    Browse Discord guilds (servers) for the current user.

    Args:
        user: Current user
        q: Optional search query
        page_token: Pagination token
        page_size: Number of results per page

    Returns:
        Dictionary with items, next_page_token, and metadata
    """
    # List guilds from IntegrationResource records
    resources = await list_integration_resources(
        user,
        provider="discord",
        resource_type="guild",
    )

    # Apply search filter if provided
    filtered_resources = resources
    if q:
        q_lower = q.lower()
        filtered_resources = [
            r for r in resources
            if q_lower in (r.name or "").lower() or q_lower in (r.resource_id or "").lower()
        ]

    # Pagination
    offset = _parse_offset(page_token)
    paged_resources = filtered_resources[offset:offset + page_size]

    items = [
        {
            "id": r.resource_id,  # guild_id
            "name": r.name or f"Discord Server {r.resource_id}",
            "display_name": r.name or f"Discord Server {r.resource_id}",
            "type": "guild",
            "metadata": r.resource_metadata or {},
        }
        for r in paged_resources
    ]

    next_page_token = str(offset + page_size) if offset + page_size < len(filtered_resources) else None

    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


async def _browse_oauth_resources(  # pylint: disable=too-many-arguments  # Keeps `browse_resources` small while staying explicit at call sites.
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
    Browse API-backed resources using an OAuth access token and ResourceBrowser.

    This keeps `browse_resources` small enough to satisfy pylint complexity limits.
    """
    access_token = await get_valid_access_token(user, provider)
    if access_token is None:
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
    assert access_token is not None

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

    browser = ResourceBrowser(access_token, provider)

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


def _parse_offset(page_token: Optional[str]) -> int:
    if not page_token:
        return 0
    try:
        return int(page_token)
    except ValueError:
        return 0


def _extract_name(entry: Any, keys: Iterable[str]) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value:
                return str(value)
    return None


def _filter_entries(raw_entries: list[Any], *, name_keys: Iterable[str], query: Optional[str], skip_system: bool) -> list[str]:
    filtered: list[str] = []
    for entry in raw_entries:
        name = _extract_name(entry, name_keys)
        if not name:
            continue
        if skip_system and (name == "information_schema" or name.startswith("pg_")):
            continue
        filtered.append(name)
    if query:
        lowered = query.lower()
        filtered = [name for name in filtered if lowered in name.lower()]
    return filtered


def _paginate_items(
    names: list[str],
    *,
    page_size: int,
    offset: int,
    item_type: str,
    description: Optional[str] = None,
) -> dict[str, Any]:
    paged = names[offset: offset + page_size]
    items = [
        {
            "id": name,
            "name": name,
            "display_name": name,
            "type": item_type,
            **({"description": description} if description else {}),
        }
        for name in paged
    ]
    next_page_token = str(offset + page_size) if offset + page_size < len(names) else None
    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


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

    depends_on_values = _parse_depends_on(depends_on, error_detail="Invalid depends_on JSON for Supabase schemas")
    resource_id = _resolve_resource_id(integration_resource_id, depends_on_values)

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    await _attempt_metadata_provision(resource)

    offset = _parse_offset(page_token)
    raw_schemas = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_schemas",
        payload={},
    )

    names = _filter_entries(raw_schemas, name_keys=("schema_name", "name"), query=q, skip_system=True)
    return _paginate_items(names, page_size=page_size, offset=offset, item_type="schema")


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

    depends_on_values = _parse_depends_on(depends_on, error_detail="Invalid depends_on JSON for Supabase tables")
    resource_id = _resolve_resource_id(integration_resource_id, depends_on_values)
    schema_override = depends_on_values.get("schema")
    schema_name = (
        schema_override.strip()
        if isinstance(schema_override, str) and schema_override.strip()
        else (schema or "public").strip() or "public"
    )

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    await _attempt_metadata_provision(resource)

    offset = _parse_offset(page_token)
    raw_tables = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_tables",
        payload={"_schema": schema_name},
    )

    names = _filter_entries(raw_tables, name_keys=("table_name", "name"), query=q, skip_system=False)
    return _paginate_items(names, page_size=page_size, offset=offset, item_type="table", description=schema_name)


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
    for provider in ["google", "github"]:
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

    # Special handling for Discord channels (API-backed, requires guild_id)
    if provider == "discord" and resource_type == "channel":
        depends_on_values = _parse_depends_on(depends_on, error_detail="Invalid depends_on JSON")
        guild_id = str(depends_on_values.get("guild_id") or "")
        if not guild_id:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Missing required parameter",
                detail="guild_id is required to list Discord channels. Please select a guild first.",
                status=400
            )

        return await _browse_discord_channels(user, guild_id, q, page_token, page_size)

    # Special handling for Discord guilds (database-backed, not API-backed)
    if provider == "discord" and resource_type == "guild":
        return await _browse_discord_guilds(user, q, page_token, page_size)

    return await _browse_oauth_resources(
        user=user,
        provider=provider,
        resource_type=resource_type,
        q=q,
        parent_id=parent_id,
        page_token=page_token,
        page_size=page_size,
        depends_on=depends_on,
    )
