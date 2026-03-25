"""Integration resource routes."""
# pylint: disable=duplicate-code,too-many-lines,broad-exception-caught
# Reason: Supabase REST validation mirrors supabase database tool usage; this module aggregates many integration endpoints and helpers.
# Broad exception catching is intentional in connection test endpoints for graceful error reporting to the user.
import json
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.integrations.models import PostgresBindRequest, PostgresTestRequest, SupabaseBindRequest, SupabaseManualBindRequest
from seer.api.integrations.services import (
    bind_postgres_database_manual,
    bind_supabase_project,
    bind_supabase_project_manual,
    deactivate_integration_resource,
    get_valid_access_token,
    list_postgres_bindings,
    list_resource_secrets,
    serialize_integration_resource,
    serialize_integration_secret,
    update_postgres_database,
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


# =============================================================================
# POSTGRES DATABASE BINDING ROUTES
# =============================================================================


@router.post("/postgres/databases/bind")
async def bind_postgres_database_route(request: Request, payload: PostgresBindRequest):
    """
    Bind a PostgreSQL database using either a connection string or individual fields.

    The database can be configured with an access mode:
    - restricted (default): Only SELECT, EXPLAIN, SHOW allowed
    - unrestricted: All SQL operations allowed
    """
    user: User = request.state.db_user

    resource = await bind_postgres_database_manual(
        user,
        name=payload.name,
        connection_string=payload.connection_string,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        db_user=payload.user,
        password=payload.password,
        ssl_mode=payload.ssl_mode,
        access_mode=payload.access_mode,
    )

    secrets = await list_resource_secrets(user, resource.id)
    return {
        "resource": serialize_integration_resource(resource),
        "secrets": [serialize_integration_secret(s) for s in secrets],
    }


@router.patch("/postgres/databases/{resource_id}")
async def update_postgres_database_route(request: Request, resource_id: int, payload: PostgresBindRequest):
    """
    Update an existing PostgreSQL database binding.

    If password is not provided, the existing credentials are kept.
    Only metadata fields (name, ssl_mode, access_mode) are updated.
    """
    user: User = request.state.db_user

    resource = await update_postgres_database(
        user,
        resource_id,
        name=payload.name,
        connection_string=payload.connection_string,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        db_user=payload.user,
        password=payload.password,
        ssl_mode=payload.ssl_mode,
        access_mode=payload.access_mode,
    )

    secrets = await list_resource_secrets(user, resource.id)
    return {
        "resource": serialize_integration_resource(resource),
        "secrets": [serialize_integration_secret(s) for s in secrets],
    }


@router.post("/postgres/databases/test")
async def test_postgres_connection_route(request: Request, payload: PostgresTestRequest):
    """
    Test a PostgreSQL connection without saving it.

    Returns connection status and basic database info.
    """
    import asyncpg  # pylint: disable=import-outside-toplevel  # Reason: Only needed for this endpoint

    from seer.tools.postgres.common import build_connection_string  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

    _ = request  # User validation happens via middleware

    # Build connection string
    if payload.connection_string:
        connection_string = payload.connection_string
    else:
        if not payload.host or not payload.database or not payload.user or not payload.password:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Missing required fields",
                detail="Either connection_string or (host, database, user, password) are required",
                status=400,
            )
        # Type assertions after validation (raise_problem never returns)
        assert payload.host is not None
        assert payload.database is not None
        assert payload.user is not None
        assert payload.password is not None
        connection_string = build_connection_string(
            host=payload.host,
            port=payload.port,
            database=payload.database,
            user=payload.user,
            password=payload.password,
            ssl_mode=payload.ssl_mode,
        )

    try:
        conn = await asyncpg.connect(connection_string, timeout=10.0)
        try:
            # Get basic info
            version = await conn.fetchval("SELECT version()")
            db_name = await conn.fetchval("SELECT current_database()")
            db_user = await conn.fetchval("SELECT current_user")

            return {
                "status": "ok",
                "connected": True,
                "database": db_name,
                "user": db_user,
                "server_version": version,
            }
        finally:
            await conn.close()

    except asyncpg.InvalidCatalogNameError as e:
        return {
            "status": "error",
            "connected": False,
            "error": f"Database does not exist: {str(e)}",
        }
    except asyncpg.InvalidPasswordError:
        return {
            "status": "error",
            "connected": False,
            "error": "Invalid username or password",
        }
    except OSError as e:
        return {
            "status": "error",
            "connected": False,
            "error": f"Cannot connect to server: {str(e)}",
        }
    except Exception as e:
        logger.exception("PostgreSQL connection test failed")
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
        }


@router.get("/postgres/resources/bindings")
async def list_postgres_bindings_route(request: Request):
    """
    List all saved PostgreSQL database bindings for the current user.

    Used by the resource picker to show available databases.
    """
    user: User = request.state.db_user
    resources = await list_postgres_bindings(user)

    return {
        "items": [
            {
                "id": str(r.id),
                "name": r.name or r.resource_key,
                "resource_key": r.resource_key,
                "metadata": r.resource_metadata or {},
            }
            for r in resources
        ],
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


# =============================================================================
# TRIGGER EVENT BROWSER ROUTES - For browsing real trigger events
# =============================================================================

def _validate_trigger_event_request(
    provider: str,
    trigger_key: str,
    provider_connection_id: Optional[int],
    subscription_id: Optional[int],
    filter_params: Optional[str],
):
    """Validate browse_trigger_events request parameters and return parsed config/filters."""
    # Import here to avoid circular imports
    from seer.services.integrations.trigger_event_browser import (  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
        TRIGGER_BROWSING_CONFIG,
        TRIGGER_PROVIDER_MAP,
    )

    browsing_config = TRIGGER_BROWSING_CONFIG.get(trigger_key)
    if browsing_config is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Unsupported trigger key",
            detail=f"Trigger key '{trigger_key}' does not support event browsing. "
                   f"Supported: {list(TRIGGER_BROWSING_CONFIG.keys())}",
            status=400,
        )
    assert browsing_config is not None  # For type narrowing after raise_problem

    expected_provider = TRIGGER_PROVIDER_MAP.get(trigger_key)
    if expected_provider != provider:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Provider mismatch",
            detail=f"Trigger '{trigger_key}' requires provider '{expected_provider}', not '{provider}'",
            status=400,
        )

    if browsing_config.mode == "polling" and provider_connection_id is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing parameter",
            detail=f"provider_connection_id is required for polling trigger '{trigger_key}'",
            status=400,
        )
    if browsing_config.mode == "persisted" and subscription_id is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing parameter",
            detail=f"subscription_id is required for persisted trigger '{trigger_key}'",
            status=400,
        )

    parsed_filter_params = None
    if filter_params:
        try:
            parsed_filter_params = json.loads(filter_params)
        except json.JSONDecodeError:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid JSON",
                detail="Invalid filter_params JSON",
                status=400,
            )

    return parsed_filter_params


@router.get("/resources/{provider}/trigger_events/{trigger_key:path}")
async def browse_trigger_events(  # pylint: disable=too-many-arguments,too-many-positional-arguments # Reason: FastAPI requires explicit query params for OpenAPI schema
    request: Request,
    provider: str,
    trigger_key: str,
    *,
    provider_connection_id: Optional[int] = Query(
        None, description="OAuth connection ID for polling triggers (gmail, discord)"
    ),
    subscription_id: Optional[int] = Query(
        None, description="Subscription ID for persisted triggers (webhooks, forms)"
    ),
    trigger_id: str = Query(
        "trigger_test", description="Trigger ID to use in envelope (for expression resolution)"
    ),
    page_token: Optional[str] = Query(None, description="Pagination token"),
    page_size: int = Query(5, ge=1, le=50, description="Number of items per page"),
    filter_params: Optional[str] = Query(
        None,
        description="JSON object with trigger-specific filters (e.g., {\"label_ids\": [\"INBOX\"], \"query\": \"is:unread\"})",
    ),
):
    """
    Browse real trigger events from a connected account or stored events for workflow testing.

    This endpoint allows users to browse actual events (like Gmail emails, Discord messages,
    webhook payloads, form submissions) for workflow testing.

    For polling triggers (gmail, discord): provide provider_connection_id
    For persisted triggers (webhooks, forms): provide subscription_id

    Each returned event includes a full trigger envelope ready for workflow execution.

    Args:
        provider: Provider name (google, discord, generic, supabase, form)
        trigger_key: Trigger type key (poll.gmail.email_received, webhook.generic, etc.)
        provider_connection_id: ID of the OAuth connection (for polling triggers)
        subscription_id: ID of the TriggerSubscription (for persisted triggers)
        trigger_id: Trigger ID for the envelope (used for expression resolution in workflows)
        page_token: Token for pagination
        page_size: Number of results per page (max 50)
        filter_params: JSON object with trigger-specific filters

    Returns:
        {
            "items": [...],  # List of trigger events with envelopes
            "next_page_token": "...",
            "trigger_key": "poll.gmail.email_received",
            "supports_search": true
        }

    Example (polling):
        GET /resources/google/trigger_events/poll.gmail.email_received
            ?provider_connection_id=123
            &filter_params={"label_ids":["INBOX"]}

    Example (persisted):
        GET /resources/generic/trigger_events/webhook.generic
            ?subscription_id=456
    """
    from seer.services.integrations.trigger_event_browser import (  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
        TriggerEventBrowser,
        TriggerEventListOptions,
    )

    user: User = request.state.db_user

    parsed_filter_params = _validate_trigger_event_request(
        provider, trigger_key, provider_connection_id, subscription_id, filter_params
    )

    browser = TriggerEventBrowser(user)
    try:
        result = await browser.list_events(
            provider_connection_id=provider_connection_id,
            subscription_id=subscription_id,
            options=TriggerEventListOptions(
                trigger_key=trigger_key,
                trigger_id=trigger_id,
                page_size=page_size,
                page_token=page_token,
                filter_params=parsed_filter_params,
            ),
        )
        return result
    except ValueError as exc:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid request",
            detail=str(exc),
            status=400,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Handle unexpected errors gracefully
        logger.exception("Error browsing trigger events")
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Event browsing failed",
            detail=f"Error browsing trigger events: {str(exc)}",
            status=500,
        )


@router.get("/trigger_events/types")
async def list_trigger_event_types(_request: Request):
    """
    List all trigger keys that support event browsing.

    Returns information about which triggers support browsing real events
    from connected accounts or stored events.
    """
    from seer.services.integrations.trigger_event_browser import (  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
        TRIGGER_BROWSING_CONFIG,
    )

    return {
        "trigger_keys": [
            {
                "trigger_key": key,
                "provider": cfg.provider,
                "mode": cfg.mode,
                "supports_search": cfg.supports_search,
            }
            for key, cfg in TRIGGER_BROWSING_CONFIG.items()
        ]
    }


# =============================================================================
# SLACK CHANNEL ACTIONS - For adding bot to channels
# =============================================================================

@router.post("/resources/slack/channel/{channel_id}/join")
async def join_slack_channel(
    request: Request,
    channel_id: str,
    workspace_id: str = Query(..., description="Slack workspace ID"),
):
    """
    Join the Slack bot to a channel.

    This allows the bot to monitor messages in the channel for triggers.
    Only works for public channels - private channels require manual invite.

    Args:
        channel_id: Slack channel ID to join
        workspace_id: Slack workspace ID

    Returns:
        Channel information after joining
    """
    from seer.services.integrations.providers.slack import SlackProvider  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
    from seer.services.integrations.resource_providers.slack import SlackResourceProvider  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

    user: User = request.state.db_user

    # Get bot token for workspace
    resource_provider = SlackResourceProvider()
    try:
        bot_token = await resource_provider.get_bot_token_for_workspace(user, workspace_id)
    except Exception as exc:
        logger.error("Failed to get bot token for workspace %s: %s", workspace_id, exc)
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Workspace access error",
            detail=str(exc),
            status=401,
        )
        raise  # Unreachable but satisfies type checker

    # Join channel
    provider = SlackProvider()
    try:
        result = await provider.join_channel(bot_token, channel_id)
        return {
            "ok": True,
            "channel": result,
        }
    except HTTPException as exc:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Failed to join channel",
            detail=exc.detail,
            status=exc.status_code,
        )
        raise  # Unreachable but satisfies type checker
