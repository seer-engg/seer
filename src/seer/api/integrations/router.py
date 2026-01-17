# pylint: disable=too-many-lines,too-complex,too-many-positional-arguments,too-many-locals
# Reason: Integration router consolidates OAuth, resource management, and provider-specific endpoints.
# The Supabase schema/table endpoints have high complexity due to dynamic depends_on handling.
# TODO: Split into separate routers (oauth.py, resources.py, supabase.py) in future refactor.

# pylint: disable=relative-beyond-top-level,broad-exception-caught,raise-missing-from
# Reason: FastAPI router uses relative imports per project structure convention.
# Broad exception catching is intentional for graceful API error handling.
# Some exceptions are re-raised as HTTPException without chaining for cleaner API responses.

# pylint: disable=import-outside-toplevel,unused-argument
# Reason: Lazy imports for list_tools to avoid circular dependencies.
# FastAPI Request parameter is used by framework for dependency injection.

import base64
import json
import time
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from seer.api.integrations.providers import get_integration_provider
from seer.api.integrations.providers.base import OAuthAuthorizeContext, OAuthHelpers
from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.database import IntegrationResource, IntegrationSecret, User
from seer.logger import get_logger
from seer.tools.supabase.common import _resolve_rest_url, _service_headers
from seer.tools.oauth_manager import get_oauth_token

from .oauth import oauth
from .resource_browser import ResourceBrowser
from .services import (
    bind_supabase_project,
    bind_supabase_project_manual,
    deactivate_integration_resource,
    delete_connection_by_id,
    disconnect_provider,
    get_connection_for_provider,
    get_oauth_provider,
    get_valid_access_token,
    has_required_scopes,
    list_connections,
    list_integration_resources,
    list_resource_secrets,
    parse_scopes,
    serialize_integration_resource,
    serialize_integration_secret,
    store_oauth_connection,
)
from .constants import SUPABASE_RESOURCE_PROVIDER

logger = get_logger("api.integrations.router")

router = APIRouter(prefix="/integrations", tags=["integrations"])


class SupabaseBindRequest(BaseModel):
    project_ref: str = Field(..., min_length=3, description="Supabase project reference")
    connection_id: Optional[str] = Field(
        default=None,
        description="Specific Supabase OAuth connection ID (optional)",
    )


class SupabaseManualBindRequest(BaseModel):
    project_ref: str = Field(..., min_length=3, description="Supabase project reference")
    connection_id: Optional[str] = Field(
        default=None,
        description="Existing Supabase OAuth connection ID (skips manual secret input)",
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Friendly project display name",
    )
    service_role_key: Optional[str] = Field(
        default=None,
        description="Supabase service role key (required without connection_id)",
        min_length=8,
    )
    anon_key: Optional[str] = Field(
        default=None,
        description="Optional Supabase anon/public key",
    )


class ToolStatus(BaseModel):
    tool_name: str
    integration_type: Optional[str]
    provider: Optional[str]
    supports_oauth: bool
    supports_manual_secrets: bool
    connected: bool
    missing_scopes: List[str] = Field(default_factory=list)
    connection_id: Optional[str] = None
    provider_account_id: Optional[str] = None


class ToolsStatusResponse(BaseModel):
    tools: List[ToolStatus]


def encode_state(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_state(state: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(state).decode())


def _validate_scope_and_get_provider(scope: str, provider: str):
    if not scope:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing scope parameter",
            detail="scope parameter is required. Frontend must specify OAuth scopes.",
            status=400
        )
    oauth_provider = get_oauth_provider(provider)
    provider_impl = get_integration_provider(oauth_provider)
    if not provider_impl:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Provider not configured",
            detail=f"OAuth provider '{oauth_provider}' is not configured",
            status=400
        )
    return oauth_provider, provider_impl


def _check_existing_scopes(
    existing_connection,
    normalized_scope_list,
    oauth_provider: str,
    redirect_to: Optional[str],
    integration_type: Optional[str],
):
    if existing_connection and existing_connection.scopes and existing_connection.refresh_token_enc:
        if has_required_scopes(existing_connection.scopes, normalized_scope_list):
            logger.info(
                "User already has all required scopes for %s. Requested=%s Granted=%s",
                oauth_provider,
                normalized_scope_list,
                existing_connection.scopes[:100],
            )
            final_redirect = redirect_to or f"{config.FRONTEND_URL}/settings/integrations"
            connected_param = integration_type or oauth_provider
            return RedirectResponse(url=f"{final_redirect}?connected={connected_param}")
    return None


def _build_oauth_state(
    user: User,
    redirect_to: Optional[str],
    oauth_provider: str,
    integration_type: Optional[str],
    scope_string: str,
) -> str:
    state_data = {
        'user_id': user.user_id,
        'user_email': user.email,
        'redirect_to': redirect_to or f"{config.FRONTEND_URL}/settings/integrations",
        'oauth_provider': oauth_provider,
        'integration_type': integration_type or oauth_provider,
        'requested_scope': scope_string,
    }
    return encode_state(state_data)


def _extract_and_validate_state(request: Request):
    state = request.query_params.get('state')
    if not state:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing state parameter",
            detail="Missing state",
            status=400
        )
    try:
        state_data = decode_state(state)
    except Exception:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid state parameter",
            detail="Invalid state",
            status=400
        )
    user_id = state_data.get('user_id')
    if not user_id:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing user_id",
            detail="Missing user_id in state",
            status=400
        )
    return state_data


def _log_token_structure(token: dict) -> None:
    token_keys = list(token.keys())
    logger.info(
        "Token structure - Keys: %s, has userinfo: %s, has access_token: %s, "
        "has id_token: %s",
        token_keys,
        'userinfo' in token,
        'access_token' in token,
        'id_token' in token,
    )


def _log_scope_info(token: dict, granted_scopes: str, requested_scope: Optional[str]) -> None:
    requested = requested_scope.split() if requested_scope else []
    token_scope = token.get('scope', '')
    granted_list = token_scope.split() if token_scope else []
    storing = granted_scopes.split() if granted_scopes else []
    logger.info(
        "OAuth scopes - Requested: %s, Provider granted: %s, Storing: %s",
        requested,
        granted_list,
        storing,
    )


# =============================================================================
# STATIC ROUTES - Must come BEFORE dynamic routes to avoid path conflicts
# =============================================================================

@router.get("/")
async def list_integrations(request: Request):
    """
    List all integration connections for the current user.

    Returns connections organized by OAuth provider with scope information.
    Frontend can use this to determine which tools are connected.
    """
    user: User = request.state.db_user
    logger.info("Listing integrations for user %s", user.user_id)
    connections = await list_connections(user)
    res = []
    for conn in connections:
        # Construct composite ID so frontend can use it for deletion if needed
        composite_id = f"{conn.provider}:{conn.id}"

        res.append({
            "id": composite_id,
            "status": "ACTIVE" if conn.status == 'active' else "INACTIVE",
            "user_id": user.user_id,
            "toolkit": {
                "slug": conn.provider  # OAuth provider (google, github, etc.)
            },
            "connection": {
                "user_id": user.user_id,
                "provider_account_id": conn.provider_account_id
            },
            # Include scopes so frontend can check tool-level connectivity
            "scopes": conn.scopes or "",
            "provider": conn.provider
        })
    return {"items": res}


@router.get("/tools/status", response_model=ToolsStatusResponse)
async def get_tools_connection_status(request: Request):
    """
    Get connection status for all tools.

    Returns a list of all tools with their connection status based on
    whether the user has a connection with the required scopes.

    This is the primary endpoint for frontend to check which tools are connected.
    """
    from seer.tools.base import (
        list_tools as get_all_tools,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with tools.base module
    )

    from .tool_status_service import (  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with tool_status_service
        build_provider_connections_map,
        build_provider_secrets_map,
        build_tool_status,
        determine_tool_auth_requirements,
    )

    user: User = request.state.db_user
    logger.info("Getting tools connection status for user %s", user.user_id)

    connections = await list_connections(user)
    provider_connections = build_provider_connections_map(connections)
    provider_secrets = await build_provider_secrets_map(user)
    all_tools = get_all_tools()

    results = []
    for tool in all_tools:
        auth_requirements = determine_tool_auth_requirements(tool)
        tool_provider = tool.provider or tool.integration_type
        oauth_provider = get_oauth_provider(tool_provider) if tool_provider else None
        conn_info = provider_connections.get(oauth_provider) if oauth_provider else None

        results.append(build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider=oauth_provider,
            provider_aliases=[tool_provider] if tool_provider else [],
            conn_info=conn_info,
            provider_secrets=provider_secrets,
        ))

    return {"tools": results}


# =============================================================================
# DYNAMIC ROUTES - Must come AFTER static routes
# =============================================================================

@router.get("/{provider}/connect")
async def connect(
    request: Request,
    provider: str,
    redirect_to: str = Query(None),
    scope: str = Query(...),
    integration_type: str = Query(None),
):
    """
    Start OAuth flow for a provider.

    Args:
        provider: OAuth provider name (google, github)
        redirect_to: Redirect URL after auth
        scope: OAuth scope from frontend (REQUIRED - frontend controls which scopes to request)
        integration_type: Optional integration type that triggered this connection (for tracking)

    Note:
        Frontend must always pass scope parameter. This ensures frontend controls
        which permissions are requested (read-only is core differentiation).

        Connections are stored by OAuth provider (e.g., 'google'),
        not integration type. Multiple integration types
        (gmail, googlesheets, googledrive) share the same Google connection.

        If user already has all required scopes, OAuth is skipped and success
        is returned immediately. For Google OAuth, incremental authorization
        (include_granted_scopes=true) is only used when requesting NEW scopes
        in addition to existing ones, to avoid showing all previously granted
        scopes in the consent screen.
    """
    oauth_provider, provider_impl = _validate_scope_and_get_provider(scope, provider)
    requested_scopes_list = list(parse_scopes(scope))
    user: User = request.state.db_user
    existing_connection = await get_connection_for_provider(user, oauth_provider)

    authorize_context = OAuthAuthorizeContext(
        user=user,
        oauth_provider=oauth_provider,
        integration_type=integration_type or provider,
        requested_scopes=requested_scopes_list,
        existing_connection=existing_connection,
        helpers=OAuthHelpers(has_required_scopes=has_required_scopes),
    )

    scope_string = provider_impl.get_oauth_scope(authorize_context)
    normalized_scope_list = list(parse_scopes(scope_string))

    early_return = _check_existing_scopes(
        existing_connection, normalized_scope_list, oauth_provider, redirect_to, integration_type
    )
    if early_return:
        return early_return

    redirect_uri = request.url_for('auth_callback', provider=oauth_provider)
    if config.REDIRECT_URI_SCHEME == "https" and redirect_uri.scheme == "http":
        redirect_uri = redirect_uri.replace(scheme="https")

    logger.info(
        "Starting OAuth flow: provider=%s, integration_type=%s, scopes=%s",
        oauth_provider,
        integration_type,
        scope_string[:100],
    )
    state = _build_oauth_state(
        user, redirect_to, oauth_provider, integration_type, scope_string
    )

    client = oauth.create_client(oauth_provider)
    authorize_kwargs = provider_impl.build_authorize_kwargs(
        authorize_context, state=state, scope=scope_string
    )
    authorize_kwargs.setdefault("state", state)
    authorize_kwargs.setdefault("scope", scope_string)

    return await client.authorize_redirect(request, redirect_uri, **authorize_kwargs)


@router.get("/{provider}/callback", name="auth_callback")
async def auth_callback(request: Request, provider: str):
    """
    Handle OAuth callback from provider.

    Stores connection with OAuth provider (e.g., 'google'), merging scopes
    if a connection already exists for this provider.
    """
    oauth_provider = get_oauth_provider(provider)

    # Validate custom state FIRST (before Authlib's session-based validation)
    # This allows stateless OAuth that works across multiple workers
    state_data = _extract_and_validate_state(request)
    user_id = state_data['user_id']
    redirect_to = state_data.get('redirect_to')
    integration_type = state_data.get('integration_type')

    logger.info(
        "OAuth callback received: provider=%s, integration_type=%s, validating state",
        oauth_provider,
        integration_type,
    )

    # Extract authorization code from callback
    code = request.query_params.get('code')
    if not code:
        error = request.query_params.get('error')
        error_description = request.query_params.get('error_description', 'No authorization code received')
        logger.error("OAuth callback missing code: error=%s, description=%s", error, error_description)
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth callback error",
            detail=f"{error}: {error_description}" if error else error_description,
            status=400
        )

    # Manually exchange authorization code for tokens
    # This bypasses Authlib's session-based state validation which fails with multiple workers
    client = oauth.create_client(oauth_provider)
    redirect_uri = str(request.url_for('auth_callback', provider=oauth_provider))
    if config.REDIRECT_URI_SCHEME == "https" and "http://" in redirect_uri:
        redirect_uri = redirect_uri.replace("http://", "https://")

    try:
        token_url = client.server_metadata.get('token_endpoint') or client.access_token_url

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                token_url,
                data={
                    'grant_type': 'authorization_code',
                    'code': code,
                    'redirect_uri': redirect_uri,
                    'client_id': client.client_id,
                    'client_secret': client.client_secret,
                },
                headers={'Accept': 'application/json'},
                timeout=30.0,
            )
            response.raise_for_status()
            token = response.json()

        # Convert expires_in (seconds) to expires_at (timestamp)
        # This matches Authlib's token handling behavior
        if 'expires_in' in token and 'expires_at' not in token:
            token['expires_at'] = int(time.time()) + token['expires_in']

        logger.info("OAuth token exchange successful: provider=%s", oauth_provider)

    except httpx.HTTPStatusError as exc:
        # Specific handler for HTTP errors (400, 401, 500, etc.)
        logger.error(
            "OAuth token exchange failed",
            extra={
                "url": token_url,
                "status_code": exc.response.status_code,
                "body": exc.response.text[:500],
                "provider": oauth_provider,
            },
        )
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth token exchange error",
            detail=f"Token endpoint returned {exc.response.status_code}: {exc.response.text[:200]}",
            status=400,
        )
    except json.JSONDecodeError:
        # Specific handler for invalid JSON responses
        logger.error(
            "Invalid JSON response from token endpoint",
            extra={"url": token_url, "provider": oauth_provider},
        )
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth token exchange error",
            detail="Invalid response format from OAuth provider",
            status=400,
        )
    except Exception as exc:
        # Catch-all for unexpected errors (network, timeout, etc.)
        logger.exception(
            "Unexpected error during token exchange",
            extra={"url": token_url, "provider": oauth_provider},
        )
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth token exchange error",
            detail=f"Unexpected error: {type(exc).__name__}",
            status=500,
        )

    logger.info(
        "OAuth callback: provider=%s, integration_type=%s",
        oauth_provider,
        integration_type,
    )
    _log_token_structure(token)

    provider_impl = get_integration_provider(oauth_provider)
    if not provider_impl:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Provider not configured",
            detail=f"OAuth provider '{oauth_provider}' is not configured",
            status=400
        )

    granted_scopes = provider_impl.resolve_granted_scopes(
        token=token, state_data=state_data
    )
    _log_scope_info(token, granted_scopes, state_data.get('requested_scope'))

    user_info = await provider_impl.fetch_user_profile(
        client=client, token=token, state_data=state_data
    )

    await store_oauth_connection(
        user_id=user_id,
        provider=oauth_provider,
        token=token,
        profile=user_info,
        granted_scopes=granted_scopes,
        integration_type=integration_type
    )

    connected_param = integration_type or oauth_provider
    return RedirectResponse(url=f"{redirect_to}?connected={connected_param}")


@router.get("/{integration_type}/status")
async def get_integration_status(request: Request, integration_type: str):
    """
    Get connection status for a specific integration type.

    This checks if the user has a connection with the required scopes for
    all tools belonging to this integration type.

    Args:
        integration_type: Integration type (gmail, googlesheets, googledrive, github, etc.)

    Returns:
        Connection status including whether all required scopes are granted
    """
    from seer.tools.base import list_tools as get_all_tools

    user: User = request.state.db_user
    oauth_provider = get_oauth_provider(integration_type)

    # Get connection for this provider
    connections = await list_connections(user)
    conn = next((c for c in connections if c.provider == oauth_provider), None)

    if not conn:
        return {
            "integration_type": integration_type,
            "provider": oauth_provider,
            "connected": False,
            "has_required_scopes": False,
            "granted_scopes": [],
            "missing_scopes": [],
            "connection_id": None
        }

    # Get all tools for this integration type and collect required scopes
    all_tools = get_all_tools()
    integration_tools = [t for t in all_tools if t.integration_type == integration_type]

    # Collect all unique required scopes for this integration
    all_required_scopes = set()
    for tool in integration_tools:
        all_required_scopes.update(tool.required_scopes)

    granted_scopes = set(conn.scopes.split()) if conn.scopes else set()
    missing = list(all_required_scopes - granted_scopes)

    return {
        "integration_type": integration_type,
        "provider": oauth_provider,
        "connected": True,
        "has_required_scopes": len(missing) == 0,
        "granted_scopes": list(granted_scopes),
        "missing_scopes": missing,
        "connection_id": f"{conn.provider}:{conn.id}",
        "provider_account_id": conn.provider_account_id
    }


@router.post("/{provider}/disconnect")
async def disconnect(provider: str, request: Request):
    user: User = request.state.db_user
    await disconnect_provider(user, provider)
    return {"status": "success"}


@router.delete("/{connection_id}")
async def delete_connection(connection_id: str, request: Request):
    user: User = request.state.db_user
    await delete_connection_by_id(user, connection_id)
    return {"status": "success"}


# =============================================================================
# PERSISTED RESOURCE ROUTES
# =============================================================================

@router.get("/{provider}/resources/bindings")
async def list_persisted_resources(
    request: Request,
    provider: str,
    resource_type: Optional[str] = Query(
        None, description="Filter by resource type (e.g., project)"
    ),
):
    user: User = request.state.db_user
    resources = await list_integration_resources(
        user,
        provider=provider,
        resource_type=resource_type,
    )
    return {"items": [serialize_integration_resource(r) for r in resources]}


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
        if not payload.service_role_key:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Missing service_role_key",
                detail="service_role_key is required when connection_id is not provided",
                status=400
            )
        resource = await bind_supabase_project_manual(
            user,
            project_ref=payload.project_ref,
            service_role_key=payload.service_role_key,
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


async def _fetch_supabase_metadata(
    *,
    rest_url: str,
    service_role_key: str,
    path: str,
    params: dict,
) -> list[dict]:
    url = f"{rest_url.rstrip('/')}/{path.lstrip('/')}"
    headers = _service_headers(service_role_key)
    if path.startswith("information_schema."):
        headers["Accept-Profile"] = "information_schema"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return []
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Supabase metadata fetch failed %s %s %s",url, exc.response.status_code, exc.response.text[:200]
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail="Failed to fetch Supabase metadata. Please check your project binding.",
        )
    except Exception as exc:
        logger.exception("Supabase metadata fetch failed", extra={"url": url})
        raise HTTPException(status_code=500, detail="Failed to fetch Supabase metadata") from exc


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
    oauth_connection = await resource.oauth_connection
    if not oauth_connection:
        logger.info("Skipping metadata function provisioning: no OAuth connection on resource %s", resource.id)
        return

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


@router.get("/supabase/resources/schemas")
async def list_supabase_schemas(
    request: Request,
    integration_resource_id: Optional[int] = Query(
        None, description="Persisted Supabase project resource ID", ge=1
    ),
    depends_on: Optional[str] = Query(None, description="Dependent parameters (JSON)"),
    q: Optional[str] = Query(None, description="Search schema name"),
    page_token: Optional[str] = Query(None, description="Offset-based pagination token"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page (max 100)"),
):
    user: User = request.state.db_user

    resource_id = integration_resource_id
    if resource_id is None and depends_on:
        try:
            parsed = json.loads(depends_on)
            candidate = parsed.get("integration_resource_id")
            if candidate is not None:
                resource_id = int(candidate)
        except (ValueError, json.JSONDecodeError):
            raise HTTPException(status_code=400, detail="Invalid depends_on JSON for Supabase schemas")

    if resource_id is None:
        raise HTTPException(status_code=400, detail="integration_resource_id is required")

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    try:
        await _ensure_supabase_metadata_functions(resource)
    except HTTPException as exc:
        logger.info("Proceeding without auto-provisioning Supabase metadata functions: %s", exc.detail)

    offset = 0
    if page_token:
        try:
            offset = int(page_token)
        except ValueError:
            offset = 0

    raw_schemas = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_schemas",
        payload={},
    )

    filtered: list[str] = []
    for entry in raw_schemas:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = entry.get("schema_name") or entry.get("name")
        else:
            name = None
        if not name:
            continue
        if name == "information_schema" or name.startswith("pg_"):
            continue
        filtered.append(name)

    if q:
        filtered = [name for name in filtered if q.lower() in name.lower()]

    paged = filtered[offset: offset + page_size]

    items = [
        {
            "id": name,
            "name": name,
            "display_name": name,
            "type": "schema",
        }
        for name in paged
    ]

    next_page_token = str(offset + page_size) if offset + page_size < len(filtered) else None

    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


@router.get("/supabase/resources/tables")
async def list_supabase_tables(
    request: Request,
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

    resource_id = integration_resource_id
    schema_name = (schema or "public").strip() or "public"
    if depends_on:
        try:
            depends = json.loads(depends_on)
            schema_override = depends.get("schema")
            candidate = depends.get("integration_resource_id")
            if candidate is not None and resource_id is None:
                resource_id = int(candidate)
            if isinstance(schema_override, str) and schema_override.strip():
                schema_name = schema_override.strip()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid depends_on JSON for Supabase tables")

    if resource_id is None:
        raise HTTPException(status_code=400, detail="integration_resource_id is required")

    resource, service_role_key, rest_url = await _get_supabase_rest_context(user, resource_id)
    try:
        await _ensure_supabase_metadata_functions(resource)
    except HTTPException as exc:
        logger.info("Proceeding without auto-provisioning Supabase metadata functions: %s", exc.detail)

    offset = 0
    if page_token:
        try:
            offset = int(page_token)
        except ValueError:
            offset = 0

    raw_tables = await _call_supabase_rpc(
        rest_url=rest_url,
        service_role_key=service_role_key,
        function="list_tables",
        payload={"_schema": schema_name},
    )

    filtered: list[str] = []
    for entry in raw_tables:
        if isinstance(entry, str):
            table_name = entry
        elif isinstance(entry, dict):
            table_name = entry.get("table_name") or entry.get("name")
        else:
            table_name = None
        if not table_name:
            continue
        filtered.append(table_name)

    if q:
        filtered = [name for name in filtered if q.lower() in name.lower()]

    paged = filtered[offset: offset + page_size]

    items = [
        {
            "id": name,
            "name": name,
            "display_name": name,
            "type": "table",
            "description": schema_name,
        }
        for name in paged
    ]

    next_page_token = str(offset + page_size) if offset + page_size < len(filtered) else None

    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


# =============================================================================
# RESOURCE BROWSER ROUTES - For browsing integration resources
# =============================================================================

@router.get("/resources/types")
async def list_resource_types(request: Request):
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
async def list_provider_resource_types(request: Request, provider: str):
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

    # Get valid access token
    access_token = await get_valid_access_token(user, provider)
    if not access_token:
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

    # Parse depends_on if provided
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

    # Create browser and list resources
    browser = ResourceBrowser(access_token, provider)

    try:
        result = await browser.list_resources(
            resource_type=resource_type,
            query=q,
            parent_id=parent_id,
            page_token=page_token,
            page_size=page_size,
            depends_on_values=depends_on_values,
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

    except ValueError as e:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid request",
            detail=str(e),
            status=400
        )
    except Exception as e:
        logger.exception("Error browsing resources: %s", e)
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Resource browsing failed",
            detail=f"Error browsing resources: {str(e)}",
            status=500
        )
