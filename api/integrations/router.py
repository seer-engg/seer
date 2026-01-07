from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from .oauth import oauth
from .services import (
    store_oauth_connection,
    list_connections,
    disconnect_provider,
    delete_connection_by_id,
    get_oauth_provider,
    has_required_scopes,
    get_connection_for_provider,
    get_valid_access_token,
    parse_scopes,
    list_integration_resources,
    list_resource_secrets,
    deactivate_integration_resource,
    serialize_integration_resource,
    serialize_integration_secret,
    bind_supabase_project,
    bind_supabase_project_manual,
    build_provider_connection_info,
    determine_auth_mode,
    build_tool_result_base,
)
from .resource_browser import ResourceBrowser
import json
import base64
import os
from typing import Optional
from shared.logger import get_logger
from shared.database.models import User
from api.integrations.providers import get_integration_provider
from api.integrations.providers.base import OAuthAuthorizeContext, OAuthHelpers
logger = get_logger("api.integrations.router")
from shared.config import config

router = APIRouter(prefix="/integrations", tags=["integrations"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


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
def encode_state(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_state(state: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(state).decode())


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
    logger.info(f"Listing integrations for user {user.user_id}")
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


@router.get("/tools/status")
async def get_tools_connection_status(request: Request):
    """Get connection status for all tools with OAuth/secret requirements."""
    from shared.tools.base import list_tools as get_all_tools
    user: User = request.state.db_user
    logger.info(f"Getting tools connection status for user {user.user_id}")

    connections = await list_connections(user)
    provider_connections = {conn.provider: build_provider_connection_info(conn) for conn in connections}

    # Get all registered tools
    all_tools = get_all_tools()

    results = []
    for tool in all_tools:
        tool_provider = tool.provider or tool.integration_type
        required_scopes = list(tool.required_scopes or [])
        required_secrets = list(getattr(tool, "required_secrets", []) or [])
        auth_mode, requires_oauth, requires_secrets, supports_tokenless_auth = determine_auth_mode(required_scopes, required_secrets)
        base_result = build_tool_result_base(tool, auth_mode, requires_oauth, requires_secrets, supports_tokenless_auth)

        def build_result(extra: dict) -> dict:
            result = base_result.copy()
            result.update(extra)
            return result

        if not tool_provider:
            # Non-OAuth tool
            results.append(build_result({
                "provider": None,
                "connected": True,  # Non-OAuth tools are always "connected"
                "has_required_scopes": True,
                "access_token_valid": True,  # Non-OAuth tools don't need tokens
                "missing_scopes": [],
                "connection_id": None,
                "provider_account_id": None,
                "has_refresh_token": False,
            }))
            continue

        # Normalize to OAuth provider
        oauth_provider = get_oauth_provider(tool_provider)
        conn_info = provider_connections.get(oauth_provider) if oauth_provider else None

        if not requires_oauth:
            # Tokens are optional; treat as connected even without an OAuth record
            results.append(build_result({
                "provider": oauth_provider,
                "connected": True,
                "has_required_scopes": True,
                "access_token_valid": True,
                "missing_scopes": [],
                "connection_id": conn_info["connection_id"] if conn_info else None,
                "provider_account_id": conn_info["provider_account_id"] if conn_info else None,
                "has_refresh_token": conn_info["has_refresh_token"] if conn_info else False,
            }))
            continue

        if not conn_info:
            results.append(build_result({
                "provider": oauth_provider,
                "connected": False,
                "has_required_scopes": False,
                "access_token_valid": False,  # No connection means no valid token
                "missing_scopes": required_scopes,
                "connection_id": None,
                "provider_account_id": None,
                "has_refresh_token": False,
            }))
            continue

        # Check if connection has required scopes
        has_scopes = has_required_scopes(conn_info["scopes"], required_scopes)

        # Check if access token is valid (exists and not expired)
        access_token_valid = conn_info.get("access_token_valid", False)

        # Check if refresh_token exists (needed for token refresh)
        has_refresh_token = conn_info.get("has_refresh_token", False)

        # Connection is functional if scopes present AND access token valid
        # Access token is valid if: (exists and not expired) OR refresh token exists
        fully_connected = has_scopes and access_token_valid

        # Find missing scopes - use parse_scopes() to handle both comma and space-separated formats
        granted_set = parse_scopes(conn_info["scopes"]) if conn_info["scopes"] else set()
        missing = [s for s in required_scopes if s not in granted_set]

        results.append(build_result({
            "provider": oauth_provider,
            "connected": fully_connected,  # True if scopes present AND access token valid
            "has_required_scopes": has_scopes,
            "access_token_valid": access_token_valid,  # Whether access token exists and (is not expired or can be refreshed)
            "has_refresh_token": has_refresh_token,  # Whether refresh token exists (for warnings)
            "missing_scopes": missing,
            "connection_id": conn_info["connection_id"],
            "provider_account_id": conn_info["provider_account_id"],
        }))

    return {"tools": results}


# =============================================================================
# DYNAMIC ROUTES - Must come AFTER static routes
# =============================================================================

@router.get("/{provider}/connect")
async def connect(
    request: Request,
    provider: str,
    redirect_to: str = Query(None),
    scope: str = Query(...),  # OAuth scope from frontend (REQUIRED - frontend controls scopes)
    integration_type: str = Query(None),
):
    """Start OAuth flow for a provider. Skips if user already has required scopes."""
    if not scope:
        raise HTTPException(status_code=400, detail="scope parameter is required. Frontend must specify OAuth scopes.")

    oauth_provider = get_oauth_provider(provider)
    provider_impl = get_integration_provider(oauth_provider)
    if not provider_impl:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{oauth_provider}' is not configured")

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

    if existing_connection and existing_connection.scopes and existing_connection.refresh_token_enc:
        if has_required_scopes(existing_connection.scopes, normalized_scope_list):
            logger.info(
                "User already has all required scopes for %s. Requested=%s Granted=%s",
                oauth_provider,
                normalized_scope_list,
                existing_connection.scopes[:100],
            )
            final_redirect = redirect_to or f"{FRONTEND_URL}/settings/integrations"
            connected_param = integration_type or oauth_provider
            return RedirectResponse(url=f"{final_redirect}?connected={connected_param}")

    redirect_uri = request.url_for('auth_callback', provider=oauth_provider)
    if config.seer_mode == "cloud" and redirect_uri.scheme == "http":
        redirect_uri = redirect_uri.replace(scheme="https")

    state_data = {
        'user_id': user.user_id,
        'user_email': user.email,
        'redirect_to': redirect_to or f"{FRONTEND_URL}/settings/integrations",
        'oauth_provider': oauth_provider,
        'integration_type': integration_type or provider,
        'requested_scope': scope_string,
    }
    logger.info(
        "Starting OAuth flow: provider=%s, integration_type=%s, scopes=%s",
        oauth_provider,
        integration_type,
        scope_string[:100],
    )
    state = encode_state(state_data)

    client = oauth.create_client(oauth_provider)
    authorize_kwargs = provider_impl.build_authorize_kwargs(
        authorize_context,
        state=state,
        scope=scope_string,
    )
    authorize_kwargs.setdefault("state", state)
    authorize_kwargs.setdefault("scope", scope_string)

    return await client.authorize_redirect(request, redirect_uri, **authorize_kwargs)

@router.get("/{provider}/callback", name="auth_callback")
async def auth_callback(request: Request, provider: str):
    """Handle OAuth callback from provider, stores connection and merges scopes."""
    oauth_provider = get_oauth_provider(provider)

    client = oauth.create_client(oauth_provider)
    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Retrieve user_id from state
    # Authlib validates state match, but we need to extract data from it.
    state = request.query_params.get('state')
    if not state:
        raise HTTPException(status_code=400, detail="Missing state")

    try:
        state_data = decode_state(state)
        user_id = state_data.get('user_id')
        redirect_to = state_data.get('redirect_to')
        requested_scope = state_data.get('requested_scope')
        integration_type = state_data.get('integration_type')  # Track which integration triggered this
    except:
        raise HTTPException(status_code=400, detail="Invalid state")

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id in state")

    logger.info(f"OAuth callback: provider={oauth_provider}, integration_type={integration_type}")

    # Log token structure for debugging (without sensitive values)
    token_keys = list(token.keys())
    has_userinfo = 'userinfo' in token
    has_access_token = 'access_token' in token
    has_id_token = 'id_token' in token
    logger.info(
        f"Token structure - Keys: {token_keys}, "
        f"has userinfo: {has_userinfo}, "
        f"has access_token: {has_access_token}, "
        f"has id_token: {has_id_token}"
    )

    provider_impl = get_integration_provider(oauth_provider)
    if not provider_impl:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{oauth_provider}' is not configured")

    granted_scopes = provider_impl.resolve_granted_scopes(token=token, state_data=state_data)

    requested_scopes_list = requested_scope.split() if requested_scope else []
    granted_scopes_list = token.get('scope', '').split() if token.get('scope') else []
    storing_scopes_list = granted_scopes.split() if granted_scopes else []
    logger.info(
        f"OAuth scopes - Requested: {requested_scopes_list}, "
        f"Provider granted: {granted_scopes_list}, "
        f"Storing: {storing_scopes_list}"
    )

    user_info = await provider_impl.fetch_user_profile(client=client, token=token, state_data=state_data)

    # Store connection with OAuth provider (not integration type)
    # Scopes will be merged if connection already exists
    await store_oauth_connection(
        user_id=user_id,
        provider=oauth_provider,
        token=token,
        profile=user_info,
        granted_scopes=granted_scopes,
        integration_type=integration_type
    )

    # Return with integration_type so frontend knows which tool was connected
    connected_param = integration_type or oauth_provider
    return RedirectResponse(url=f"{redirect_to}?connected={connected_param}")

@router.get("/{integration_type}/status")
async def get_integration_status(request: Request, integration_type: str):
    """Get connection status for a specific integration type (gmail, googlesheets, github, etc.)."""
    from shared.tools.base import list_tools as get_all_tools
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
    resource_type: Optional[str] = Query(None, description="Filter by resource type (e.g., project)"),
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
    """Persist Supabase project using manual secrets or OAuth connection_id."""
    user: User = request.state.db_user

    if payload.connection_id:
        resource = await bind_supabase_project(user, payload.project_ref, payload.connection_id)
    else:
        if not payload.service_role_key:
            raise HTTPException(
                status_code=400,
                detail="service_role_key is required when connection_id is not provided",
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
async def browse_resources(
    request: Request,
    provider: str,
    resource_type: str,
    q: Optional[str] = Query(None, description="Search query"),
    parent_id: Optional[str] = Query(None, description="Parent folder ID for hierarchy navigation"),
    page_token: Optional[str] = Query(None, description="Pagination token"),
    page_size: int = Query(50, ge=1, le=100, description="Number of items per page"),
    depends_on: Optional[str] = Query(None, description="JSON object of dependent parameter values"),
):
    """Browse resources (files, spreadsheets, repos) for ResourcePicker UI."""
    user: User = request.state.db_user

    # Get valid access token
    access_token = await get_valid_access_token(user, provider)
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"No active {provider} connection. Please connect your {provider} account first."
        )

    # Parse depends_on if provided
    depends_on_values = None
    if depends_on:
        try:
            depends_on_values = json.loads(depends_on)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid depends_on JSON")

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
            logger.error(f"Resource browser error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error browsing resources: {e}")
        raise HTTPException(status_code=500, detail=f"Error browsing resources: {str(e)}")
