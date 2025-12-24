from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from .oauth import oauth
from .services import (
    store_oauth_connection,
    list_connections,
    disconnect_provider,
    delete_connection_by_id,
    get_oauth_provider,
    get_tool_connection_status,
    has_required_scopes,
    get_connection_for_provider,
    get_valid_access_token
)
from .resource_browser import ResourceBrowser
import json
import base64
import os
import logging
from typing import Optional
from shared.logger import get_logger
from shared.database.models import User
logger = get_logger("api.integrations.router")

router = APIRouter(prefix="/integrations", tags=["integrations"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

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
    """
    Get connection status for all tools.
    
    Returns a list of all tools with their connection status based on
    whether the user has a connection with the required scopes.
    
    This is the primary endpoint for frontend to check which tools are connected.
    """
    from shared.tools.base import list_tools as get_all_tools
    
    user: User = request.state.db_user
    logger.info(f"Getting tools connection status for user {user.user_id}")
    
    # Get all connections for this user
    connections = await list_connections(user)
    
    # Build a map of provider -> connection with scopes and refresh_token status
    provider_connections = {}
    for conn in connections:
        provider_connections[conn.provider] = {
            "scopes": conn.scopes or "",
            "connection_id": f"{conn.provider}:{conn.id}",
            "provider_account_id": conn.provider_account_id,
            "has_refresh_token": bool(conn.refresh_token_enc),  # Check if refresh_token exists
            "connection": conn  # Store connection object for refresh_token check
        }
    
    # Get all registered tools
    all_tools = get_all_tools()
    
    results = []
    for tool in all_tools:
        tool_provider = tool.provider or tool.integration_type
        if not tool_provider:
            # Non-OAuth tool
            results.append({
                "tool_name": tool.name,
                "integration_type": tool.integration_type,
                "provider": None,
                "connected": True,  # Non-OAuth tools are always "connected"
                "has_required_scopes": True,
                "missing_scopes": [],
                "connection_id": None
            })
            continue
        
        # Normalize to OAuth provider
        oauth_provider = get_oauth_provider(tool_provider)
        
        # Check if user has connection for this provider
        conn_info = provider_connections.get(oauth_provider)
        
        if not conn_info:
            results.append({
                "tool_name": tool.name,
                "integration_type": tool.integration_type,
                "provider": oauth_provider,
                "connected": False,
                "has_required_scopes": False,
                "missing_scopes": tool.required_scopes,
                "connection_id": None
            })
            continue
        
        # Check if connection has required scopes
        has_scopes = has_required_scopes(conn_info["scopes"], tool.required_scopes)
        
        # Check if refresh_token exists (needed for token refresh)
        has_refresh_token = conn_info.get("has_refresh_token", False)
        
        # Connection is fully functional only if it has scopes AND refresh_token
        # (refresh_token is needed for token refresh when access_token expires)
        fully_connected = has_scopes and has_refresh_token
        
        # Find missing scopes
        granted_set = set(conn_info["scopes"].split()) if conn_info["scopes"] else set()
        missing = [s for s in tool.required_scopes if s not in granted_set]
        
        results.append({
            "tool_name": tool.name,
            "integration_type": tool.integration_type,
            "provider": oauth_provider,
            "connected": fully_connected,  # Only True if scopes AND refresh_token exist
            "has_required_scopes": has_scopes,
            "has_refresh_token": has_refresh_token,
            "missing_scopes": missing,
            "connection_id": conn_info["connection_id"],
            "provider_account_id": conn_info["provider_account_id"]
        })
    
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
    integration_type: str = Query(None),  # Integration type for tracking (e.g., 'gmail', 'googlesheets')
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
        
        Connections are stored by OAuth provider (e.g., 'google'), not integration type.
        Multiple integration types (gmail, googlesheets, googledrive) share the same Google connection.
        
        For Google OAuth, we use incremental authorization (include_granted_scopes=true)
        to preserve previously granted scopes when adding new ones.
    """
    
    if not scope:
        raise HTTPException(status_code=400, detail="scope parameter is required. Frontend must specify OAuth scopes.")
    
    # Normalize to OAuth provider
    oauth_provider = get_oauth_provider(provider)
    
    redirect_uri = request.url_for('auth_callback', provider=oauth_provider)
    
    # Store user_id, scope, and final redirect in state
    # Scope is stored so we can save it when token is received
    user: User = request.state.db_user
    state_data = {
        'user_id': user.user_id,
        'user_email': user.email,
        'redirect_to': redirect_to or f"{FRONTEND_URL}/settings/integrations",
        'oauth_provider': oauth_provider,
        'integration_type': integration_type or provider,  # Track which integration triggered this
        'requested_scope': scope  # Store requested scope to save in callback
    }

    logger.info(f"Starting OAuth flow: provider={oauth_provider}, integration_type={integration_type}, scopes={scope[:100]}...")
    state = encode_state(state_data)
    
    client = oauth.create_client(oauth_provider)
    # Always pass scope from frontend - no defaults
    kwargs = {'state': state, 'scope': scope}
    
    # For Google OAuth, enable incremental authorization to preserve previously granted scopes
    # This ensures that when connecting a new tool (e.g., Google Sheets), existing scopes 
    # (e.g., Gmail) are included in the new access token instead of being replaced
    # See: https://developers.google.com/identity/protocols/oauth2/web-server#incrementalAuth
    if oauth_provider == 'google':
        kwargs['include_granted_scopes'] = 'true'
        kwargs['access_type'] = 'offline'
        kwargs['prompt'] = 'consent'
        
    return await client.authorize_redirect(request, redirect_uri, **kwargs)

@router.get("/{provider}/callback", name="auth_callback")
async def auth_callback(request: Request, provider: str):
    """
    Handle OAuth callback from provider.
    
    Stores connection with OAuth provider (e.g., 'google'), merging scopes
    if a connection already exists for this provider.
    """
    # Normalize to OAuth provider
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
        
    # Get user profile
    if oauth_provider == 'google':
        # Fetch from userinfo endpoint
        resp = await client.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        user_info = resp.json()
    elif oauth_provider == 'github':
        resp = await client.get('user', token=token)
        user_info = resp.json()
    else:
        user_info = {}
    
    # Extract granted scopes from token response
    # Token may contain 'scope' field (space-separated) or we use requested_scope
    granted_scopes = token.get('scope') or requested_scope or ''
    
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
    """
    Get connection status for a specific integration type.
    
    This checks if the user has a connection with the required scopes for
    all tools belonging to this integration type.
    
    Args:
        integration_type: Integration type (gmail, googlesheets, googledrive, github, etc.)
    
    Returns:
        Connection status including whether all required scopes are granted
    """
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
    # Dependent parameter values (JSON encoded)
    depends_on: Optional[str] = Query(None, description="JSON object of dependent parameter values"),
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
