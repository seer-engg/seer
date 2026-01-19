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

from seer.services.integrations.providers import get_integration_provider
from seer.services.integrations.providers.base import OAuthAuthorizeContext, OAuthHelpers
from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.database import User
from seer.logger import get_logger

from seer.services.integrations.auth.oauth import oauth
from .services import (
    delete_connection_by_id,
    disconnect_provider,
    get_connection_for_provider,
    list_integration_resources,
    serialize_integration_resource,
)
from seer.services.integrations.auth.oauth import get_oauth_provider
from .models import ToolsStatusResponse
logger = get_logger("api.integrations.router")
from seer.services.integrations.auth.helpers import parse_scopes, has_required_scopes, list_connections, store_oauth_connection
from seer.services.integrations.tool_status_service import get_tools_connection_status_for_user

router = APIRouter(prefix="/integrations", tags=["integrations"])
from seer.api.integrations.resource_router import router as resource_router



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
    user: User = request.state.db_user
    results = await get_tools_connection_status_for_user(user)
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


router.include_router(resource_router)
