# pylint: disable=too-many-lines,too-complex,too-many-positional-arguments,too-many-locals,too-many-statements
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

# pylint: disable=wrong-import-position,wrong-import-order
# Reason: Imports organized by logical grouping for router setup
from typing import Optional

from authlib.integrations.base_client import OAuthError
from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse

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
from .models import ToolsStatusResponse, AccountInfo, ConnectionsByProvider, ToolAccountInfo, ToolAccountsResponse
from .metadata_models import IntegrationMetadataResponse
logger = get_logger("api.integrations.router")
from seer.services.integrations.auth.helpers import (
    get_connection_display_name,
    has_required_scopes,
    list_connections,
    parse_scopes,
    store_oauth_connection,
)
from seer.services.integrations.tool_status_service import get_tools_connection_status_for_user, PROVIDERS_WITHOUT_REFRESH_TOKENS

router = APIRouter(prefix="/integrations", tags=["integrations"])
from seer.api.integrations.resource_router import router as resource_router


def _build_oauth_redirect_uri(request: Request, oauth_provider: str) -> str:
    """Build OAuth callback redirect URI with webhook_base_url fallback."""
    if config.webhook_base_url:
        # pylint: disable=no-member  # Pydantic Field descriptor; actual type is str at runtime
        return f"{config.webhook_base_url.rstrip('/')}/api/integrations/{oauth_provider}/callback"

    redirect_uri = str(request.url_for('auth_callback', provider=oauth_provider))
    if config.redirect_uri_scheme == "https" and "http://" in redirect_uri:
        redirect_uri = redirect_uri.replace("http://", "https://")
    return redirect_uri


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
    # Check if this provider requires refresh tokens for re-consent logic
    requires_refresh = oauth_provider not in PROVIDERS_WITHOUT_REFRESH_TOKENS
    has_valid_connection = (
        existing_connection
        and existing_connection.scopes
        and (existing_connection.refresh_token_enc or not requires_refresh)
    )

    if has_valid_connection:
        if has_required_scopes(existing_connection.scopes, normalized_scope_list):
            logger.info(
                "User already has all required scopes for %s. Requested=%s Granted=%s",
                oauth_provider,
                normalized_scope_list,
                existing_connection.scopes[:100],
            )
            final_redirect = redirect_to or f"{config.frontend_url}/settings/integrations"
            connected_param = integration_type or oauth_provider
            return RedirectResponse(url=f"{final_redirect}?connected={connected_param}")
    return None


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


async def _handle_discord_post_oauth(
    connection,
    state_data: dict,
    user_id: str,
    oauth_provider: str,
    request: Request,
) -> None:
    """Handle Discord-specific post-OAuth processing (permissions & guild storage)."""
    from seer.tools.discord.permissions import calculate_permissions, get_permission_names

    # Calculate and store permissions from requested tools
    requested_scope = state_data.get('requested_scope', '')
    tool_names = requested_scope.split() if requested_scope else []
    calculated_permissions = calculate_permissions(tool_names)

    # Update connection metadata with permissions
    if connection:
        metadata = connection.provider_metadata or {}

        # Merge permissions (bitwise OR) with existing permissions
        existing_perms = metadata.get("permissions", 0)
        merged_perms = existing_perms | calculated_permissions

        metadata["permissions"] = merged_perms
        metadata["permission_names"] = list(get_permission_names(merged_perms))

        connection.provider_metadata = metadata
        await connection.save()

        logger.info(
            "Stored Discord permissions: existing=%s, calculated=%s, merged=%s, names=%s",
            existing_perms,
            calculated_permissions,
            merged_perms,
            metadata["permission_names"]
        )

    # Store guild information
    guild_id = request.query_params.get('guild_id')
    if guild_id and config.discord_bot_token:
        provider_impl = get_integration_provider(oauth_provider)
        if provider_impl:
            # Type check for DiscordProvider
            from seer.services.integrations.providers.discord import DiscordProvider
            if isinstance(provider_impl, DiscordProvider):
                try:
                    guild_info = await provider_impl.fetch_guild_info(
                        guild_id=guild_id,
                        bot_token=config.discord_bot_token
                    )

                    # Store guild as IntegrationResource
                    from seer.api.integrations.services import _upsert_integration_resource
                    user = await User.get(user_id=user_id)
                    await _upsert_integration_resource(
                        user=user,
                        oauth_connection=connection,
                        provider="discord",
                        resource_type="guild",
                        resource_id=guild_id,
                        resource_key=None,
                        name=guild_info.get("name", f"Discord Server {guild_id}"),
                        metadata={
                            "guild_id": guild_id,
                            "guild_name": guild_info.get("name"),
                            "icon": guild_info.get("icon"),
                            "owner_id": guild_info.get("owner_id"),
                        }
                    )
                    logger.info(
                        "Stored Discord guild: guild_id=%s, name=%s",
                        guild_id,
                        guild_info.get("name")
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to store Discord guild: guild_id=%s, error=%s",
                        guild_id,
                        exc,
                        exc_info=True
                    )
                    # Don't fail the OAuth flow if guild storage fails
                    # The connection is still stored, user can retry later


async def _handle_slack_post_oauth(
    connection,
    token: dict,
    user_id: str,
) -> None:
    """Handle Slack-specific post-OAuth processing (workspace storage)."""
    # Slack token response includes team info directly
    team_info = token.get("team", {})
    workspace_id = team_info.get("id")

    if workspace_id:
        try:
            # Store workspace as IntegrationResource
            from seer.api.integrations.services import _upsert_integration_resource
            user = await User.get(user_id=user_id)
            await _upsert_integration_resource(
                user=user,
                oauth_connection=connection,
                provider="slack",
                resource_type="workspace",
                resource_id=workspace_id,
                resource_key=None,
                name=team_info.get("name", f"Slack Workspace {workspace_id}"),
                metadata={
                    "workspace_id": workspace_id,
                    "workspace_name": team_info.get("name"),
                    "bot_user_id": token.get("bot_user_id"),
                }
            )
            logger.info(
                "Stored Slack workspace: workspace_id=%s, name=%s",
                workspace_id,
                team_info.get("name")
            )
        except Exception as exc:
            logger.error(
                "Failed to store Slack workspace: workspace_id=%s, error=%s",
                workspace_id,
                exc,
                exc_info=True
            )
            # Don't fail the OAuth flow if workspace storage fails


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


@router.get("/connections", response_model=ConnectionsByProvider)
async def list_connections_grouped(request: Request):
    """
    List all connections grouped by OAuth provider.

    Returns connections organized by provider, useful for multi-account scenarios
    where users need to see all their connected accounts per provider.

    Example response:
    {
        "connections": {
            "google": [
                {"id": 123, "provider_account_id": "alice@gmail.com", ...},
                {"id": 124, "provider_account_id": "bob@gmail.com", ...}
            ],
            "github": [
                {"id": 125, "provider_account_id": "alice", ...}
            ]
        }
    }
    """
    user: User = request.state.db_user
    connections = await list_connections(user)

    grouped: dict = {}
    for conn in connections:
        provider = conn.provider
        if provider not in grouped:
            grouped[provider] = []
        grouped[provider].append(AccountInfo(
            id=conn.id,
            provider=conn.provider,
            provider_account_id=conn.provider_account_id or f"ID:{conn.id}",
            display_name=get_connection_display_name(conn),
            status=conn.status,
            scopes=conn.scopes,
        ))

    return ConnectionsByProvider(connections=grouped)


@router.get("/tools/{tool_name}/accounts", response_model=ToolAccountsResponse)
async def get_tool_accounts(request: Request, tool_name: str):
    """
    Get available OAuth accounts for a specific tool.

    Returns all user accounts that can be used with this tool, including
    scope validation status for each account.

    Used by frontend workflow builder to let users select which account
    to use when they have multiple accounts for the same provider.

    Args:
        tool_name: Name of the tool (e.g., "google.gmail.send_email")

    Returns:
        ToolAccountsResponse with list of accounts and whether selection is required
    """
    from seer.tools.base import get_tool  # pylint: disable=import-outside-toplevel

    user: User = request.state.db_user

    tool = get_tool(tool_name)
    if tool is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Tool not found",
            detail=f"Tool '{tool_name}' not found",
            status=404
        )

    # If tool doesn't require OAuth, return empty response
    if not tool.required_scopes:
        return ToolAccountsResponse(
            tool_name=tool_name,
            provider=tool.provider,
            accounts=[],
            requires_selection=False,
        )

    provider = tool.provider
    if not provider:
        return ToolAccountsResponse(
            tool_name=tool_name,
            provider=None,
            accounts=[],
            requires_selection=False,
        )

    # Get all connections for this provider
    connections = await list_connections(user)
    provider_connections = [c for c in connections if c.provider == provider]

    accounts = []
    for conn in provider_connections:
        # Check scope coverage
        missing = []
        if conn.scopes:
            has_scopes = has_required_scopes(conn.scopes, tool.required_scopes)
            if not has_scopes:
                # Find which scopes are missing
                granted_set = parse_scopes(conn.scopes)
                for scope in tool.required_scopes:
                    if scope not in granted_set:
                        missing.append(scope)
        else:
            has_scopes = False
            missing = list(tool.required_scopes)

        accounts.append(ToolAccountInfo(
            id=conn.id,
            provider_account_id=conn.provider_account_id or f"ID:{conn.id}",
            display_name=get_connection_display_name(conn),
            has_required_scopes=has_scopes,
            missing_scopes=missing,
        ))

    return ToolAccountsResponse(
        tool_name=tool_name,
        provider=provider,
        accounts=accounts,
        requires_selection=len(accounts) > 1,
    )


@router.get("/metadata", response_model=IntegrationMetadataResponse)
async def get_integration_metadata():
    """
    Get metadata for all available integrations.

    Returns integration configurations including display names, icons, OAuth providers,
    available scopes, and detection patterns. This allows the frontend to render
    integrations dynamically without hardcoded configurations.

    The response includes:
    - integrations: List of all integration types with their metadata
    - provider_to_types: Mapping from OAuth providers to integration types

    This endpoint does not require authentication as the data is static configuration.
    """
    from seer.services.integrations.metadata import get_all_integration_metadata  # pylint: disable=import-outside-toplevel  # Reason: Lazy import to avoid circular dependency with tools module

    return get_all_integration_metadata()


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
    force_new_account: bool = Query(False),
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

    # Skip early return check when forcing new account (user wants to add another OAuth account)
    if not force_new_account:
        early_return = _check_existing_scopes(
            existing_connection, normalized_scope_list, oauth_provider, redirect_to, integration_type
        )
        if early_return:
            return early_return

    redirect_uri = _build_oauth_redirect_uri(request, oauth_provider)

    logger.info(
        "Starting OAuth flow: provider=%s, integration_type=%s, scopes=%s",
        oauth_provider,
        integration_type,
        scope_string[:100],
    )

    # Clean up old Authlib state keys to prevent session/cookie bloat
    # These accumulate from incomplete OAuth flows and can cause cookie size issues
    if hasattr(request.session, 'keys'):
        old_state_keys = [k for k in list(request.session.keys()) if k.startswith('_state_')]
        for key in old_state_keys:
            request.session.pop(key, None)
        if old_state_keys:
            logger.debug("Cleaned up %d old OAuth state keys", len(old_state_keys))

    # Store custom OAuth data in session (Authlib handles state/PKCE automatically)
    # This data will be retrieved in the callback to complete the connection
    oauth_data = {
        'user_id': user.user_id,
        'user_email': user.email,
        'redirect_to': redirect_to or f"{config.frontend_url}/settings/integrations",
        'oauth_provider': oauth_provider,
        'integration_type': integration_type or oauth_provider,
        'requested_scope': scope_string,
    }
    request.session['oauth_data'] = oauth_data

    logger.debug(
        "Stored oauth_data in session: user_id=%s, session_keys=%s",
        user.user_id,
        list(request.session.keys()) if hasattr(request.session, 'keys') else 'N/A',
    )

    client = oauth.create_client(oauth_provider)

    # Get provider-specific authorize kwargs (e.g., incremental auth for Google)
    authorize_kwargs = provider_impl.build_authorize_kwargs(
        authorize_context, state="", scope=scope_string  # state is handled by Authlib
    )
    # Remove state from kwargs - Authlib will generate its own
    authorize_kwargs.pop("state", None)
    authorize_kwargs.setdefault("scope", scope_string)

    # Authlib handles state generation, PKCE (if configured), and redirect
    return await client.authorize_redirect(request, redirect_uri, **authorize_kwargs)


@router.get("/{provider}/callback", name="auth_callback")
async def auth_callback(request: Request, provider: str):
    """
    Handle OAuth callback from provider.

    Uses Authlib's authorize_access_token() which handles:
    - State validation (CSRF protection)
    - PKCE code_verifier submission (if configured)
    - Token exchange with proper auth method

    Stores connection with OAuth provider (e.g., 'google'), merging scopes
    if a connection already exists for this provider.
    """
    oauth_provider = get_oauth_provider(provider)

    # Debug: log session and cookie state
    session_keys = list(request.session.keys()) if hasattr(request.session, 'keys') else 'N/A'
    session_cookie = request.cookies.get('session', 'NOT_FOUND')
    logger.info(
        "OAuth callback: provider=%s, session_keys=%s, session_cookie_present=%s",
        oauth_provider,
        session_keys,
        session_cookie != 'NOT_FOUND',
    )

    # Retrieve custom OAuth data stored in session during /connect
    oauth_data = request.session.get('oauth_data', {})
    user_id = oauth_data.get('user_id')
    redirect_to = oauth_data.get('redirect_to', f"{config.frontend_url}/settings/integrations")
    integration_type = oauth_data.get('integration_type')

    if not user_id:
        # Session expired or not found - user needs to restart OAuth flow
        logger.warning(
            "OAuth callback missing session data for provider=%s, session_keys=%s",
            oauth_provider,
            session_keys,
        )
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Session expired",
            detail="OAuth session expired. Please try connecting again.",
            status=400
        )

    logger.info(
        "OAuth callback received: provider=%s, integration_type=%s",
        oauth_provider,
        integration_type,
    )

    # Check for OAuth errors from provider
    error = request.query_params.get('error')
    if error:
        error_description = request.query_params.get('error_description', 'Authorization denied')
        logger.error("OAuth callback error: error=%s, description=%s", error, error_description)
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth authorization error",
            detail=f"{error}: {error_description}",
            status=400
        )

    client = oauth.create_client(oauth_provider)

    # Authlib handles state validation, PKCE verification, and token exchange
    try:
        token = await client.authorize_access_token(request)
        logger.info("OAuth token exchange successful: provider=%s", oauth_provider)
    except OAuthError as exc:
        logger.error(
            "OAuth token exchange failed: provider=%s, error=%s",
            oauth_provider,
            str(exc),
        )
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="OAuth token exchange error",
            detail=str(exc),
            status=400,
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

    # Use oauth_data as state_data for provider methods (backward compatible)
    state_data = oauth_data

    granted_scopes = await provider_impl.resolve_granted_scopes(
        token=token, state_data=state_data
    )
    _log_scope_info(token, granted_scopes, state_data.get('requested_scope'))

    user_info = await provider_impl.fetch_user_profile(
        client=client, token=token, state_data=state_data
    )

    # For Discord, we need to handle the profile differently since bot installations
    # don't return a traditional user profile. Use user_id as provider_account_id.
    if oauth_provider == "discord":
        # Discord bot installations don't have user profiles in the token
        # Use a synthetic profile with the user_id as the id
        user_info = {"id": user_id, "type": "bot_installation"}

    connection = await store_oauth_connection(
        user_id=user_id,
        provider=oauth_provider,
        token=token,
        profile=user_info,
        granted_scopes=granted_scopes,
        integration_type=integration_type
    )

    # Provider-specific post-OAuth processing
    if oauth_provider == "discord":
        await _handle_discord_post_oauth(connection, state_data, user_id, oauth_provider, request)

    if oauth_provider == "slack":
        await _handle_slack_post_oauth(connection, token, user_id)

    # Clear OAuth session data after successful connection
    request.session.pop('oauth_data', None)

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


# =============================================================================
# DISCORD-SPECIFIC ENDPOINTS
# =============================================================================

@router.get("/discord/guilds")
async def list_discord_guilds(request: Request):
    """
    List Discord servers (guilds) where the bot is installed for the current user.

    Returns guild information from IntegrationResource records.
    """
    user: User = request.state.db_user
    resources = await list_integration_resources(
        user,
        provider="discord",
        resource_type="guild",
    )
    return {"guilds": [serialize_integration_resource(r) for r in resources]}


@router.post("/discord/guilds/{guild_id}/disconnect")
async def disconnect_discord_guild(guild_id: str, request: Request):
    """
    Disconnect bot from a specific Discord server.
    Marks the IntegrationResource as inactive.
    """
    user: User = request.state.db_user
    # Find and deactivate the specific guild resource
    from seer.database import IntegrationResource
    resource = await IntegrationResource.get_or_none(
        user=user,
        provider="discord",
        resource_type="guild",
        resource_id=guild_id,
        status="active"
    )
    if not resource:
        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Guild not found",
            detail=f"Discord guild {guild_id} not found or already disconnected",
            status=404
        )
    resource.status = "inactive"
    await resource.save()
    return {"status": "success", "guild_id": guild_id}


router.include_router(resource_router)
