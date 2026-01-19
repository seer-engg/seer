"""
Service for building tool connection status.

Handles the logic of determining tool authentication requirements,
connection status, and token validity.
"""
from typing import Any, Dict, List, Optional, Set

from seer.database import IntegrationSecret, OAuthConnection, User
from seer.logger import get_logger

from seer.services.integrations.auth.helpers import has_required_scopes, list_connections
from seer.services.integrations.auth.oauth import get_oauth_provider

logger = get_logger(__name__)


def build_provider_connections_map(connections: List[OAuthConnection]) -> Dict[str, Dict[str, Any]]:
    """
    Build a map of provider -> connection info with token status.

    Args:
        connections: List of user's OAuth connections

    Returns:
        Dict mapping provider to connection info including scopes and token validity
    """
    provider_connections = {}
    for conn in connections:
        provider_connections[conn.provider] = {
            "scopes": conn.scopes or "",
            "connection_id": f"{conn.provider}:{conn.id}",
            "provider_account_id": conn.provider_account_id,
            "has_refresh_token": bool(conn.refresh_token_enc),
            "connection": conn
        }
    return provider_connections


async def build_provider_secrets_map(user: User) -> Dict[str, Set[str]]:
    """
    Build a map of provider -> available secret names.

    Args:
        user: Current user

    Returns:
        Dict mapping provider to a set of secret names the user has stored
    """
    provider_secrets: Dict[str, Set[str]] = {}
    secrets = await IntegrationSecret.filter(user=user, status="active").values("provider", "name")
    for secret in secrets:
        provider = secret["provider"]
        name = secret["name"]
        provider_secrets.setdefault(provider, set()).add(name)
    return provider_secrets


def determine_tool_auth_requirements(tool: Any) -> Dict[str, Any]:
    """
    Determine authentication requirements for a tool.

    Args:
        tool: Tool object from registry

    Returns:
        Dict with auth requirements and derived properties
    """
    required_scopes = list(tool.required_scopes or [])
    required_secrets = list(getattr(tool, "required_secrets", []) or [])
    requires_oauth = bool(required_scopes)
    requires_secrets = bool(required_secrets)
    supports_tokenless_auth = not requires_oauth

    auth_mode = "none"
    if requires_oauth and requires_secrets:
        auth_mode = "oauth_and_secrets"
    elif requires_oauth:
        auth_mode = "oauth"
    elif requires_secrets:
        auth_mode = "secrets"

    return {
        "required_scopes": required_scopes,
        "required_secrets": required_secrets,
        "requires_oauth": requires_oauth,
        "requires_secrets": requires_secrets,
        "supports_tokenless_auth": supports_tokenless_auth,
        "auth_mode": auth_mode,
        "supports_oauth": True,
        "supports_manual_secrets": requires_secrets,
    }


def build_tool_status(
    tool: Any,
    auth_requirements: Dict[str, Any],
    *,
    provider: Optional[str],
    provider_aliases: Optional[List[str]],
    conn_info: Optional[Dict[str, Any]],
    provider_secrets: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Build the minimal tool status payload for the /tools/status endpoint.

    Connected means:
    - OAuth tools: refresh token present AND required scopes granted.
    - Manual secret tools: all required secrets stored.
    - Both: either condition satisfied.
    """
    required_scopes = auth_requirements["required_scopes"]
    required_secrets = auth_requirements["required_secrets"]
    supports_oauth = auth_requirements["supports_oauth"]
    supports_manual_secrets = auth_requirements["supports_manual_secrets"]

    provider_keys = [p for p in [provider, *(provider_aliases or [])] if p]

    missing_scopes: List[str] = []
    oauth_connected = False
    connection_id = None
    provider_account_id = None

    if supports_oauth:
        granted_scopes = conn_info["scopes"] if conn_info else ""
        has_refresh_token = conn_info["has_refresh_token"] if conn_info else False
        has_scopes = has_required_scopes(granted_scopes, required_scopes) if conn_info else False
        if conn_info:
            missing_scopes = [
                scope for scope in required_scopes
                if not has_required_scopes(granted_scopes, [scope])
            ]
        else:
            missing_scopes = required_scopes
        oauth_connected = bool(conn_info and has_refresh_token and has_scopes)
        if conn_info:
            connection_id = conn_info.get("connection_id")
            provider_account_id = conn_info.get("provider_account_id")

    secrets_connected = False
    if supports_manual_secrets:
        for key in provider_keys:
            secret_names = provider_secrets.get(key, set())
            if all(secret in secret_names for secret in required_secrets):
                secrets_connected = True
                break

    if supports_oauth or supports_manual_secrets:
        connected = oauth_connected or secrets_connected
    else:
        connected = True

    return {
        "tool_name": tool.name,
        "integration_type": tool.integration_type,
        "provider": provider,
        "supports_oauth": supports_oauth,
        "supports_manual_secrets": supports_manual_secrets,
        "connected": connected,
        "missing_scopes": missing_scopes if supports_oauth else [],
        "connection_id": connection_id,
        "provider_account_id": provider_account_id,
    }



async def get_tools_connection_status_for_user(user: User) -> List[Dict[str, Any]]:
    """
    Get the connection status for all tools for a user.
    """
    from seer.tools.base import (
        list_tools as get_all_tools,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with tools.base module
    )

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
    return results
