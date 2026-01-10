"""
Service for building tool connection status.

Handles the complex logic of determining tool authentication requirements,
connection status, and token validity.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.database import OAuthConnection
from shared.logger import get_logger

from .services import has_required_scopes, parse_scopes

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
        has_access_token = bool(conn.access_token_enc)
        has_refresh_token = bool(conn.refresh_token_enc)
        is_token_expired = False
        if conn.expires_at:
            is_token_expired = conn.expires_at < datetime.now(timezone.utc)

        access_token_valid = (has_access_token and not is_token_expired) or has_refresh_token

        provider_connections[conn.provider] = {
            "scopes": conn.scopes or "",
            "connection_id": f"{conn.provider}:{conn.id}",
            "provider_account_id": conn.provider_account_id,
            "has_refresh_token": has_refresh_token,
            "access_token_valid": access_token_valid,
            "connection": conn
        }
    return provider_connections


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
    }


def build_base_tool_status(tool: Any, auth_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build base tool status dict with common fields.

    Args:
        tool: Tool object
        auth_requirements: Auth requirements from determine_tool_auth_requirements

    Returns:
        Base dict with tool identification and auth requirements
    """
    return {
        "tool_name": tool.name,
        "integration_type": tool.integration_type,
        "requires_oauth_connection": auth_requirements["requires_oauth"],
        "requires_secrets": auth_requirements["requires_secrets"],
        "supports_tokenless_auth": auth_requirements["supports_tokenless_auth"],
        "auth_mode": auth_requirements["auth_mode"],
    }


def build_tool_status_for_non_oauth_tool(
    tool: Any,
    auth_requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build status for tools that don't require OAuth.

    Args:
        tool: Tool object
        auth_requirements: Auth requirements

    Returns:
        Complete status dict for non-OAuth tool
    """
    status = build_base_tool_status(tool, auth_requirements)
    status.update({
        "provider": None,
        "connected": True,
        "has_required_scopes": True,
        "access_token_valid": True,
        "missing_scopes": [],
        "connection_id": None,
        "provider_account_id": None,
        "has_refresh_token": False,
    })
    return status


def build_tool_status_for_optional_oauth(
    tool: Any,
    auth_requirements: Dict[str, Any],
    oauth_provider: Optional[str],
    conn_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build status for tools where OAuth is optional.

    Args:
        tool: Tool object
        auth_requirements: Auth requirements
        oauth_provider: Normalized OAuth provider name
        conn_info: Connection info from provider map (may be None)

    Returns:
        Complete status dict for optional OAuth tool
    """
    status = build_base_tool_status(tool, auth_requirements)
    status.update({
        "provider": oauth_provider,
        "connected": True,
        "has_required_scopes": True,
        "access_token_valid": True,
        "missing_scopes": [],
        "connection_id": conn_info["connection_id"] if conn_info else None,
        "provider_account_id": conn_info["provider_account_id"] if conn_info else None,
        "has_refresh_token": conn_info["has_refresh_token"] if conn_info else False,
    })
    return status


def build_tool_status_for_missing_connection(
    tool: Any,
    auth_requirements: Dict[str, Any],
    oauth_provider: Optional[str]
) -> Dict[str, Any]:
    """
    Build status for tools requiring OAuth but without a connection.

    Args:
        tool: Tool object
        auth_requirements: Auth requirements
        oauth_provider: Normalized OAuth provider name

    Returns:
        Complete status dict showing disconnected state
    """
    status = build_base_tool_status(tool, auth_requirements)
    status.update({
        "provider": oauth_provider,
        "connected": False,
        "has_required_scopes": False,
        "access_token_valid": False,
        "missing_scopes": auth_requirements["required_scopes"],
        "connection_id": None,
        "provider_account_id": None,
        "has_refresh_token": False,
    })
    return status


def build_tool_status_for_oauth_tool(
    tool: Any,
    auth_requirements: Dict[str, Any],
    oauth_provider: Optional[str],
    conn_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build status for tools with OAuth requirement and active connection.

    Args:
        tool: Tool object
        auth_requirements: Auth requirements
        oauth_provider: Normalized OAuth provider name
        conn_info: Connection info from provider map

    Returns:
        Complete status dict with scope and token validation
    """
    required_scopes = auth_requirements["required_scopes"]

    has_scopes = has_required_scopes(conn_info["scopes"], required_scopes)
    access_token_valid = conn_info.get("access_token_valid", False)
    has_refresh_token = conn_info.get("has_refresh_token", False)
    fully_connected = has_scopes and access_token_valid

    granted_set = parse_scopes(conn_info["scopes"]) if conn_info["scopes"] else set()
    missing = [s for s in required_scopes if s not in granted_set]

    status = build_base_tool_status(tool, auth_requirements)
    status.update({
        "provider": oauth_provider,
        "connected": fully_connected,
        "has_required_scopes": has_scopes,
        "access_token_valid": access_token_valid,
        "has_refresh_token": has_refresh_token,
        "missing_scopes": missing,
        "connection_id": conn_info["connection_id"],
        "provider_account_id": conn_info["provider_account_id"],
    })
    return status
