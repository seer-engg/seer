"""
Bootstrap service for parallel data fetching.

Consolidates multiple API calls into a single endpoint with parallel processing.
"""
# pylint: disable=import-outside-toplevel # Intentional: avoids circular imports and improves startup time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock

from fastapi import Request

from api.models.router import list_models

# Import existing service functions
from api.tools.services import list_tools
from api.workflows.services import list_node_types, list_workflows
from shared.database.models import User
from shared.logger import get_logger

logger = get_logger("api.bootstrap.services")


def _create_mock_request(user: User) -> Request:
    """Create mock request for bootstrap context (no HTTP request available)."""
    mock_request = MagicMock(spec=Request)
    mock_request.state.db_user = user
    return mock_request


async def _fetch_tools() -> Dict[str, Any]:
    """Fetch tools list. Returns empty list on error."""
    from starlette.exceptions import HTTPException

    try:
        result = await list_tools()
        return result.get("tools", [])
    except (HTTPException, asyncio.TimeoutError, ValueError) as e:
        logger.error("Error fetching tools: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching tools: %s", e)
        raise


async def _fetch_models() -> list:
    """Fetch models list. Returns empty list on error."""
    try:
        models = await list_models()
        # Convert Pydantic models to dicts
        return [model.model_dump() for model in models]
    except (ValueError, TypeError, asyncio.TimeoutError) as e:
        logger.error("Error fetching models: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching models: %s", e)
        raise


async def _fetch_tools_status(user: User) -> list:
    """Fetch tools connection status. Returns empty list on error."""
    from api.integrations.router import get_tools_connection_status
    from starlette.exceptions import HTTPException

    mock_request = _create_mock_request(user)

    try:
        result = await get_tools_connection_status(mock_request)
        return result.get("tools", [])
    except (HTTPException, asyncio.TimeoutError, RuntimeError) as e:
        logger.error("Error fetching tools status: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching tools status: %s", e)
        raise


async def _fetch_connections(user: User) -> list:
    """Fetch user connections. Returns empty list on error."""
    from api.integrations.router import list_integrations
    from starlette.exceptions import HTTPException

    mock_request = _create_mock_request(user)

    try:
        result = await list_integrations(mock_request)
        return result.get("items", [])
    except (HTTPException, asyncio.TimeoutError, ValueError) as e:
        logger.error("Error fetching connections: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching connections: %s", e)
        raise


async def _fetch_node_types() -> Dict[str, Any]:
    """Fetch workflow node types. Returns empty dict on error."""
    try:
        result = await list_node_types()
        # Convert Pydantic model to dict
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result if isinstance(result, dict) else {}
    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error fetching node types: %s", e)
        return {}
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching node types: %s", e)
        raise


async def _fetch_workflows(user: User) -> Dict[str, Any]:
    """Fetch user workflows. Returns empty result on error."""
    from starlette.exceptions import HTTPException

    try:
        result = await list_workflows(user, limit=50)
        # Convert Pydantic model to dict
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result if isinstance(result, dict) else {"items": [], "next_cursor": None}
    except (HTTPException, asyncio.TimeoutError, ValueError) as e:
        logger.error("Error fetching workflows: %s", e)
        return {"items": [], "next_cursor": None}
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching workflows: %s", e)
        raise


async def _fetch_connections_raw(user: User) -> list:
    """Fetch raw connections from database. Used internally to avoid duplicate queries."""
    try:
        from api.integrations.services import list_connections
        return await list_connections(user)
    except (asyncio.TimeoutError, ValueError) as e:
        logger.error("Error fetching raw connections: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error fetching raw connections: %s", e)
        raise


def _build_provider_connections_map(connections: list) -> dict:
    """Build a map of provider -> connection info."""
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
        }
    return provider_connections


def _build_tool_status_result(
    tool, conn_info, required_scopes, has_required_scopes_fn, parse_scopes_fn
):
    """Build status result for a tool with connection info."""
    requires_oauth = bool(required_scopes)
    requires_secrets = bool(getattr(tool, "required_secrets", []))
    supports_tokenless_auth = not requires_oauth

    auth_mode = "none"
    if requires_oauth and requires_secrets:
        auth_mode = "oauth_and_secrets"
    elif requires_oauth:
        auth_mode = "oauth"
    elif requires_secrets:
        auth_mode = "secrets"

    base_result = {
        "tool_name": tool.name,
        "integration_type": tool.integration_type,
        "requires_oauth_connection": requires_oauth,
        "requires_secrets": requires_secrets,
        "supports_tokenless_auth": supports_tokenless_auth,
        "auth_mode": auth_mode,
    }

    if not conn_info:
        return {
            **base_result,
            "connected": False,
            "has_required_scopes": False,
            "access_token_valid": False,
            "missing_scopes": required_scopes,
            "connection_id": None,
            "provider_account_id": None,
            "has_refresh_token": False,
        }

    has_scopes = has_required_scopes_fn(conn_info["scopes"], required_scopes)
    access_token_valid = conn_info.get("access_token_valid", False)
    has_refresh_token = conn_info.get("has_refresh_token", False)
    fully_connected = has_scopes and access_token_valid

    granted_set = parse_scopes_fn(conn_info["scopes"]) if conn_info["scopes"] else set()
    missing = [s for s in required_scopes if s not in granted_set]

    return {
        **base_result,
        "connected": fully_connected,
        "has_required_scopes": has_scopes,
        "access_token_valid": access_token_valid,
        "has_refresh_token": has_refresh_token,
        "missing_scopes": missing,
        "connection_id": conn_info["connection_id"],
        "provider_account_id": conn_info["provider_account_id"],
    }


async def _build_tools_status_from_connections(connections: list) -> list:
    """Build tools status using pre-fetched connections to avoid duplicate DB query."""
    try:
        from api.integrations.services import (
            get_oauth_provider,
            has_required_scopes,
            parse_scopes,
        )
        from shared.tools.base import list_tools as get_all_tools

        provider_connections = _build_provider_connections_map(connections)
        all_tools = get_all_tools()

        results = []
        for tool in all_tools:
            tool_provider = tool.provider or tool.integration_type
            required_scopes = list(tool.required_scopes or [])
            requires_oauth = bool(required_scopes)

            if not tool_provider:
                # Non-OAuth tool
                results.append({
                    "tool_name": tool.name,
                    "integration_type": tool.integration_type,
                    "requires_oauth_connection": False,
                    "requires_secrets": bool(getattr(tool, "required_secrets", [])),
                    "supports_tokenless_auth": True,
                    "auth_mode": "secrets" if getattr(tool, "required_secrets", []) else "none",
                    "provider": None,
                    "connected": True,
                    "has_required_scopes": True,
                    "access_token_valid": True,
                    "missing_scopes": [],
                    "connection_id": None,
                    "provider_account_id": None,
                    "has_refresh_token": False,
                })
                continue

            oauth_provider = get_oauth_provider(tool_provider)
            conn_info = provider_connections.get(oauth_provider) if oauth_provider else None

            if not requires_oauth:
                # Tokens are optional
                results.append({
                    "tool_name": tool.name,
                    "integration_type": tool.integration_type,
                    "requires_oauth_connection": False,
                    "requires_secrets": bool(getattr(tool, "required_secrets", [])),
                    "supports_tokenless_auth": True,
                    "auth_mode": "secrets" if getattr(tool, "required_secrets", []) else "none",
                    "provider": oauth_provider,
                    "connected": True,
                    "has_required_scopes": True,
                    "access_token_valid": True,
                    "missing_scopes": [],
                    "connection_id": conn_info["connection_id"] if conn_info else None,
                    "provider_account_id": conn_info["provider_account_id"] if conn_info else None,
                    "has_refresh_token": conn_info["has_refresh_token"] if conn_info else False,
                })
                continue

            result = _build_tool_status_result(
                tool, conn_info, required_scopes, has_required_scopes, parse_scopes
            )
            results.append({**result, "provider": oauth_provider})

        return results
    except (AttributeError, KeyError, ValueError) as e:
        logger.error("Error building tools status: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error building tools status: %s", e)
        raise


async def _format_connections(connections: list, user: User) -> list:
    """Format raw connections into the expected API format."""
    try:
        res = []
        for conn in connections:
            composite_id = f"{conn.provider}:{conn.id}"
            res.append({
                "id": composite_id,
                "status": "ACTIVE" if conn.status == 'active' else "INACTIVE",
                "user_id": user.user_id,
                "toolkit": {
                    "slug": conn.provider
                },
                "connection": {
                    "user_id": user.user_id,
                    "provider_account_id": conn.provider_account_id
                },
                "scopes": conn.scopes or "",
                "provider": conn.provider
            })
        return res
    except (AttributeError, TypeError) as e:
        logger.error("Error formatting connections: %s", e)
        return []
    except Exception as e:
        # Unexpected error - this should not happen
        logger.exception("Unexpected error formatting connections: %s", e)
        raise


async def fetch_bootstrap_data(user: User) -> Dict[str, Any]:
    """
    Fetch all bootstrap data in parallel with optimized database queries.

    Uses asyncio.gather to run all data fetches concurrently.
    Response time equals the slowest query, not the sum of all queries.

    Optimizations:
    - Fetches connections once and reuses for both tools_status and connections endpoints
    - Eliminates duplicate database queries

    Args:
        user: The authenticated user

    Returns:
        Dict containing all bootstrap data with empty arrays/dicts for failed sections
    """
    # Run all fetches in parallel (connections_raw is fetched once and shared)
    results = await asyncio.gather(
        _fetch_tools(),
        _fetch_models(),
        _fetch_connections_raw(user),  # Fetch once for both tools_status and connections
        _fetch_node_types(),
        _fetch_workflows(user),
        return_exceptions=True  # Don't fail entire request if one section fails
    )

    # Unpack results with error handling
    tools = results[0] if not isinstance(results[0], Exception) else []
    models = results[1] if not isinstance(results[1], Exception) else []
    raw_connections = results[2] if not isinstance(results[2], Exception) else []
    node_types = results[3] if not isinstance(results[3], Exception) else {}
    workflows = (
        results[4] if not isinstance(results[4], Exception)
        else {"items": [], "next_cursor": None}
    )

    # Build tools_status and format connections using the single connections query result
    tools_status_task = _build_tools_status_from_connections(raw_connections)
    connections_task = _format_connections(raw_connections, user)

    tools_status, connections = await asyncio.gather(
        tools_status_task,
        connections_task,
        return_exceptions=True
    )

    # Handle errors
    if isinstance(tools_status, Exception):
        logger.error("Error in tools_status: %s", tools_status)
        tools_status = []
    if isinstance(connections, Exception):
        logger.error("Error in connections: %s", connections)
        connections = []

    return {
        "tools": tools,
        "models": models,
        "tools_status": tools_status,
        "connections": connections,
        "node_types": node_types,
        "workflows": workflows,
        "cached": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
