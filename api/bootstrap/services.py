"""
Bootstrap service for parallel data fetching.

Consolidates multiple API calls into a single endpoint with parallel processing.
"""
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone

from shared.logger import get_logger
from shared.database.models import User

# Import existing service functions
from api.tools.services import list_tools
from api.models.router import list_models
from api.workflows.services import list_node_types, list_workflows

logger = get_logger("api.bootstrap.services")


async def _fetch_tools() -> Dict[str, Any]:
    """Fetch tools list. Returns empty list on error."""
    try:
        result = await list_tools()
        return result.get("tools", [])
    except Exception as e:
        logger.error(f"Error fetching tools: {e}", exc_info=True)
        return []


async def _fetch_models() -> list:
    """Fetch models list. Returns empty list on error."""
    try:
        models = await list_models()
        # Convert Pydantic models to dicts
        return [model.model_dump() for model in models]
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        return []


async def _fetch_tools_status(user: User) -> list:
    """Fetch tools connection status. Returns empty list on error."""
    try:
        from api.integrations.router import get_tools_connection_status
        from fastapi import Request
        from unittest.mock import MagicMock

        # Create a mock request with the user
        mock_request = MagicMock(spec=Request)
        mock_request.state.db_user = user

        result = await get_tools_connection_status(mock_request)
        return result.get("tools", [])
    except Exception as e:
        logger.error(f"Error fetching tools status: {e}", exc_info=True)
        return []


async def _fetch_connections(user: User) -> list:
    """Fetch user connections. Returns empty list on error."""
    try:
        from api.integrations.router import list_integrations
        from fastapi import Request
        from unittest.mock import MagicMock

        # Create a mock request with the user
        mock_request = MagicMock(spec=Request)
        mock_request.state.db_user = user

        result = await list_integrations(mock_request)
        return result.get("items", [])
    except Exception as e:
        logger.error(f"Error fetching connections: {e}", exc_info=True)
        return []


async def _fetch_node_types() -> Dict[str, Any]:
    """Fetch workflow node types. Returns empty dict on error."""
    try:
        result = await list_node_types()
        # Convert Pydantic model to dict
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.error(f"Error fetching node types: {e}", exc_info=True)
        return {}


async def _fetch_workflows(user: User) -> Dict[str, Any]:
    """Fetch user workflows. Returns empty result on error."""
    try:
        result = await list_workflows(user, limit=50)
        # Convert Pydantic model to dict
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result if isinstance(result, dict) else {"items": [], "next_cursor": None}
    except Exception as e:
        logger.error(f"Error fetching workflows: {e}", exc_info=True)
        return {"items": [], "next_cursor": None}


async def fetch_bootstrap_data(user: User) -> Dict[str, Any]:
    """
    Fetch all bootstrap data in parallel.

    Uses asyncio.gather to run all data fetches concurrently.
    Response time equals the slowest query, not the sum of all queries.

    Args:
        user: The authenticated user

    Returns:
        Dict containing all bootstrap data with empty arrays/dicts for failed sections
    """
    logger.info(f"Fetching bootstrap data for user {user.user_id}")

    start_time = datetime.now(timezone.utc)

    # Run all fetches in parallel
    results = await asyncio.gather(
        _fetch_tools(),
        _fetch_models(),
        _fetch_tools_status(user),
        _fetch_connections(user),
        _fetch_node_types(),
        _fetch_workflows(user),
        return_exceptions=True  # Don't fail entire request if one section fails
    )

    # Unpack results with error handling
    tools = results[0] if not isinstance(results[0], Exception) else []
    models = results[1] if not isinstance(results[1], Exception) else []
    tools_status = results[2] if not isinstance(results[2], Exception) else []
    connections = results[3] if not isinstance(results[3], Exception) else []
    node_types = results[4] if not isinstance(results[4], Exception) else {}
    workflows = results[5] if not isinstance(results[5], Exception) else {"items": [], "next_cursor": None}

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Bootstrap data fetched in {elapsed:.2f}s: "
        f"tools={len(tools)}, models={len(models)}, "
        f"tools_status={len(tools_status)}, connections={len(connections)}, "
        f"workflows={len(workflows.get('items', []))}"
    )

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
