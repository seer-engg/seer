"""
Tool services for listing and executing tools.
"""
from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.database import User
from seer.logger import get_logger
from seer.tools.executor import execute_tool as _execute_tool
from seer.tools.registry import get_tools_by_integration

logger = get_logger("api.tools.services")


async def list_tools(integration_type: Optional[str] = None) -> Dict[str, Any]:
    """
    List available tools.

    Args:
        integration_type: Optional filter by integration type (e.g., 'gmail', 'github')

    Returns:
        Dict with 'tools' list containing tool metadata (parameters + output schema)
    """
    try:
        tools = get_tools_by_integration(integration_type)
        logger.info("Listing tools: integration_type=%s, count={len(tools)}", integration_type)
        return {"tools": tools}
    except Exception as e:
        logger.exception("Error listing tools: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tools: {str(e)}"
        )


async def execute_tool_service(
    tool_name: str,
    user: User,
    connection_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a tool.

    Args:
        tool_name: Name of the tool to execute
        user: User
        connection_id: OAuthConnection ID (required for OAuth tools)
        arguments: Tool arguments

    Returns:
        Dict with 'data' (result) and 'success' (bool)
    """
    try:
        logger.info(
            "Executing tool: tool_name=%s, user_id=%s, connection_id=%s, has_arguments=%s",
            tool_name, user.user_id, connection_id, arguments is not None
        )

        result = await _execute_tool(
            tool_name=tool_name,
            user=user,
            connection_id=connection_id,
            arguments=arguments or {}
        )

        return {
            "data": result,
            "success": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error executing tool %s: {e}", tool_name)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing tool: {str(e)}"
        )
