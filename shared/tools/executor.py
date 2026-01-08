"""
Tool executor for running registered tools with unified credential resolution.
"""

import time
from typing import Any, Dict, Optional

from fastapi import HTTPException

from shared.database.models import User
from shared.logger import get_logger
from shared.tools.base import get_tool
from shared.tools.credential_resolver import CredentialResolver, ResolvedCredentials
from shared.analytics import analytics
from shared.config import config

logger = get_logger("shared.tools.executor")


async def execute_tool(
    tool_name: str,
    user: User,
    connection_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Execute a tool with OAuth token management.

    Args:
        tool_name: Name of the tool to execute
        user: User
        connection_id: OAuth connection ID (if tool requires OAuth)
        arguments: Tool arguments

    Returns:
        Tool execution result

    Raises:
        HTTPException: If tool not found, scopes invalid, or execution fails
    """
    arguments = arguments or {}

    tool = get_tool(tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found"
        )

    resolver = CredentialResolver(user=user, tool=tool, connection_id=connection_id)
    try:
        resolved = await resolver.resolve(arguments or {})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Credential resolution failed", extra={"tool": tool_name})
        raise HTTPException(status_code=500, detail=f"Credential resolution failed: {str(exc)}") from exc

    # Track tool execution timing and status
    start_time = time.time()
    success = True
    error_type = None

    try:
        logger.info(f"Executing tool '{tool_name}' for user {user.user_id}")
        result = await _execute_with_optional_credentials(tool, resolved, arguments or {})
        logger.info(f"Tool '{tool_name}' executed successfully")
        return result
    except HTTPException:
        success = False
        error_type = "HTTPException"
        raise
    except Exception as e:
        success = False
        error_type = type(e).__name__
        logger.exception(f"Tool execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )
    finally:
        # Always capture tool execution event
        duration_ms = (time.time() - start_time) * 1000
        analytics.capture(
            distinct_id=user.user_id,
            event="tool_executed",
            properties={
                "tool_name": tool_name,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error_type": error_type,
                "deployment_mode": config.seer_mode,
            },
        )


async def _execute_with_optional_credentials(
    tool,
    resolved: ResolvedCredentials,
    arguments: Dict[str, Any],
) -> Any:
    try:
        return await tool.execute(resolved.access_token, arguments, credentials=resolved)
    except TypeError as exc:
        # Backwards compatibility for tools that haven't added the credentials kwarg yet
        message = str(exc)
        if "credentials" in message and "unexpected keyword argument" in message:
            return await tool.execute(resolved.access_token, arguments)
        raise
