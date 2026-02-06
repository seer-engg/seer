"""
Tool executor for running registered tools with unified credential resolution.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.database import User
from seer.logger import get_logger
from seer.tools.base import get_tool
from seer.tools.coercion import coerce_arguments
from seer.tools.credential_resolver import CredentialResolver, ResolvedCredentials

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

    # Schema-driven coercion: parse LLM outputs based on parameter types
    schema = tool.get_parameters_schema()
    arguments = coerce_arguments(arguments, schema)

    resolver = CredentialResolver(user=user, tool=tool, connection_id=connection_id)
    try:
        resolved = await resolver.resolve(arguments or {})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Credential resolution failed", extra={"tool": tool_name})
        raise HTTPException(status_code=500, detail=f"Credential resolution failed: {str(exc)}") from exc

    try:
        logger.info("Executing tool '%s' for user {user.user_id}", tool_name)
        result = await _execute_with_optional_credentials(tool, resolved, arguments or {})
        logger.info("Tool '%s' executed successfully", tool_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Tool execution failed: %s", e)

        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        ) from e


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
