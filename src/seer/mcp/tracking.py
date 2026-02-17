"""
PostHog tracking for MCP tool calls.

Provides a decorator to track MCP tool invocations with:
- Tool name and parameters (sanitized)
- Execution time
- Success/failure status
- User context from MCP auth

All tracking is non-blocking using schedule_async_task pattern.

Usage:
    from seer.mcp.tracking import track_mcp_tool

    @mcp.tool()
    @track_mcp_tool("list_workflows")
    async def list_workflows(limit: int = 50, cursor: Optional[str] = None) -> str:
        ...
"""
import functools
import time
from typing import Any, Callable

from seer.config import config
from seer.logger import get_logger
from seer.mcp.auth import get_mcp_authenticated_user
from seer.observability.posthog_client import capture_event, identify_user

logger = get_logger(__name__)

# Safe parameter keys to include in tracking (avoid logging sensitive data)
SAFE_PARAM_KEYS = {"workflow_id", "run_id", "limit", "cursor", "query", "section", "integration"}


def track_mcp_tool(tool_name: str):
    """
    Decorator to track MCP tool calls in PostHog.

    Wraps the tool function to capture execution metrics in a non-blocking manner.
    The decorator should be applied after @mcp.tool() to preserve the FastMCP registration.

    Args:
        tool_name: Name of the MCP tool being called (e.g., "list_workflows")

    Example:
        @mcp.tool()
        @track_mcp_tool("list_workflows")
        async def list_workflows(limit: int = 50) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Skip tracking if PostHog is not configured
            if not config.is_posthog_configured:
                return await func(*args, **kwargs)

            start_time = time.perf_counter()
            success = True
            error_message = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                _track_tool_call(tool_name, latency_ms, success, error_message, kwargs)

        return wrapper
    return decorator


def _track_tool_call(
    tool_name: str,
    latency_ms: float,
    success: bool,
    error_message: str | None,
    params: dict,
) -> None:
    """
    Track an MCP tool call event in PostHog (non-blocking).

    Args:
        tool_name: Name of the tool that was called
        latency_ms: Execution time in milliseconds
        success: Whether the call succeeded
        error_message: Error message if failed (truncated to 500 chars)
        params: Tool parameters (only safe keys are included)
    """
    # Get authenticated user from context
    verified_token = get_mcp_authenticated_user()
    distinct_id = verified_token.user_id if verified_token else "anonymous"

    # Build properties
    properties = {
        "tool_name": tool_name,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "seer_mode": config.seer_mode,
    }

    if error_message:
        # Truncate long error messages
        properties["error"] = error_message[:500]

    # Include only safe parameter info (avoid sensitive data like full workflow specs)
    safe_params = {}
    for key in SAFE_PARAM_KEYS:
        if key in params and params[key] is not None:
            safe_params[key] = params[key]
    if safe_params:
        properties["params"] = safe_params

    # Add user context if authenticated
    if verified_token:
        properties["user_email"] = verified_token.email
        properties["authenticated"] = True

        # Identify user (will be deduped by PostHog)
        identify_user(
            distinct_id=verified_token.user_id,
            properties={
                "email": verified_token.email,
                "first_name": verified_token.first_name,
                "last_name": verified_token.last_name,
            }
        )
    else:
        properties["authenticated"] = False

    # Capture the event (non-blocking)
    capture_event(
        distinct_id=distinct_id,
        event="mcp_tool_call",
        properties=properties,
    )
