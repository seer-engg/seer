"""
PostHog tracking for Nexus agent tool calls.

Provides a decorator to track Nexus tool invocations with:
- Tool name
- Execution time
- Success/failure status
- User context from thread ContextVar

All tracking is non-blocking, mirroring src/seer/mcp/tracking.py.

Usage:
    from seer.agents.nexus.tracking import track_nexus_tool

    @tool
    @track_nexus_tool("submit_workflow_spec")
    async def submit_workflow_spec(workflow_spec: Any) -> str:
        ...
"""
import functools
import time
from typing import Any, Callable

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


def track_nexus_tool(tool_name: str):
    """
    Decorator to track Nexus agent tool calls in PostHog.

    Must be applied AFTER @tool (innermost decorator) so the tracking
    wraps the actual async function rather than the LangChain tool object.

    Args:
        tool_name: Human-readable name of the tool (e.g., "submit_workflow_spec")

    Example:
        @tool
        @track_nexus_tool("submit_workflow_spec")
        async def submit_workflow_spec(workflow_spec: Any) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not config.is_posthog_configured:
                return await func(*args, **kwargs)

            start_time = time.perf_counter()
            success = True
            error_str: str | None = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                success = False
                error_str = str(exc)[:500]
                raise
            finally:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                _track_tool_call(tool_name, latency_ms, success, error_str)

        return wrapper
    return decorator


def _track_tool_call(
    tool_name: str,
    latency_ms: int,
    success: bool,
    error_str: str | None,
) -> None:
    """Emit nexus_tool_called event to PostHog (non-blocking)."""
    # pylint: disable=import-outside-toplevel  # Reason: Lazy import to avoid circular dependency
    from seer.agents.nexus.context import _current_thread_id, get_user_for_thread
    from seer.analytics.workflow_tracking import capture_workflow_event
    import asyncio

    thread_id = _current_thread_id.get()
    if not thread_id:
        return

    properties: dict = {
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "success": success,
    }
    if error_str:
        properties["error"] = error_str

    async def do_capture() -> None:
        user = await get_user_for_thread(thread_id)
        if not user:
            return
        await capture_workflow_event(
            event="nexus_tool_called",
            user_email=user.email,
            properties=properties,
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(do_capture())
        else:
            loop.run_until_complete(do_capture())
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Tracking failures must not break tool execution
        logger.warning("Failed to schedule nexus_tool_called tracking: %s", exc)
