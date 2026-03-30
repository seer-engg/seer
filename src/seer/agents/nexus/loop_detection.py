"""
Middleware to detect and break repeated identical tool call loops.

When the agent calls the same tool with the same arguments multiple times
in a row, this middleware intercepts and injects a message telling the agent
to try a different approach or use a terminal tool (complete_response,
submit_workflow_spec, ask_clarification_questions).

Uses the aafter_model hook to inspect the agent's latest tool calls and
compare them against recent history.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from seer.logger import get_logger

logger = get_logger(__name__)

# How many consecutive calls to the same tool before we intervene
_DUPLICATE_THRESHOLD = 3

# How many times we inject the loop-break message before giving up
_MAX_LOOP_BREAK_ATTEMPTS = 2

_LOOP_BREAK_MARKER = "LOOP DETECTED: You are repeating the same tool call"

_LOOP_BREAK_MESSAGE = (
    "{marker} — you called `{tool_name}` with the same arguments {count} times "
    "and got the same result each time. The information you need is NOT available "
    "through this tool call. You MUST try a DIFFERENT approach:\n"
    "- Use a different tool or different search query\n"
    "- Use schedule.cron trigger instead if no specific trigger exists\n"
    "- Call ask_clarification_questions() to ask the user for guidance\n"
    "- Call complete_response() to explain what you found and ask for help\n"
    "- Call submit_workflow_spec() with the best available alternative\n\n"
    "Do NOT repeat the same tool call again."
)


def _find_repeated_tool_calls(messages: list[Any]) -> tuple[str, int] | None:
    """Scan recent messages for repeated calls to the same tool.

    Compares by tool NAME only (not args) since models often vary
    unimportant fields like 'reasoning' while repeating the same query.

    Returns (tool_name, count) if a tool was called >= _DUPLICATE_THRESHOLD
    times consecutively. Returns None otherwise.
    """
    recent_tool_names: list[str] = []

    for msg in reversed(messages):
        if isinstance(msg, (ToolMessage, SystemMessage)):
            continue  # Skip responses and system messages
        if isinstance(msg, AIMessage):
            if not msg.tool_calls:
                break  # Stop at a non-tool-calling AI message
            # Take the first tool call name from each AIMessage
            recent_tool_names.append(msg.tool_calls[0].get("name", ""))
        else:
            break  # Stop at non-AI messages (HumanMessage, etc.)

    if not recent_tool_names:
        return None

    # Count consecutive identical tool names (from most recent)
    target = recent_tool_names[0]
    count = 0
    for name in recent_tool_names:
        if name == target:
            count += 1
        else:
            break

    if count >= _DUPLICATE_THRESHOLD:
        return target, count

    return None


def _count_loop_break_messages(messages: list[Any]) -> int:
    """Count loop-break messages injected (prevents infinite meta-loops)."""
    count = 0
    for msg in messages:
        if isinstance(msg, SystemMessage) and _LOOP_BREAK_MARKER in (msg.content or ""):
            count += 1
    return count


def _is_loop_break_active(messages: list[Any]) -> bool:
    """Check if the most recent SystemMessage is a loop-break message."""
    for msg in reversed(messages):
        if isinstance(msg, SystemMessage):
            return _LOOP_BREAK_MARKER in (msg.content or "")
        if isinstance(msg, (AIMessage, ToolMessage)):
            return False
    return False


class LoopDetectionMiddleware(AgentMiddleware):
    """Detect and break repeated identical tool call loops.

    Two hooks:
    - aafter_model: detects when the agent is about to make the same tool call
      again, injects a loop-break message and routes back to the model.
    - awrap_model_call: when loop-break is active, forces tool_choice="any"
      to ensure the model produces a real tool call (not text).
    """

    @hook_config(can_jump_to=["model"])  # pylint: disable=not-callable  # Reason: hook_config is a decorator from langchain
    async def aafter_model(  # pylint: disable=arguments-differ  # Reason: LangChain middleware hook signature
        self, state: Any, runtime: Any,
    ) -> dict[str, Any] | None:
        """After each model call, check for repeated tool calls."""
        del runtime

        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        # Only check when the agent is making tool calls
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return None  # Not making tool calls, nothing to check

        # Check for repeated identical tool calls
        repeated = _find_repeated_tool_calls(messages)
        if repeated is None:
            return None  # No repetition detected

        tool_name, count = repeated

        # Check if we've already tried to break this loop enough times
        loop_break_count = _count_loop_break_messages(messages)
        if loop_break_count >= _MAX_LOOP_BREAK_ATTEMPTS:
            logger.warning(
                "Agent stuck in tool call loop despite %d break attempts "
                "(tool=%s, count=%d)", loop_break_count, tool_name, count,
            )
            return None  # Give up, let the recursion limit handle it

        logger.info(
            "Detected repeated tool call loop: %s called %d times, "
            "injecting loop-break message (attempt %d/%d)",
            tool_name, count, loop_break_count + 1, _MAX_LOOP_BREAK_ATTEMPTS,
        )

        # Inject loop-break message and route back to model
        message = _LOOP_BREAK_MESSAGE.format(
            marker=_LOOP_BREAK_MARKER,
            tool_name=tool_name,
            count=count,
        )
        reminder = SystemMessage(content=message)
        return {"messages": [reminder], "jump_to": "model"}

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """When loop-break is active, force tool_choice='any' to ensure real tool call."""
        if hasattr(request, "messages") and _is_loop_break_active(request.messages):
            logger.debug("Loop-break active — forcing tool_choice='any' on model call")
            request.tool_choice = "any"

        return await handler(request)
