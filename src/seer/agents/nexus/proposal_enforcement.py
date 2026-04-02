"""
Middleware to enforce that the agent calls a terminal tool before finishing.

Uses two LangChain middleware hooks to create a structural guarantee:

1. aafter_model (with jump_to) — detects when the agent tries to finish
   without a successful terminal tool call, injects a reminder and loops back.
2. awrap_model_call — when enforcement is active, forces tool_choice="any"
   so the model MUST produce an actual tool call (not text that looks like one).

This is analogous to Claude Code's planning mode, where the harness enforces
that a plan file is always written.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from seer.logger import get_logger

logger = get_logger(__name__)

TERMINAL_TOOLS = frozenset({
    "submit_workflow_spec",
    "ask_clarification_questions",
    "complete_response",
})

_MAX_ENFORCEMENT_RETRIES = 3

_ENFORCEMENT_MARKER = "ENFORCEMENT: You must use an actual tool call"

_ENFORCEMENT_MESSAGE_DEFAULT = (
    f"{_ENFORCEMENT_MARKER} — do NOT write tool calls as text. "
    "Use the tool calling mechanism to invoke one of: "
    "submit_workflow_spec (for workflow changes), "
    "ask_clarification_questions (if you need user input), or "
    "complete_response (if no workflow change is needed). "
    "If your previous submit_workflow_spec calls failed validation, "
    "use get_workflow_guide() to review the schema, fix the errors, "
    "and try submitting again."
)

_ENFORCEMENT_MESSAGE_MAX_RETRIES = (
    f"{_ENFORCEMENT_MARKER}. "
    "Your submit_workflow_spec attempts have all failed and the retry limit "
    "has been reached. You MUST now call complete_response("
    "response_type='general_help', reasoning='<explain the validation errors "
    "you encountered>') to inform the user about the issues so they can help. "
    "Do NOT call submit_workflow_spec again."
)


def _has_successful_terminal_tool_call(messages: list[Any]) -> bool:
    """Check if any terminal tool was called AND succeeded in the message history.

    A terminal tool call is only considered successful if the ToolMessage
    response does NOT contain '"status": "error"'. This prevents the agent
    from finishing after failed submit_workflow_spec attempts.
    """
    # Build a map of tool_call_id → tool_name from AIMessages
    terminal_call_ids: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                call_id = tc.get("id", "")
                if name in TERMINAL_TOOLS and call_id:
                    terminal_call_ids[call_id] = name

    if not terminal_call_ids:
        return False

    # Check ToolMessage responses for those calls
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in terminal_call_ids:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # A response containing "status": "error" means the call failed
            if '"status": "error"' not in content:
                return True

    return False


def _has_max_retries_error(messages: list[Any]) -> bool:
    """Check if the most recent submit_workflow_spec response was a max_retries error."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if '"max_retries"' in content:
                return True
            # Stop at the first ToolMessage (only check the most recent)
            return False
    return False


def _count_enforcement_reminders(messages: list[Any]) -> int:
    """Count enforcement reminders injected (prevents infinite loops)."""
    count = 0
    for msg in messages:
        if isinstance(msg, SystemMessage) and _ENFORCEMENT_MARKER in (msg.content or ""):
            count += 1
    return count


def _is_enforcement_active(messages: list[Any]) -> bool:
    """Check if the most recent SystemMessage is an enforcement reminder."""
    for msg in reversed(messages):
        if isinstance(msg, SystemMessage):
            return _ENFORCEMENT_MARKER in (msg.content or "")
        # Stop scanning at the first non-system message before our marker
        if isinstance(msg, (AIMessage, ToolMessage)):
            return False
    return False


class ProposalEnforcementMiddleware(AgentMiddleware):
    """Enforce that the agent calls a terminal tool before finishing.

    Two hooks work together:
    - aafter_model: detects when the agent tries to finish without a
      successful terminal tool. Injects a reminder and loops back.
    - awrap_model_call: when enforcement is active, forces tool_choice="any"
      so the model MUST produce an actual tool call, not text output.
    """

    @hook_config(can_jump_to=["model"])  # pylint: disable=not-callable  # Reason: hook_config is a decorator from langchain
    async def aafter_model(  # pylint: disable=arguments-differ  # Reason: LangChain middleware hook signature
        self, state: Any, runtime: Any,
    ) -> dict[str, Any] | None:
        """After each model call, check if the agent is trying to finish."""
        del runtime

        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        # Only intercept when agent tries to finish (no tool calls)
        if not isinstance(last_msg, AIMessage) or last_msg.tool_calls:
            return None  # Agent is calling tools, let it proceed

        # Check if any terminal tool was called AND succeeded during this turn
        if _has_successful_terminal_tool_call(messages):
            return None  # Terminal tool succeeded, allow finish

        # Check retry count to prevent infinite loops
        enforcement_count = _count_enforcement_reminders(messages)
        if enforcement_count >= _MAX_ENFORCEMENT_RETRIES:
            logger.warning(
                "Agent exhausted enforcement retries without successful terminal tool "
                "(retries=%d)", enforcement_count,
            )
            return None  # Give up after max retries, let it finish

        # Pick the right message: if max_retries was hit, tell agent to use complete_response
        if _has_max_retries_error(messages):
            enforcement_message = _ENFORCEMENT_MESSAGE_MAX_RETRIES
            logger.info(
                "Agent hit max submission retries, directing to complete_response "
                "(enforcement attempt %d/%d)", enforcement_count + 1, _MAX_ENFORCEMENT_RETRIES,
            )
        else:
            enforcement_message = _ENFORCEMENT_MESSAGE_DEFAULT
            logger.info(
                "Agent tried to finish without successful terminal tool, injecting reminder "
                "(attempt %d/%d)", enforcement_count + 1, _MAX_ENFORCEMENT_RETRIES,
            )

        # Inject reminder and loop back to model
        reminder = SystemMessage(content=enforcement_message)
        return {"messages": [reminder], "jump_to": "model"}

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """When enforcement is active, force tool_choice='any' so model must produce a tool call."""
        if hasattr(request, "messages") and _is_enforcement_active(request.messages):
            logger.debug("Enforcement active — forcing tool_choice='any' on model call")
            request.tool_choice = "any"

        return await handler(request)
