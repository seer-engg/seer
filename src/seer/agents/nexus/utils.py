from typing import Any, List, Optional
from seer.config import config
from seer.logger import get_logger
from seer.agents.nexus.tools import (
    submit_workflow_spec,
    ask_clarification_questions,
    web_search,
    memory_tools,
)
from seer.agents.nexus.tools.workflow_tools import (
    create_bound_get_workflow,
    create_bound_analyze_workflow,
)

logger = get_logger(__name__)


def get_workflow_tools(workflow_id: Optional[str] = None) -> List:
    """
    Get all workflow manipulation tools and dynamic discovery tools.

    Uses the unified tool registry for discovery/template tools (shared with MCP),
    plus Nexus-only tools that are tightly coupled to the agent context.

    Args:
        workflow_id: If provided, creates get_workflow and analyze_workflow tools
                     with the workflow_id pre-bound so the agent doesn't need to pass it.
    """
    # Register unified tools (idempotent) and get LangGraph-compatible versions
    from seer.tools.unified_tools import register_unified_tools  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    register_unified_tools()
    from seer.tools.tool_factory import unified_registry  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    # Nexus-only tools (tightly coupled to agent context, not in factory)
    nexus_only_tools = [
        submit_workflow_spec,
        ask_clarification_questions,
        web_search,
    ]

    # Add workflow-aware tools if workflow_id is provided
    if workflow_id:
        nexus_only_tools.extend([
            create_bound_get_workflow(workflow_id),
            create_bound_analyze_workflow(workflow_id),
        ])
        logger.debug("Created workflow-bound tools for workflow_id=%s", workflow_id)

    # Add memory tools if memory is enabled
    if config.memory_enabled:
        nexus_only_tools.extend(memory_tools)
        logger.debug("Memory tools enabled: %d tools added", len(memory_tools))

    return unified_registry.get_langgraph_tools() + nexus_only_tools


def extract_thinking_from_messages(messages: List[Any]) -> List[str]:
    """
    Extract thinking/reasoning steps from agent messages.

    This looks for tool calls and intermediate reasoning in the message history.

    Args:
        messages: List of messages from agent

    Returns:
        List of thinking steps
    """
    thinking_steps = []

    for msg in messages:
        # Check for tool calls (indicates reasoning about what to do)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                thinking_steps.append(
                    f"Calling tool '{tool_call.get('name', 'unknown')}' "
                    f"with args: {tool_call.get('args', {})}"
                )

        # Check for tool results (indicates reasoning about results)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Look for reasoning patterns in content
            if "analyzing" in msg.content.lower() or "considering" in msg.content.lower():
                # Extract short reasoning snippets
                content_lines = msg.content.split("\n")
                for line in content_lines[:3]:  # First few lines often contain reasoning
                    if len(line.strip()) > 20 and len(line.strip()) < 200:
                        thinking_steps.append(line.strip())

    return thinking_steps
