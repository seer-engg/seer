from typing import Optional, Dict, Any, List
from seer.logger import get_logger
from seer.agents.workflow_agent.tools import (
    analyze_workflow,
    submit_workflow_spec,
    search_tools,
    search_triggers,
    list_available_triggers,
    get_workflow_template,
)

logger = get_logger(__name__)


# pylint: disable=unused-argument # Reason: Reserved for future state injection feature
def get_workflow_tools(workflow_state: Optional[Dict[str, Any]] = None) -> List:
    """
    Get all workflow manipulation tools and dynamic discovery tools.

    Args:
        workflow_state: Reserved for future use. Planned: inject workflow state into tool context
                        so tools can access state without requiring it as a parameter.
    """
    # TODO: Implement workflow_state injection when tool context system is ready
    # Currently, tools use _current_thread_id context instead of explicit state parameter

    # Base tools that are always available
    base_tools = [
        # Workflow manipulation tools
        analyze_workflow,
        submit_workflow_spec,
        # Dynamic tool discovery tools
        search_tools,
        # Dynamic trigger discovery tools
        search_triggers,
        list_available_triggers,
        # Template discovery tools
        get_workflow_template,
        # list_available_tools,
    ]

    return base_tools


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
