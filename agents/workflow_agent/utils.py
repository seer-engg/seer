from typing import Optional, Dict, Any, List
from shared.logger import get_logger
from agents.workflow_agent.tools import analyze_workflow, add_workflow_block, modify_workflow_block, remove_workflow_block, add_workflow_edge, remove_workflow_edge, search_tools

logger = get_logger(__name__)



def get_workflow_tools(workflow_state: Optional[Dict[str, Any]] = None) -> List:
    """
    Get all workflow manipulation tools and dynamic discovery tools.
    
    Args:
        workflow_state: Optional workflow state to inject into tools.
                        If provided, tools will use this state instead of requiring it as a parameter.
    """
    # Base tools that are always available
    base_tools = [
        # Workflow manipulation tools
        analyze_workflow,
        add_workflow_block,
        modify_workflow_block,
        remove_workflow_block,
        add_workflow_edge,
        remove_workflow_edge,
        # Dynamic tool discovery tools
        search_tools,
        # list_available_tools,
    ]
    
    if workflow_state is None:
        # Return tools as-is (they'll require workflow_state parameter)
        return base_tools
    else:
        # Note: Discovery tools don't need workflow_state, so we return them as-is
        # Workflow manipulation tools will need workflow_state passed by LLM
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
