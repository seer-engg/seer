from langchain_core.tools import tool


@tool
def think(
    scratchpad: str,
    last_tool_call: str = "",  # Optional: What tool was just called and what it returned
) -> str:
    """Use this tool to think and reason about the current situation.
    
    CRITICAL: You MUST call this tool after EVERY tool call (except 'think' itself) 
    to reflect on the results before proceeding. This is mandatory for proper reasoning.
    
    **CONTEXT:**
    - Relevant context (todos, plan, task instructions) is already available in your system prompt
    - Reference that context when reasoning - you don't need to repeat it here
    
    **FORMAT YOUR THINKING:**
    Use the scratchpad to:
    1. Analyze what just happened (last tool call and its result)
    2. Consider what this means for the current task/plan
    3. Decide what to do next
    
    This ensures you reason step-by-step rather than blindly executing tools."""
    parts = [f"Thought: {scratchpad}"]
    if last_tool_call:
        parts.append(f"Last tool: {last_tool_call}")
    return "\n".join(parts)

