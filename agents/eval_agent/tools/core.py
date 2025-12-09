"""
Core tools for eval agent: think and write_todos.
"""
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from typing import Optional, List
from shared.logger import get_logger

logger = get_logger("eval_agent.tools.core")


@tool
def think(scratchpad: str, last_tool_call: str, runtime: ToolRuntime = None) -> str:
    """
    Use this tool to think and reason about the current situation.
    
    **CRITICAL: You MUST call this tool BEFORE calling any action tool.**
    Every action tool call MUST be preceded by a think() call that plans the execution.
    
    **REQUIRED: `last_tool_call` parameter**
    - You MUST provide `last_tool_call` on EVERY call
    - Format: "Tool: <tool_name>, Result: <what happened>"
    - For first call: "Tool: None, Result: Initial call"
    - After tool execution: "Tool: <tool_name>, Result: <success/error summary>"
    
    **BEFORE CALLING ACTION TOOLS - MANDATORY PLANNING:**
    When planning to call an action tool, explicitly state in your scratchpad:
    1. Which tool you'll call
    2. What parameters you'll provide and why
    3. What you expect the result to be
    
    **AFTER CALLING ACTION TOOLS - REFLECTION:**
    Reflect on results and plan next steps. Always include what tool was called and what happened.
    
    **FORMAT YOUR THINKING:**
    Use the scratchpad to:
    1. Analyze what just happened (last tool call and its result)
    2. Consider what this means for the current task/plan
    3. Decide what to do next (if planning an action tool, state tool name and params with reasoning)
    
    This ensures you reason step-by-step rather than blindly executing tools.
    """
    # Try to stream thinking output if in streaming context
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        if writer:
            # Stream the thinking output
            writer(f"💭 Thought: {scratchpad}")
            if last_tool_call:
                writer(f"📋 Last tool: {last_tool_call}")
    except ImportError:
        # langgraph.config not available, skip streaming
        pass
    except Exception:
        # get_stream_writer() failed (not in streaming context), skip streaming
        pass
    
    # Return formatted response
    parts = [f"Thought: {scratchpad}"]
    if last_tool_call:
        parts.append(f"Last tool: {last_tool_call}")
    return "\n".join(parts)


@tool
def write_todos(todos: List[str], runtime: ToolRuntime = None) -> str:
    """
    Update the todos list. Pass a list of todo items (strings).
    
    Todos represent evaluation phases: ["PLANNING", "EXECUTION", "REFLECTION", "FINALIZATION"]
    
    **CRITICAL**: Todos track the current phase of evaluation.
    - Remove a phase from todos when it's complete
    - Add the next phase when starting it
    - When todos are empty, evaluation is complete
    
    **PHASE ORDERING**:
    - PLANNING: Extract config, generate spec, generate tests, show alignment questions
    - EXECUTION: Execute test batch (only if not plan-only mode)
    - REFLECTION: Analyze results and generate hypothesis
    - FINALIZATION: Handoff to codex and summarize
    
    Example: write_todos(["PLANNING"])  # Start planning phase
    Example: write_todos(["EXECUTION"])  # Move to execution phase
    Example: write_todos([])  # All phases complete
    
    This replaces the entire todos list. Use this to create or update your plan.
    """
    return f"✅ Todos updated: {len(todos)} phase(s) - {', '.join(todos) if todos else 'COMPLETE'}"

