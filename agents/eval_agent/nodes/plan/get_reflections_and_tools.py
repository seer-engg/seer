from typing import List, Dict

from agents.eval_agent.models import EvalAgentPlannerState, ToolSelectionLog
from agents.eval_agent.reflection_store import graph_rag_retrieval
from shared.logger import get_logger
from shared.tool_service import get_tool_service
from shared.tools import ToolEntry

logger = get_logger("eval_agent.plan.get_reflections")

async def get_reflections_and_tools(state: EvalAgentPlannerState) -> dict:
    """Get the reflections for the test generation."""
    # Get top 3 most relevant reflections + their evidence using GraphRAG
    agent_name = state.context.github_context.agent_name
    user_id = state.context.user_context.user_id

    reflections_text = await graph_rag_retrieval(
        query="what previous tests failed and why?",
        agent_name=agent_name,
        user_id=user_id,
        limit=3
    )
    available_tools: List[str] = []
    tool_entries: Dict[str, ToolEntry] = {}


    # We'll initialize context_for_scoring here to ensure it's always defined
    context_for_scoring = ""

    if state.context.mcp_services:
        try:
            tool_service = get_tool_service()
            await tool_service.initialize(state.context.mcp_services)
            tool_entries = tool_service.get_tool_entries()
            
            context_for_scoring = "\n".join( 
                filter(None, [state.context.user_context.raw_request, reflections_text])
            )
            
            prioritized = await tool_service.select_relevant_tools(
                context_for_scoring,
                max_total=40,  # Increased from 20 to accommodate multiple services
                max_per_service=20,  # Increased from 10 to ensure all relevant tools per service
            )
            
            # Convert tools to names
            available_tools = [tool.name for tool in prioritized]
            if not available_tools:
                available_tools = sorted({entry.name for entry in tool_entries.values()})
                
            logger.info(
                "Found %d prioritized MCP tools for test generation.",
                len(available_tools),
            )
        except Exception as exc:
            logger.error(f"Failed to load MCP tools for test generation: {exc}")
    else:
        logger.info("No MCP services configured; tool prompts will be limited to system tools.")
    
    log_entry = ToolSelectionLog(
        selection_context=context_for_scoring,
        selected_tools=available_tools
    )


    return {
        "reflections_text": reflections_text,
        "available_tools": available_tools,
        "tool_selection_log": log_entry,
        "tool_entries": tool_entries,

    }