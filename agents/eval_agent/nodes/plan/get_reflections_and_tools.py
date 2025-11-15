from agents.eval_agent.models import EvalAgentPlannerState, ToolSelectionLog

from agents.eval_agent.reflection_store import graph_rag_retrieval
from shared.logger import get_logger
from shared.tool_catalog import load_tool_entries, select_relevant_tools
from typing import List, Dict
from shared.tool_catalog import ToolEntry
logger = get_logger("eval_agent.plan.get_reflections")

async def get_reflections_and_tools(state: EvalAgentPlannerState) -> dict:
    """Get the reflections for the test generation."""
    # Get top 3 most relevant reflections + their evidence using GraphRAG
    agent_name = state.github_context.agent_name
    user_id = state.user_context.user_id

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

    if state.mcp_services:
        try:
            tool_entries = await load_tool_entries(state.mcp_services)
            context_for_scoring = "\n".join( 
                filter(None, [state.user_context.raw_request, reflections_text])
            )
            prioritized = await select_relevant_tools(
                tool_entries,
                context_for_scoring,
                max_total=20,
                max_per_service=5,
            )
            if not prioritized:
                prioritized = sorted({entry.name for entry in tool_entries.values()})
            available_tools = prioritized
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