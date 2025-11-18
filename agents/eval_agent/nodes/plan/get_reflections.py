from typing import List, Dict

from agents.eval_agent.models import EvalAgentPlannerState
from agents.eval_agent.reflection_store import graph_rag_retrieval
from shared.logger import get_logger

logger = get_logger("eval_agent.plan.get_reflections")

async def get_reflections(state: EvalAgentPlannerState) -> dict:
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

    return {
        "reflections_text": reflections_text,
    }