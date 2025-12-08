from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger

logger = get_logger("eval_agent.plan.get_reflections")

async def get_reflections(state: EvalAgentPlannerState) -> dict:
    """Get the reflections for the test generation."""
    # No Neo4j GraphRAG available - return empty
    # reflection_store.py deleted, no past reflections available
    logger.info("get_reflections: No Neo4j memory available, returning empty")

    return {
        "reflections_text": "",  # Empty - no past reflections
    }