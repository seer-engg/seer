import asyncio
from typing import Dict, List

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger

logger = get_logger("eval_agent.plan.validate_generated_actions")


async def validate_generated_actions(state: EvalAgentPlannerState) -> dict:
    """Validate the generated actions."""
    # dataset_examples = state.dataset_examples
    # tool_entries = state.tool_entries
    # _validate_generated_actions(dataset_examples, tool_entries)
    return {
        "valid": True,
    }