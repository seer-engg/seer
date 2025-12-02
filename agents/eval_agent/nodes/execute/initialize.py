from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger


logger = get_logger("eval_agent.execute.initialize")


async def initialize_node(state: TestExecutionState) -> dict:

    updates: dict = {}

    # Initialize pending list and accumulator on first entry
    pending = list(state.dataset_examples or [])
    updates["pending_examples"] = pending

    return updates