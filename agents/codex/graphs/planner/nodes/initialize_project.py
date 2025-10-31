from __future__ import annotations

from agents.codex.common.state import PlannerState
from shared.logger import get_logger
logger = get_logger("codex.planner.nodes.initialize_project")



async def initialize_project(state: PlannerState) -> PlannerState:
    # If a remote repo URL is provided, initialize an E2B sandbox and clone/pull there.
    logger.info("Skipping project initialization as it is already done")
    return state


