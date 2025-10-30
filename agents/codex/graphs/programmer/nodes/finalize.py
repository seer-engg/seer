from agents.codex.common.state import ProgrammerState
from shared.logger import get_logger
logger = get_logger("programmer.finalize")

async def finalize(state: ProgrammerState) -> ProgrammerState:
    logger.info(f"Finalizing programmer state: {state}")
    return state