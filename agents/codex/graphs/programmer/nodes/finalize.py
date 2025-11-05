from agents.codex.common.state import ProgrammerState
from shared.logger import get_logger
from langchain_core.messages import AIMessage
logger = get_logger("programmer.finalize")

async def finalize(state: ProgrammerState) -> ProgrammerState:
    logger.info(f"Finalizing programmer state: {state}")
    if not state.success:
        return {
            "messages": [
                AIMessage(content="The programmer was not successful in implementing the task plan. Evals are still failing Please try again."),
            ],
        }
    return state