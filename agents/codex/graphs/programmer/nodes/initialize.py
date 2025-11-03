from langchain.messages import HumanMessage
from shared.logger import get_logger
from agents.codex.common.state import ProgrammerState

logger = get_logger("programmer.initialize")

USER_PROMPT = """
    based on the request 
    <request>
    {request}
    </request>

    Implement the following task plan:
    <task_plan>
    {task_plan}
    </task_plan>

    After implementing the task plan, return a brief status summary.
"""

async def initialize(state: ProgrammerState) -> ProgrammerState:
    logger.info(f"Initializing programmer state: {state}")
    messages = list(state.messages or [])
    messages.append(
        HumanMessage(
            content=USER_PROMPT.format(
                request=state.user_context.user_expectation,
                task_plan=state.taskPlan,
            )
        )
    )
    return {
        "messages": messages,
    }