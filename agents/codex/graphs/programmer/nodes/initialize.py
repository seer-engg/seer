from shared.logger import get_logger
logger = get_logger("programmer.initialize")
from agents.codex.common.state import ProgrammerState
from langchain.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
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
    messages = list[BaseMessage](state.messages)
    messages.append(HumanMessage(content=USER_PROMPT.format(request=state.request, task_plan=state.taskPlan)))
    return {
        "messages": messages,
    }