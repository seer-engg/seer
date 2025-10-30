from shared.logger import get_logger
logger = get_logger("programmer.reflect")
from agents.codex.common.state import ProgrammerState
from langchain_core.messages import HumanMessage


PROMPT = """
To check your task implementation , i ran the tests and they failed. Please reflect on the tests and try to resolve them 
<test_results>
{test_results}
</test_results>
"""

async def reflect(state: ProgrammerState) -> ProgrammerState:
    results = state.testResults

    human_message = HumanMessage(content=PROMPT.format(test_results=results))
    return state