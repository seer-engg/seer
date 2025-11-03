from agents.codex.common.state import ProgrammerState, TaskPlan, TestResults
from shared.logger import get_logger
logger = get_logger("programmer.test_implementation")
from langchain.agents import create_agent
from agents.codex.llm.model import get_chat_model

from shared.tools import web_search, think
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from shared.eval_runner import run_evals



async def test_implementation(state: ProgrammerState) -> ProgrammerState:

    results_payload, failed_cases, scores, passed_count, total_tests, earliest_start, latest_end = await run_evals(state.deployment_url, state.testing_context.graph_name, state.testing_context.test_cases)

    if failed_cases:
        test_results = TestResults(success=False, failures=failed_cases)
    else:
        test_results = TestResults(success=True, failures=[])

    return {
        "testResults": test_results
    }
