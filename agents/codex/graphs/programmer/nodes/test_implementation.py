from agents.codex.common.state import ProgrammerState, TaskPlan, TestResults
from shared.logger import get_logger

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from shared.eval_runner import run_evals
from sandbox import deploy_server_and_confirm_ready
from e2b import AsyncSandbox
from sandbox.constants import TARGET_AGENT_COMMAND

logger = get_logger("programmer.test_implementation")


async def test_implementation(state: ProgrammerState) -> ProgrammerState:
    sbx: AsyncSandbox = await AsyncSandbox.connect(state.updated_sandbox_context.sandbox_id, timeout=60*20) # 20 minutes
    sb, handle = await deploy_server_and_confirm_ready(
        cmd=TARGET_AGENT_COMMAND,
        sb=sbx,
        cwd=state.updated_sandbox_context.working_directory,
        timeout_s=50
    )
    results_payload, failed_cases, scores, passed_count, total_tests, earliest_start, latest_end = await run_evals(state.deployment_url, state.testing_context.graph_name, state.testing_context.test_cases)
    await handle.kill()

    if failed_cases:
        test_results = TestResults(success=False, failures=failed_cases)
    else:
        test_results = TestResults(success=True, failures=[])

    return {
        "testResults": test_results
    }
