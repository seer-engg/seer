"""Test the implementation of the task plan"""
from typing import List
from e2b import AsyncSandbox

from agents.codex.state import CodexState
from shared.logger import get_logger
from shared.eval_runner import run_evals
from shared.schema import ExperimentResultContext
from sandbox import deploy_server_and_confirm_ready, TARGET_AGENT_PORT, kill_process_on_port
from sandbox.constants import TARGET_AGENT_COMMAND


logger = get_logger("programmer.test_implementation")


async def evaluator(state: CodexState) -> CodexState:
    """Test the implementation of the task plan"""
    sbx: AsyncSandbox = await AsyncSandbox.connect(state.updated_sandbox_context.sandbox_id, timeout=60*20) # 20 minutes
    _, handle = await deploy_server_and_confirm_ready(
        cmd=TARGET_AGENT_COMMAND,
        sb=sbx,
        cwd=state.updated_sandbox_context.working_directory,
        timeout_s=50
    )
    url = f"https://{sbx.get_host(TARGET_AGENT_PORT)}"

    results: List[ExperimentResultContext] = await run_evals(
        url,
        state.github_context.agent_name,
        state.dataset_examples,
        user_id=state.user_context.user_id,
    )

    await kill_process_on_port(sbx, TARGET_AGENT_PORT)

    success = len([result for result in results if not result.passed]) == 0

    return {
        "latest_test_results": results,
        "success": success,
        "attempt_number": state.attempt_number + 1,
    }
