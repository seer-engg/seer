from agents.codex.common.state import ProgrammerState
from shared.logger import get_logger
logger = get_logger("planner.test_server_ready")
from e2b import AsyncSandbox
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.codex.common.constants import TARGET_AGENT_PORT, TARGET_AGENT_COMMAND

from agents.codex.graphs.planner.deploy_server import deploy_server_and_confirm_ready


async def test_server_ready(state: ProgrammerState) -> ProgrammerState:
    logger.info(f"Testing server readiness: {state}")
    sbx: AsyncSandbox = await AsyncSandbox.connect(state.sandbox_session_id)

    try:
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=state.repo_path,
            timeout_s=50
        )
        result = await handle.kill()
    except RuntimeError as e:
        error_message = str(e)
        logger.error(f"Error starting server: {error_message}")
        return {
            "server_running": False,
            "messages": [AIMessage(content=error_message)],
        }
    return {
        "server_running": True,
    }