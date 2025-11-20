from e2b import AsyncSandbox
from langchain_core.messages import AIMessage, HumanMessage

from shared.logger import get_logger
from sandbox import deploy_server_and_confirm_ready, TARGET_AGENT_COMMAND, kill_process_on_port, TARGET_AGENT_PORT
from agents.codex.state import CodexState

logger = get_logger("codex.test_server_ready")


async def test_server_ready(state: CodexState) -> CodexState:
    """Test if the server is ready"""
    logger.info(f"Testing server readiness: {state}")
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")
    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_context.sandbox_id)

    try:
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=sandbox_context.working_directory,
            timeout_s=50
        )
        await kill_process_on_port(sbx, TARGET_AGENT_PORT)
    except RuntimeError as e:
        error_message = str(e)
        logger.error(f"Error starting server: {error_message}")
        return_state = {
            "server_running": False,
        }
        if state.taskPlan:
            return_state["coder_thread"] = [HumanMessage(content=error_message)]
        return return_state
    return {
        "server_running": True,
    }