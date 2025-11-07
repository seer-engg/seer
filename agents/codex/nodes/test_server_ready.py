from e2b import AsyncSandbox
from langchain_core.messages import AIMessage

from shared.logger import get_logger
from sandbox import deploy_server_and_confirm_ready, TARGET_AGENT_COMMAND
from agents.codex.state import CodexState

logger = get_logger("codex.test_server_ready")


async def test_server_ready(state: CodexState) -> CodexState:
    """Test if the server is ready"""
    logger.info(f"Testing server readiness: {state}")
    sbx: AsyncSandbox = await AsyncSandbox.connect(state.updated_sandbox_context.sandbox_id)

    try:
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=state.updated_sandbox_context.working_directory,
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