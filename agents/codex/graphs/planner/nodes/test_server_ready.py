from shared.logger import get_logger
from e2b import AsyncSandbox
from langchain_core.messages import AIMessage
from sandbox import deploy_server_and_confirm_ready, TARGET_AGENT_COMMAND

logger = get_logger("planner.test_server_ready")


async def test_server_ready(state: dict) -> dict:
    """Intentionally keeping the typing of this function open because it's used commonly both by programmer and planner which have different state schemas"""
    logger.info(f"Testing server readiness: {state}")
    sbx: AsyncSandbox = await AsyncSandbox.connect(state.sandbox_context.sandbox_id)

    try:
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=state.sandbox_context.working_directory,
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