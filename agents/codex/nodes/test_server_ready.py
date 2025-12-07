from e2b import AsyncSandbox
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from shared.logger import get_logger
from sandbox import deploy_server_and_confirm_ready
from shared.config import config
from agents.codex.state import CodexState
from shared.llm import get_llm
from shared.test_runner.agent_invoker import invoke_target_agent


logger = get_logger("codex.test_server_ready")

class ServerErrorOutput(BaseModel):
    error_message: str = Field(description="The error message from the server")
    reason_for_error: str = Field(description="The reason why the server is not starting")
    fix_instructions: str = Field(description="The instructions to fix the server error")

async def test_server_ready(state: CodexState) -> CodexState:
    """Test if the server is ready"""
    logger.info(f"Testing server readiness: {state}")
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")
    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_context.sandbox_id)

    try:
        _, _ = await deploy_server_and_confirm_ready(
            cmd=config.target_agent_command,
            sb=sbx,
            cwd=sandbox_context.working_directory,
            timeout_s=50
        )
        input_message = "Hi There !"
        result = await invoke_target_agent(
            sandbox_context=sandbox_context,
            agent_name=state.context.agent_name,
            input_message=input_message,
            timeout_seconds=600,
        )
        logger.warning("Server started successfully, not killed")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error starting server: {error_message}")
        return_state = {
            "server_running": False,
        }
        if state.developer_thread:
            llm = get_llm('gpt-5-nano').with_structured_output(ServerErrorOutput)
            prompt = f"""
            Please analyze why we are receving this error: {error_message}
            and reply back with the reason why the langgraph server is not starting.
            and aslo intruct the developer to fix the error.

            """
            input_messages = [HumanMessage(content=prompt)]
            output: ServerErrorOutput = await llm.ainvoke(input_messages)
            reflection = f"""
            The server is not starting due to the following error: {output.error_message}
            The reason for the error is: {output.reason_for_error}

            Please apply the necessary fix to codebase
            """

            return_state["developer_thread"] = [HumanMessage(content=reflection)]
        return return_state

    return {
        "server_running": True,
    }