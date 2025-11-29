from __future__ import annotations
from e2b_code_interpreter import AsyncSandbox
from agents.codex.state import CodexState
from shared.logger import get_logger
from sandbox import deploy_server_and_confirm_ready
from shared.config import config

logger = get_logger("codex.nodes.deploy")


async def deploy_service(state: CodexState) -> CodexState:
    """Deploy the LangGraph app in the sandbox and publish its URL.

    Requires in state:
    - sandbox_session_id: E2B sandbox id
    - repo_path: path to repo inside sandbox

    Optional in state:
    - deployment_port: preferred port
    - deployment_graph_name: desired graph to serve
    """
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")
    try:
        sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_context.sandbox_id)
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=config.target_agent_command,
            sb=sbx,
            cwd=sandbox_context.working_directory,
            timeout_s=50
        )
        return {
            "server_running": True,
            'agent_updated': True,
        }
    except Exception as e:
        logger.error(f"Error deploying service: {e}")
        return {
            "server_running": False,
            "agent_updated": False,
        }
