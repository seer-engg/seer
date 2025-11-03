from __future__ import annotations
from e2b_code_interpreter import AsyncSandbox
from agents.codex.common.state import PlannerState
from shared.logger import get_logger
from sandbox import deploy_server_and_confirm_ready, TARGET_AGENT_COMMAND, TARGET_AGENT_PORT

logger = get_logger("codex.planner.nodes.deploy")


async def deploy_service(state: PlannerState) -> PlannerState:
    """Deploy the LangGraph app in the sandbox and publish its URL.

    Requires in state:
    - sandbox_session_id: E2B sandbox id
    - repo_path: path to repo inside sandbox

    Optional in state:
    - deployment_port: preferred port
    - deployment_graph_name: desired graph to serve
    """
    try:
        sbx: AsyncSandbox = await AsyncSandbox.connect(state.updated_sandbox_context.sandbox_id)
        sb, handle = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=state.updated_sandbox_context.working_directory,
            timeout_s=50
        )
        server_url = sb.get_host(TARGET_AGENT_PORT)
        return {
            "server_running": True,
            "deployment_url": server_url,
            'agent_updated': True,
        }
    except Exception as e:
        logger.error(f"Error deploying service: {e}")
        return {
            "server_running": False,
            "agent_updated": False,
        }
