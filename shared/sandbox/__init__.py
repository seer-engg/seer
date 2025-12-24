from .base import (
    cd_and_run_in_sandbox,
    get_sandbox,
)
from .commands import kill_process_on_port
from .deploy import deploy_server_and_confirm_ready
from .initialize import initialize_e2b_sandbox, setup_project
from e2b import AsyncSandbox
from shared.config import config
from typing import Optional, Dict
__all__ = [
    "cd_and_run_in_sandbox",
    "get_sandbox",
    "deploy_server_and_confirm_ready",
    "initialize_e2b_sandbox",
    "setup_project",
    "prepare_target_agent",
    "kill_process_on_port",
]


async def prepare_target_agent(repo_url: str, setup_script: str,env_vars: Optional[Dict[str, str]] = None, timeout: int = 900) -> tuple[AsyncSandbox, str]:
    """
    Prepare the target agent by initializing the sandbox, setting up the project, and deploying the server.
    Args:
        repo_url: The URL of the repository to prepare.
        setup_script: The script to run to setup the project.
        timeout: The timeout in seconds to wait for the server to be ready.
    Returns:
        A tuple containing the sandbox and the deployed URL.
    """
    sbx, repo_dir, _ = await initialize_e2b_sandbox(
        repo_url=repo_url,
        env_vars=env_vars,
    )
    await setup_project(sbx.sandbox_id, repo_dir, setup_script)

    sbx, _ = await deploy_server_and_confirm_ready(
        cmd=config.target_agent_command,
        sb=sbx,
        cwd=repo_dir,
        timeout_s=60
    )

    deployed_url = sbx.get_host(config.target_agent_port)
    return sbx, deployed_url
