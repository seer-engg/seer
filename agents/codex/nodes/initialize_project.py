from __future__ import annotations

from agents.codex.state import CodexState
from shared.logger import get_logger
logger = get_logger("codex.nodes.initialize_project")

from sandbox import initialize_e2b_sandbox, setup_project
from shared.config import config
from shared.schema import SandboxContext



async def initialize_project(state: CodexState) -> CodexState:
    """If a remote repo URL is provided, initialize an E2B sandbox and clone/pull there."""
    logger.info("Skipping project initialization as it is already done")
    env_vars = {
        "COMPOSIO_USER_ID": state.context.user_id,
    }
    sbx, repo_dir, branch_in_sandbox = await initialize_e2b_sandbox(
        repo_url=state.context.github_context.repo_url,
        branch_name=state.context.sandbox_context.working_branch,
        env_vars=env_vars,
    )
    await setup_project(sbx.sandbox_id, repo_dir, config.target_agent_setup_script)
    state.context.sandbox_context = SandboxContext(
        sandbox_id=sbx.sandbox_id,
        working_directory=repo_dir,
        working_branch=branch_in_sandbox,
    )
    return {
        "context": state.context,
    }

