from __future__ import annotations

import os
import base64
import shlex
from typing import Optional, Dict
from e2b import AsyncSandbox, CommandResult

from .constants import _build_git_shell_script
from shared.logger import get_logger
from shared.config import config

logger = get_logger("sandbox.initialize")

def _masked(s: str) -> str:
    token = config.github_token or ""
    if not token:
        return s
    return s.replace(token, "***")


async def initialize_e2b_sandbox(
    repo_url: str,
    branch_name: str = "main",
    github_token: Optional[str] = config.github_token,
    env_vars: Optional[Dict[str, str]] = None,
) -> tuple[AsyncSandbox, str, str]:
    """
    Create (or resume) an E2B sandbox and ensure the GitHub repository is cloned
    and up to date on the requested branch.

    Returns: (sbx, repo_dir, branch_in_sandbox)
    """
    if not os.getenv("E2B_API_KEY"):
        raise RuntimeError("E2B_API_KEY not configured in environment")

    logger.info("Creating E2B sandbox for codex...")
    envs = config.target_agent_envs
    if env_vars:
        envs.update(env_vars)
    sbx: AsyncSandbox = await AsyncSandbox.beta_create(
      template=config.base_template_alias,
      auto_pause=True,
      envs=envs,
      timeout=60*30, # 30 minutes
    )
    sandbox_id = sbx.sandbox_id
    logger.info(f"Sandbox created: {sandbox_id}")

    shell_script = await _build_git_shell_script()
    logger.info("Cloning/updating repository inside sandbox (shell)...")

    # Encode the shell script to avoid escaping issues and run it via bash
    script_b64 = base64.b64encode(shell_script.encode()).decode()
    token_value = github_token or ""
    # Quote env values safely
    repo_url_q = shlex.quote(repo_url)
    branch_name_q = shlex.quote(branch_name)
    token_value_q = shlex.quote(token_value)
    # Build a robust command that writes the script via base64 heredoc and executes it
    cmd = f"""REPO_URL={repo_url_q} BRANCH={branch_name_q} TOKEN={token_value_q} bash -lc 'set -euo pipefail; TMP=$(mktemp -t codex_sbx.XXXXXX.sh); base64 -d > "$TMP" << "B64EOF"
{script_b64}
B64EOF
chmod +x "$TMP"; bash "$TMP"'"""

    execution: CommandResult = await sbx.commands.run(cmd, timeout=60*5) # 5 minutes

    exit_code = execution.exit_code
    stdout = execution.stdout
    stderr = execution.stderr

    if exit_code != 0:
        logger.error(_masked(f"Sandbox git setup error: {stderr or stdout}"))
        raise RuntimeError("Failed to prepare repository in sandbox")

    res = await sbx.commands.run("sudo apt install -y tree")
    if res.exit_code != 0:
        logger.error(f"Failed to install tree: {res.stderr or res.stdout}")
        raise RuntimeError("Failed to install tree")

    # Parse stdout lines to find our markers
    repo_dir = ""
    branch_in_sandbox = branch_name
    for line in stdout.splitlines():
        if line.startswith("SANDBOX_REPO_DIR="):
            repo_dir = line.split("=", 1)[1].strip()
        if line.startswith("SANDBOX_BRANCH="):
            branch_in_sandbox = line.split("=", 1)[1].strip()

    return sbx, repo_dir, branch_in_sandbox



async def setup_project(sandbox_id: str, repo_dir: str, setup_script: str) -> str:

    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_id)
    execution: CommandResult = await sbx.commands.run(setup_script, cwd=repo_dir)
    exit_code = execution.exit_code
    stdout = execution.stdout
    stderr = execution.stderr
    logger.info(f"Setup project: {stdout}")

    if exit_code != 0:
        logger.error(f"Failed to setup project: {stderr or stdout}")
        raise RuntimeError("Failed to setup project")