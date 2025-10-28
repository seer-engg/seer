from __future__ import annotations

from agents.codex.common.state import PlannerState
from shared.logger import get_logger
import os
logger = get_logger("codex.planner.nodes.initialize_project")

import textwrap

import base64
import shlex
from typing import Optional
from sandbox import Sandbox

async def _build_git_shell_script() -> str:
    # Real shell script to run inside sandbox. Uses http.extraHeader to avoid embedding tokens in remotes.
    script = """
set -euo pipefail

: "${REPO_URL:?REPO_URL is required}"
BRANCH="${BRANCH:-main}"
TOKEN="${TOKEN:-}"

REPO_DIR="$(basename "$REPO_URL")"
REPO_DIR="${REPO_DIR%.git}"

AUTH_CFG=()
if [ -n "$TOKEN" ]; then
  AUTH_CFG=(-c "http.extraHeader=Authorization: bearer $TOKEN")
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  git "${AUTH_CFG[@]}" clone "$REPO_URL"
fi

cd "$REPO_DIR"

# Fetch branch; ignore failure if branch doesn't exist yet
git "${AUTH_CFG[@]}" fetch origin "$BRANCH" || true

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
elif git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  git checkout -B "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Pull latest; ignore if nothing to pull
git "${AUTH_CFG[@]}" pull --ff-only origin "$BRANCH" || true

echo "SANDBOX_REPO_DIR=$(pwd)"
echo "SANDBOX_BRANCH=$BRANCH"
"""
    return textwrap.dedent(script)


async def initialize_e2b_sandbox(
    repo_url: str,
    branch_name: str = "main",
    github_token: Optional[str] = None,
    existing_sandbox_id: Optional[str] = None,

) -> tuple[str, str, str]:
    """
    Create (or resume) an E2B sandbox and ensure the GitHub repository is cloned
    and up to date on the requested branch.

    Returns: (sandbox_id, repo_dir, branch_in_sandbox)
    """
    if not os.getenv("E2B_API_KEY"):
        raise RuntimeError("E2B_API_KEY not configured in environment")

    logger.info("Creating E2B sandbox for codex...")
    sbx = await Sandbox.create()
    sandbox_id = sbx.id
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

    execution = await sbx.run_command(cmd, login_shell=False)

    exit_code = execution.exit_code
    stdout = execution.stdout
    stderr = execution.stderr

    if exit_code != 0:
        logger.error(_masked(f"Sandbox git setup error: {stderr or stdout}"))
        raise RuntimeError("Failed to prepare repository in sandbox")

    # Parse stdout lines to find our markers
    repo_dir = ""
    branch_in_sandbox = branch_name
    for line in stdout.splitlines():
        if line.startswith("SANDBOX_REPO_DIR="):
            repo_dir = line.split("=", 1)[1].strip()
        if line.startswith("SANDBOX_BRANCH="):
            branch_in_sandbox = line.split("=", 1)[1].strip()

    return str(sandbox_id), repo_dir, branch_in_sandbox




async def initialize_project(state: PlannerState) -> PlannerState:
    # If a remote repo URL is provided, initialize an E2B sandbox and clone/pull there.
    repo_url = state.get("repo_url")
    logger.info(f"State: {state}")
    logger.info(f"Initializing sandbox for repo_url: {repo_url}")
    if repo_url:
        branch_name = state.get("branch_name") or "main"
        github_token = os.getenv("GITHUB_TOKEN")
        existing_id = state.get("sandbox_session_id")

        sandbox_id, repo_dir, branch_in_sandbox = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
            existing_sandbox_id=existing_id,
        )

        messages = list(state.get("messages", []))
        messages.append({
            "role": "system",
            "content": f"E2B sandbox ready (id={sandbox_id}); repo cloned at {repo_dir} on branch {branch_in_sandbox}.",
        })

        new_state = dict(state)
        new_state["sandbox_session_id"] = sandbox_id
        # Store repo_dir in repo_path so downstream context actions have a path
        new_state["repo_path"] = repo_dir
        new_state["messages"] = messages
        return new_state
    return state



def _masked(s: str) -> str:
    token = os.getenv("GITHUB_TOKEN") or ""
    if not token:
        return s
    return s.replace(token, "***")

