from __future__ import annotations

import os
import textwrap
from typing import Optional
import base64
import shlex

from e2b_code_interpreter import Sandbox

from shared.logger import get_logger


logger = get_logger("codex.sandbox")


_SANDBOX_REGISTRY: dict[str, Sandbox] = {}


def _sandbox_id(sbx: Sandbox) -> str:
    return getattr(sbx, "sandbox_id", None) or getattr(sbx, "id", None) or ""


def _register_sandbox(sbx: Sandbox) -> str:
    sid = _sandbox_id(sbx)
    if not sid:
        raise RuntimeError("Unable to determine sandbox id")
    _SANDBOX_REGISTRY[sid] = sbx
    return sid


def get_sandbox(sandbox_id: str) -> Sandbox:
    sbx = _SANDBOX_REGISTRY.get(str(sandbox_id) or "")
    if not sbx:
        raise RuntimeError("Sandbox instance not available in current process")
    return sbx


def cd_and_run_in_sandbox(sandbox_id: str, repo_dir: str, command: str) -> tuple[int, str, str]:
    sbx = get_sandbox(sandbox_id)
    cmd = f"cd {shlex.quote(repo_dir)} && bash -lc {shlex.quote(command)}"
    res = sbx.commands.run(cmd)
    return (
        getattr(res, "exit_code", 0),
        getattr(res, "stdout", "") or "",
        getattr(res, "stderr", "") or "",
    )


def ensure_sandbox_ready(repo_path: str) -> None:
    if not repo_path:
        raise ValueError("repo_path is required for local runs")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path not found: {repo_path}")


def _masked(s: str) -> str:
    token = os.getenv("GITHUB_TOKEN") or ""
    if not token:
        return s
    return s.replace(token, "***")


def _build_git_shell_script() -> str:
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


def initialize_e2b_sandbox(
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
    sandbox = Sandbox.create()
    sandbox_id = _register_sandbox(sandbox)
    logger.info(f"Sandbox created: {sandbox_id}")

    shell_script = _build_git_shell_script()
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

    execution = sandbox.commands.run(cmd)

    exit_code = getattr(execution, "exit_code", 0)
    stdout = getattr(execution, "stdout", "") or ""
    stderr = getattr(execution, "stderr", "") or ""

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
