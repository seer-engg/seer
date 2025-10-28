from __future__ import annotations

import base64
import os
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import textwrap

from e2b_code_interpreter import AsyncSandbox as E2BSandbox

from shared.logger import get_logger


logger = get_logger("sandbox")


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class Sandbox:
    """Thin wrapper around E2B Code Interpreter sandbox.

    Features:
    - create a new sandbox or connect to an existing one by id
    - optional time limit with auto-kill
    - run shell commands with cwd/env support
    - run python code (persisted in session per E2B semantics)
    - read/write files (text/binary)
    - list files, ensure directories, existence checks
    - lifecycle: kill, context-manager support
    """

    def __init__(
        self,
        sandbox: E2BSandbox,
        time_limit_seconds: Optional[int] = None,
    ) -> None:
        self._sbx = sandbox
        self._time_limit_seconds = int(time_limit_seconds) if time_limit_seconds else None
        self._deadline_ts: Optional[float] = (
            (time.time() + self._time_limit_seconds) if self._time_limit_seconds else None
        )
        self._killer_thread: Optional[threading.Thread] = None
        if self._deadline_ts:
            self._killer_thread = threading.Thread(target=self._auto_kill_loop, daemon=True)
            self._killer_thread.start()

    # ---- Construction ----
    @classmethod
    async def create(
        cls,
        time_limit_seconds: Optional[int] = None,
        *,
        check_api_key: bool = True,
    ) -> "Sandbox":
        if check_api_key and not os.getenv("E2B_API_KEY"):
            raise RuntimeError("E2B_API_KEY not configured in environment")
        sbx = await E2BSandbox.create()
        logger.info(f"E2B sandbox created: {getattr(sbx, 'sandbox_id', getattr(sbx, 'id', ''))}")
        return cls(sbx, time_limit_seconds=time_limit_seconds)

    @classmethod
    async def connect(
        cls,
        sandbox_id: str,
        time_limit_seconds: Optional[int] = None,
        *,
        check_api_key: bool = True,
    ) -> "Sandbox":
        if check_api_key and not os.getenv("E2B_API_KEY"):
            raise RuntimeError("E2B_API_KEY not configured in environment")
        # e2b-code-interpreter currently doesn't expose an official connect by id for python API.
        # Fall back to creating a new sandbox if connect is not available.
        try:
            if hasattr(E2BSandbox, "connect"):
                sbx = await E2BSandbox.connect(sandbox_id)  # type: ignore[attr-defined]
            else:
                logger.warning("Sandbox.connect not available; creating a new sandbox instead.")
                sbx = await E2BSandbox.create()
        except Exception:
            # Safety: if connect fails, create a fresh sandbox
            sbx = await E2BSandbox.create()
        return await cls(sbx, time_limit_seconds=time_limit_seconds)

    # ---- Properties ----
    @property
    def id(self) -> str:
        return getattr(self._sbx, "sandbox_id", getattr(self._sbx, "id", ""))

    # ---- Lifecycle ----
    async def kill(self) -> None:
        try:
            await self._sbx.kill()
        except Exception as e:
            logger.warning(f"Error killing sandbox {self.id}: {e}")

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.kill()

    # ---- Internals ----
    async def _auto_kill_loop(self) -> None:
        assert self._deadline_ts is not None
        while True:
            remaining = self._deadline_ts - time.time()
            if remaining <= 0:
                logger.info(f"Time limit reached; killing sandbox {self.id}")
                await self.kill()
                break
            time.sleep(min(remaining, 5.0))

    # ---- Exec helpers ----
    @staticmethod
    def _quote_env(env: Optional[Dict[str, str]]) -> str:
        if not env:
            return ""
        parts = []
        for k, v in env.items():
            if v is None:
                continue
            parts.append(f"{shlex.quote(str(k))}={shlex.quote(str(v))}")
        return " ".join(parts)

    async def run_command(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        login_shell: bool = True,
    ) -> CommandResult:
        env_prefix = self._quote_env(env)
        base_cmd = command if not login_shell else f"bash -lc {shlex.quote(command)}"
        if cwd:
            full_cmd = f"{env_prefix} cd {shlex.quote(cwd)} && {base_cmd}" if env_prefix else f"cd {shlex.quote(cwd)} && {base_cmd}"
        else:
            full_cmd = f"{env_prefix} {base_cmd}" if env_prefix else base_cmd

        exec_res = await self._sbx.commands.run(full_cmd)
        return CommandResult(
            exit_code=getattr(exec_res, "exit_code", 0),
            stdout=(getattr(exec_res, "stdout", "") or ""),
            stderr=(getattr(exec_res, "stderr", "") or ""),
        )

    async def run_code(self, code: str) -> Tuple[bool, str]:
        """Run Python code in the sandbox using the E2B persistent interpreter.

        Returns (success, output_or_error)
        """
        res = await self._sbx.run_code(code)
        if getattr(res, "error", None):
            return False, str(getattr(res, "error"))
        return True, getattr(res, "text", "") or ""

    # ---- File helpers ----
    async def write_text(self, path: str, content: str) -> None:
        data = content.encode()
        b64 = base64.b64encode(data).decode()
        cmd = f"base64 -d > {shlex.quote(path)} << 'B64EOF'\n{b64}\nB64EOF"
        res = await self.run_command(cmd, login_shell=False)
        if res.exit_code != 0:
            raise RuntimeError(f"write_text failed: {res.stderr or res.stdout}")

    async def write_bytes(self, path: str, content: bytes) -> None:
        b64 = base64.b64encode(content).decode()
        cmd = f"base64 -d > {shlex.quote(path)} << 'B64EOF'\n{b64}\nB64EOF"
        res = await self.run_command(cmd, login_shell=False)
        if res.exit_code != 0:
            raise RuntimeError(f"write_bytes failed: {res.stderr or res.stdout}")

    async def read_text(self, path: str) -> str:
        cmd = f"[ -f {shlex.quote(path)} ] && base64 < {shlex.quote(path)} || true"
        res = await self.run_command(cmd, login_shell=False)
        if res.exit_code != 0:
            raise RuntimeError(f"read_text failed: {res.stderr}")
        if not res.stdout:
            return ""
        try:
            return base64.b64decode(res.stdout).decode()
        except Exception:
            # Fallback plain read
            res2 = self.run_command(f"cat {shlex.quote(path)}", login_shell=False)
            return res2.stdout

    async def read_bytes(self, path: str) -> bytes:
        cmd = f"[ -f {shlex.quote(path)} ] && base64 < {shlex.quote(path)} || true"
        res = await self.run_command(cmd, login_shell=False)
        if res.exit_code != 0:
            raise RuntimeError(f"read_bytes failed: {res.stderr}")
        if not res.stdout:
            return b""
        return base64.b64decode(res.stdout)

    # ---- Misc utilities ----
    async def list_dir(self, path: str) -> str:
        res = await self.run_command(f"ls -la {shlex.quote(path)}", login_shell=False)
        return res.stdout or res.stderr

    async def mkdir_p(self, path: str) -> None:
        res = await self.run_command(f"mkdir -p {shlex.quote(path)}", login_shell=False)
        if res.exit_code != 0:
            raise RuntimeError(f"mkdir_p failed: {res.stderr or res.stdout}")

    async def exists(self, path: str) -> bool:
        res = await self.run_command(f"[ -e {shlex.quote(path)} ] && echo YES || echo NO", login_shell=False)
        return res.stdout.strip() == "YES"

    async def get_cwd(self) -> str:
        res = await self.run_command("pwd", login_shell=False)
        return res.stdout.strip()


# ---- Registry & helpers moved from codex.shared.initialize_sandbox ----
_SANDBOX_REGISTRY: dict[str, Sandbox] = {}


async def _register_sandbox(sbx: Sandbox) -> str:
    sid = sbx.id
    if not sid:
        raise RuntimeError("Unable to determine sandbox id")
    _SANDBOX_REGISTRY[sid] = sbx
    return sid


async def get_sandbox(sandbox_id: str) -> Sandbox:
    sbx = _SANDBOX_REGISTRY.get(str(sandbox_id) or "")
    if not sbx:
        raise RuntimeError("Sandbox instance not available in current process")
    return sbx


async def cd_and_run_in_sandbox(sandbox_id: str, repo_dir: str, command: str) -> tuple[int, str, str]:
    sbx = await get_sandbox(sandbox_id)
    res = await sbx.run_command(command, cwd=repo_dir, login_shell=True)
    return res.exit_code, res.stdout, res.stderr


async def ensure_sandbox_ready(repo_path: str) -> None:
    if not repo_path:
        raise ValueError("repo_path is required for local runs")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path not found: {repo_path}")


def _masked(s: str) -> str:
    token = os.getenv("GITHUB_TOKEN") or ""
    if not token:
        return s
    return s.replace(token, "***")


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
    sandbox_id = await _register_sandbox(sbx)
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

