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
        timeout: Optional[int] = None,
    ) -> None:
        self._sbx = sandbox
        self._timeout = int(timeout) if timeout else None
        self._deadline_ts: Optional[float] = (
            (time.time() + self._timeout) if self._timeout else None
        )
        self._killer_thread: Optional[threading.Thread] = None
        if self._deadline_ts:
            self._killer_thread = threading.Thread(target=self._auto_kill_loop, daemon=True)
            self._killer_thread.start()

    # ---- Construction ----
    @classmethod
    async def create(
        cls,
        timeout: Optional[int] = 300,
        *,
        check_api_key: bool = True,
    ) -> "Sandbox":
        if check_api_key and not os.getenv("E2B_API_KEY"):
            raise RuntimeError("E2B_API_KEY not configured in environment")
        sbx = await E2BSandbox.beta_create(auto_pause=True,timeout=timeout)
        logger.info(f"E2B sandbox created: {getattr(sbx, 'sandbox_id', getattr(sbx, 'id', ''))}")
        return cls(sbx, timeout=timeout)

    @classmethod
    async def connect(
        cls,
        sandbox_id: str,
        timeout: Optional[int] = 300,
        *,
        check_api_key: bool = True,
    ) -> "Sandbox":
        if check_api_key and not os.getenv("E2B_API_KEY"):
            raise RuntimeError("E2B_API_KEY not configured in environment")
        # e2b-code-interpreter currently doesn't expose an official connect by id for python API.
        # Fall back to creating a new sandbox if connect is not available.
        try:
            if hasattr(E2BSandbox, "connect"):
                sbx = await E2BSandbox.connect(sandbox_id, timeout=timeout)  # type: ignore[attr-defined]
            else:
                logger.warning("Sandbox.connect not available; creating a new sandbox instead.")
                sbx = await E2BSandbox.beta_create(auto_pause=True,timeout=timeout)
        except Exception:
            # Safety: if connect fails, create a fresh sandbox
            sbx = await E2BSandbox.beta_create(auto_pause=True,timeout=timeout)
        return cls(sbx, timeout=timeout)

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

    async def pause(self) -> None:
        await self._sbx.beta_pause()

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


# ---- helpers moved from codex.shared.initialize_sandbox ----


async def get_sandbox(sandbox_id: str) -> Sandbox:
    sbx = await Sandbox.connect(sandbox_id)
    if not sbx:
        raise RuntimeError("Sandbox instance not available in current process")
    return sbx


async def cd_and_run_in_sandbox(sandbox_id: str, repo_dir: str, command: str) -> tuple[int, str, str]:
    sbx = await get_sandbox(sandbox_id)
    res = await sbx.run_command(command, cwd=repo_dir, login_shell=True)
    return res.exit_code, res.stdout, res.stderr

