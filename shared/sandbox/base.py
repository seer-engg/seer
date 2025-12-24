from __future__ import annotations

from e2b_code_interpreter import AsyncSandbox


from shared.logger import get_logger


logger = get_logger("sandbox")

# ---- helpers moved from codex.shared.initialize_sandbox ----


async def get_sandbox(sandbox_id: str) -> AsyncSandbox:
    sbx = await AsyncSandbox.connect(sandbox_id)
    if not sbx:
        raise RuntimeError("Sandbox instance not available in current process")
    return sbx


async def cd_and_run_in_sandbox(sandbox_id: str, repo_dir: str, command: str) -> tuple[int, str, str]:
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    res = await sbx.commands.run(command, cwd=repo_dir, login_shell=True)
    return res.exit_code, res.stdout, res.stderr

