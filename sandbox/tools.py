from langchain.tools import tool
from langchain.tools import ToolRuntime
from dataclasses import asdict

from sandbox.base import get_sandbox
from e2b_code_interpreter import CommandResult
from shared.logger import get_logger

logger = get_logger("sandbox.tools")

@tool
async def run_command_in_sandbox(command: str, runtime: ToolRuntime) -> str:
    """
    Run a command in the sandbox.
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")

    sbx = await get_sandbox(sandbox_id)
    try:
        res: CommandResult = await sbx.run_command(command, cwd=repo_path, login_shell=True)
    except Exception as e:
        logger.error(f"Error running command in sandbox: {e}")
        return f"Error: {e}"

    result = f"Exit code: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
    if hasattr(res, "error"):
        result += f"\nError: {res.error}"
    return result


