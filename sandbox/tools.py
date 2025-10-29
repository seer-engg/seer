from langchain.tools import tool
from langchain.tools import ToolRuntime

from sandbox.base import get_sandbox
from e2b_code_interpreter import CommandResult, AsyncSandbox
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

    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
    except Exception as e:
        logger.error(f"Error running command in sandbox: {e}")
        return f"Error: {e}"

    result = f"Exit code: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
    if hasattr(res, "error"):
        result += f"\nError: {res.error}"
    return result


@tool
async def read_file_in_sandbox(file_path: str, runtime: ToolRuntime) -> str:
    """
    Read a file from the sandbox.
    
    Args:
        file_path: Path to the file relative to the repository root
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Construct full path
        full_path = f"{repo_path}/{file_path}" if not file_path.startswith("/") else file_path
        content = await sbx.files.read(full_path)
        return content
    except Exception as e:
        logger.error(f"Error reading file in sandbox: {e}")
        return f"Error reading file: {e}"


@tool
async def grep_in_sandbox(pattern: str, file_path: str, runtime: ToolRuntime) -> str:
    """
    Search for a pattern in files using grep in the sandbox.
    
    Args:
        pattern: The pattern to search for
        file_path: Path to file or directory to search in (relative to repo root). Use '.' for entire repo.
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Use grep with recursive search, line numbers, and ignore case option
        command = f"grep -rn '{pattern}' {file_path}"
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
        
        if res.exit_code == 0:
            return f"Matches found:\n{res.stdout}"
        elif res.exit_code == 1:
            return "No matches found"
        else:
            return f"Error (exit code {res.exit_code}):\n{res.stderr}"
    except Exception as e:
        logger.error(f"Error running grep in sandbox: {e}")
        return f"Error: {e}"


@tool
async def write_file_in_sandbox(file_path: str, content: str, runtime: ToolRuntime) -> str:
    """
    Write or overwrite a file in the sandbox.
    
    Args:
        file_path: Path to the file relative to the repository root
        content: The content to write to the file
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Construct full path
        full_path = f"{repo_path}/{file_path}" if not file_path.startswith("/") else file_path
        await sbx.files.write(full_path, content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        logger.error(f"Error writing file in sandbox: {e}")
        return f"Error writing file: {e}"


@tool
async def list_files_in_sandbox(directory_path: str, runtime: ToolRuntime) -> str:
    """
    List files and directories in the sandbox.
    
    Args:
        directory_path: Path to the directory relative to the repository root. Use '.' for repo root.
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Use ls -la for detailed listing
        command = f"ls -la {directory_path}"
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
        
        if res.exit_code == 0:
            return res.stdout
        else:
            return f"Error listing directory (exit code {res.exit_code}):\n{res.stderr}"
    except Exception as e:
        logger.error(f"Error listing files in sandbox: {e}")
        return f"Error: {e}"


@tool
async def create_file_in_sandbox(file_path: str, content: str = "", runtime: ToolRuntime = None) -> str:
    """
    Create a new file in the sandbox. Fails if file already exists.
    
    Args:
        file_path: Path to the file relative to the repository root
        content: The initial content to write to the file (optional, defaults to empty)
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Construct full path
        full_path = f"{repo_path}/{file_path}" if not file_path.startswith("/") else file_path
        
        # Check if file exists first
        check_cmd = f"test -f {full_path} && echo 'exists' || echo 'does not'"
        check_res: CommandResult = await sbx.commands.run(check_cmd, cwd=repo_path)
        
        if "exists" in check_res.stdout:
            return f"Error: File {file_path} already exists. Use write_file_in_sandbox to overwrite."
        
        # Create the file
        await sbx.files.write(full_path, content)
        return f"Successfully created file {file_path}"
    except Exception as e:
        logger.error(f"Error creating file in sandbox: {e}")
        return f"Error creating file: {e}"


@tool
async def create_directory_in_sandbox(directory_path: str, runtime: ToolRuntime) -> str:
    """
    Create a new directory in the sandbox. Creates parent directories as needed.
    
    Args:
        directory_path: Path to the directory relative to the repository root
    """
    sandbox_id = runtime.state.get("sandbox_session_id")
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime state")
    repo_path = runtime.state.get("repo_path")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime state")
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Use mkdir -p to create directory and parents
        command = f"mkdir -p {directory_path}"
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
        
        if res.exit_code == 0:
            return f"Successfully created directory {directory_path}"
        else:
            return f"Error creating directory (exit code {res.exit_code}):\n{res.stderr}"
    except Exception as e:
        logger.error(f"Error creating directory in sandbox: {e}")
        return f"Error creating directory: {e}"
