from langchain.tools import tool
from langchain.tools import ToolRuntime

from sandbox.base import get_sandbox
from e2b_code_interpreter import CommandResult, AsyncSandbox
from shared.logger import get_logger
from e2b import CommandExitException
from dataclasses import dataclass
from shared.schema import SandboxContext

logger = get_logger("sandbox.tools")

@dataclass
class SandboxToolContext:
    """Context for sandbox tools containing sandbox configuration."""
    sandbox_context: SandboxContext

def vaildate_sandbox_tool_call(runtime: ToolRuntime[SandboxToolContext]) -> tuple[str, str]:
    """
    Validate and extract sandbox context from runtime.
    
    Returns:
        tuple[str, str]: (sandbox_id, repo_path)
    """
    if not runtime.context:
        raise ValueError("Runtime context not found. Make sure context is passed when invoking the agent.")
    
    sandbox_context = runtime.context.sandbox_context
    sandbox_id = sandbox_context.sandbox_id
    repo_path = sandbox_context.working_directory
    
    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in runtime context")
    if not repo_path:
        raise ValueError("Repository directory not found in runtime context")
    
    return sandbox_id, repo_path

@tool
async def run_command(command: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Run a shell command in working directory of the repository.
    
    Args:
        command: The command to run
    
    Returns:
        The output of the command including exit code, stdout, and stderr
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)

    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
    except CommandExitException as e:
        logger.error(f"Error running command in sandbox: {e}")
        return f"Error: {e.stderr} \n {e.error} {e.stdout}"

    result = f"Exit code: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
    if hasattr(res, "error"):
        result += f"\nError: {res.error}"
    return result


@tool
async def read_file(file_path: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Read a file from the repository.

    Args:
        file_path: Path to the file relative to the repository root

    Returns:
        The content of the file, or an error message if the file does not exist
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
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
async def grep(pattern: str, file_path: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Search for a pattern in files using grep in the repository.

    Args:
        pattern: The pattern to search for
        file_path: Path to file or directory to search in (relative to repo root). Use '.' for entire repo.

    Returns:
        The output of the grep command, or an error message if the pattern is not found
    
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)

    try:   
        sbx: AsyncSandbox = await get_sandbox(sandbox_id)
            # Use grep with recursive search, line numbers, and ignore case option
        command = f"grep -rn '{pattern}' {file_path}"
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
        
        if res.exit_code == 0:
            return f"Matches found:\n{res.stdout}"
        elif res.exit_code == 1:
            return "No matches found"
        else:
            return f"Error (exit code {res.exit_code}):\n{res.stderr}"
    except CommandExitException as e:
        if e.exit_code == 1:
            return f"No matches found for pattern: {pattern} in file: {file_path}"
        else:
            return f"Error (exit code {e.exit_code}):\n{e.stderr}"
    


@tool
async def write_file(file_path: str, content: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Write or overwrite a file in the repository.
    
    Args:
        file_path: Path to the file relative to the repository root
        content: The content to write to the file

    Returns:
        A success message if the file was written, or an error message if the file could not be written
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
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
async def patch_file(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[SandboxToolContext],
    replace_all: bool 
) -> str:
    """
    Edit a specific portion of a file by replacing old_string with new_string.
    This is useful for making targeted edits without rewriting the entire file.
    
    Args:
        file_path: Path to the file relative to the repository root
        old_string: The exact text to find and replace (must match exactly including whitespace)
        new_string: The text to replace it with
        replace_all: If True, replace all occurrences. If False (default), replace only the first occurrence.

    Returns:
        A success message if the patch was applied, or an error message if:
        - The file does not exist
        - The old_string was not found in the file
        - The old_string appears multiple times and replace_all is False
        - There was an error writing the file
    
    Important:
        - The old_string must match EXACTLY including all whitespace and indentation
        - For safety, if old_string appears multiple times, the operation will fail unless replace_all=True
        - Provide enough context in old_string to make it unique within the file
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Construct full path
        full_path = f"{repo_path}/{file_path}" if not file_path.startswith("/") else file_path
        
        # Read the current file content
        try:
            content = await sbx.files.read(full_path)
        except Exception as e:
            return f"Error: Could not read file {file_path}. File may not exist. Error: {e}"
        
        # Check if old_string exists in the file
        if old_string not in content:
            return f"Error: The old_string was not found in {file_path}. Make sure it matches exactly including whitespace."
        
        # Count occurrences
        occurrence_count = content.count(old_string)
        
        if occurrence_count > 1 and not replace_all:
            return (
                f"Error: The old_string appears {occurrence_count} times in {file_path}. "
                f"To avoid ambiguity, either:\n"
                f"1. Provide more context in old_string to make it unique, or\n"
                f"2. Set replace_all=True to replace all occurrences"
            )
        
        # Perform the replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced_count = occurrence_count
        else:
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            replaced_count = 1
        
        # Write the modified content back
        await sbx.files.write(full_path, new_content)
        
        return f"Successfully patched {file_path}: replaced {replaced_count} occurrence(s)"
        
    except Exception as e:
        logger.error(f"Error patching file in sandbox: {e}")
        return f"Error patching file: {e}"


@tool
async def apply_patch(diff_content: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Apply a git-style unified diff to files in the repository.
    This is the most robust way to edit files as it uses git's patch mechanism.
    
    Args:
        diff_content: A unified diff string in standard git format. Should include:
            - File headers (--- a/path/to/file, +++ b/path/to/file)
            - Hunk headers (@@ -start,count +start,count @@)
            - Context lines (unchanged lines starting with space)
            - Removed lines (starting with -)
            - Added lines (starting with +)
    
    Returns:
        A success message if the diff was applied, or an error message with details about why it failed.
    
    Example diff format:
        --- a/src/example.py
        +++ b/src/example.py
        @@ -1,3 +1,3 @@
         def hello():
        -    print("old")
        +    print("new")
             return True
    
    Tips:
        - Include enough context lines (unchanged lines) around changes for accurate matching
        - The diff will fail if the file content doesn't match the expected state
        - You can include multiple file changes in a single diff
        - Use `git diff` format which is standard and well-tested
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Create a temporary patch file
        import uuid
        patch_filename = f"/tmp/patch_{uuid.uuid4().hex}.diff"
        
        # Write the diff content to the patch file
        await sbx.files.write(patch_filename, diff_content)
        
        # Try to apply the patch using git apply
        # --verbose for detailed output
        # --whitespace=nowarn to be lenient with whitespace
        apply_cmd = f"git apply --verbose {patch_filename}"
        res: CommandResult = await sbx.commands.run(apply_cmd, cwd=repo_path)
        
        # Clean up the temporary patch file
        cleanup_cmd = f"rm {patch_filename}"
        await sbx.commands.run(cleanup_cmd, cwd=repo_path)
        
        if res.exit_code == 0:
            return f"Successfully applied diff\nOutput: {res.stdout}"
        else:
            # Try to provide helpful error message
            error_msg = f"Failed to apply diff (exit code {res.exit_code})\n"
            error_msg += f"Stderr: {res.stderr}\n"
            error_msg += f"Stdout: {res.stdout}\n\n"
            error_msg += "Common issues:\n"
            error_msg += "- The file content doesn't match the diff context (file may have been modified)\n"
            error_msg += "- Line numbers in the diff don't match current file\n"
            error_msg += "- File path in diff doesn't match actual file path\n"
            error_msg += "- Diff format is incorrect\n\n"
            error_msg += "Try: Read the file first to see current content, then create a matching diff"
            return error_msg
            
    except Exception as e:
        logger.error(f"Error applying diff in sandbox: {e}")
        return f"Error applying diff: {e}"


@tool
async def inspect_directory(directory_path: str, runtime: ToolRuntime[SandboxToolContext], depth: int ) -> str:
    """
    List files and directories in the repository.
    
    Args:
        directory_path: Path to the directory relative to the repository root. Use '.' for repo root.
        depth: Depth of the directory tree to inspect.

    Returns:
        A tree of the directory structure including files and directories, or an error message if the directory could not be inspected
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)

    if depth is None:
        depth = 2
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Use ls -la for detailed listing
        command = f"tree -a -L {depth} -I 'node_modules|.git|.venv|dist|build|__pycache__' -F {directory_path} "
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
        
        if res.exit_code == 0:
            return res.stdout
        else:
            return f"Error listing directory (exit code {res.exit_code}):\n{res.stderr}"
    except Exception as e:
        logger.error(f"Error listing files in sandbox: {e}")
        return f"Error: {e}"


@tool
async def create_file(file_path: str, content: str , runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Create a new file in the repository. Fails if file already exists.
    
    Args:
        file_path: Path to the file relative to the repository root
        content: The initial content to write to the file

    Returns:
        A success message if the file was created, or an error message if the file could not be created
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
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
async def create_directory(directory_path: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Create a new directory in the repository. Creates parent directories as needed.
    
    Args:
        directory_path: Path to the directory relative to the repository root

    Returns:
        A success message if the directory was created, or an error message if the directory could not be created
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
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
