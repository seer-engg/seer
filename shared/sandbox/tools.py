from langchain.tools import tool
from langchain.tools import ToolRuntime

from shared.sandbox.base import get_sandbox
from e2b_code_interpreter import CommandResult, AsyncSandbox
from shared.logger import get_logger
from e2b import CommandExitException
from dataclasses import dataclass
from shared.schema import SandboxContext
from shared.indexer.service import get_index_service
import os 

logger = get_logger("sandbox.tools")

def _summarize_stream(stream: str, label: str, head: int = 10, tail: int = 10) -> str:
    """
    Return a truncated summary of long command output to keep tool responses concise.

    Args:
        stream: Full stdout/stderr string.
        label: Label to include in the formatted output.
        head: Number of lines to include from the start of the stream.
        tail: Number of lines to include from the end of the stream.
    """
    safe_stream = stream or ""
    lines = safe_stream.splitlines()
    total = len(lines)

    if total == 0:
        return f"{label}: <empty>"

    if total <= head + tail:
        header = f"{label} (total {total} lines):"
        body = "\n".join(lines)
    else:
        omitted = total - head - tail
        header = (
            f"{label} (showing first {head} + last {tail} of {total} lines; {omitted} omitted):"
        )
        body = "\n".join(lines[:head] + ["..."] + lines[-tail:])
    return f"{header}\n{body}"

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
async def run_command(command: str,context: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Run a shell command in working directory of the repository. return truncated output of the command [first 10 + last 10 lines] including exit code, stdout, and stderr
    
    Args:
        command: The command to run
        context: The context of the command ( Why you are running this command)
    Returns:
        The truncated output of the command [first 10 + last 10 lines] including exit code, stdout, and stderr
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)

    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)
    except CommandExitException as e:
        logger.error(f"Error running command in sandbox: {e}")
        stdout_summary = _summarize_stream(getattr(e, "stdout", ""), "Stdout")
        stderr_summary = _summarize_stream(getattr(e, "stderr", ""), "Stderr")
        details = getattr(e, "error", "")
        return (
            f"Command failed with exit code {getattr(e, 'exit_code', 'unknown')}:\n"
            f"{stdout_summary}\n{stderr_summary}\nDetails: {details}"
        )

    stdout_summary = _summarize_stream(res.stdout, "Stdout")
    stderr_summary = _summarize_stream(res.stderr, "Stderr")
    result = f"Exit code: {res.exit_code}\n{stdout_summary}\n{stderr_summary}"
    if hasattr(res, "error") and res.error:
        result += f"\n{_summarize_stream(res.error, 'Error')}"
    return result


@tool
async def read_files(file_paths: list[str], runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Read multiple files from the repository.

    Args:
        file_paths: List of paths to the files relative to the repository root

    Returns:
        The content of the files, or an error message if the files do not exist
    """
    sandbox_id, repo_path = vaildate_sandbox_tool_call(runtime)
    
    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    contents = []
    for file_path in file_paths:
        try:
            # Construct full path
            full_path = os.path.join(repo_path, file_path)
            content = await sbx.files.read(full_path)
            contents.append(f"{file_path}:\n{content}")
        except Exception as e:
            logger.error(f"Error reading file in sandbox: {e}")
            return f"Error reading file: {e}"
    return "\n".join(contents)


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
        # Update index for this file
        try:
            service = get_index_service()
            await service.update_files(runtime.context.sandbox_context, [file_path])
        except Exception as e:
            logger.warning(f"Index update failed after write_file: {e}")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        logger.error(f"Error writing file in sandbox: {e}")
        return f"Error writing file: {e}"


@tool
async def edit_file(
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
        replace_all: If True, replace all occurrences. If False, replace only the first occurrence.

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
        # Update index
        try:
            service = get_index_service()
            await service.update_files(runtime.context.sandbox_context, [file_path])
        except Exception as e:
            logger.warning(f"Index update failed after patch_file: {e}")
        
        return f"Successfully patched {file_path}: replaced {replaced_count} occurrence(s)"
        
    except Exception as e:
        logger.error(f"Error patching file in sandbox: {e}")
        return f"Error patching file: {e}"


async def _inspect_directory_impl(
    directory_path: str, sandbox_context: SandboxContext, depth: int = 2
) -> str:
    """Core implementation for inspecting a directory."""
    sandbox_id = sandbox_context.sandbox_id
    repo_path = sandbox_context.working_directory

    if not sandbox_id:
        raise ValueError("Sandbox session ID not found in context")
    if not repo_path:
        raise ValueError("Repository directory not found in context")

    sbx: AsyncSandbox = await get_sandbox(sandbox_id)
    try:
        # Use tree for detailed listing
        command = f"tree -a -L {depth} -I 'node_modules|.git|.venv|dist|build|__pycache__' -F {directory_path}"
        res: CommandResult = await sbx.commands.run(command, cwd=repo_path)

        if res.exit_code == 0:
            return res.stdout
        else:
            return (
                f"Error listing directory (exit code {res.exit_code}):\n{res.stderr}"
            )
    except Exception as e:
        logger.error(f"Error listing files in sandbox: {e}")
        return f"Error: {e}"


@tool
async def inspect_directory(
    directory_path: str, runtime: ToolRuntime[SandboxToolContext], depth: int = 2
) -> str:
    """
    List files and directories in the repository.
    
    Args:
        directory_path: Path to the directory relative to the repository root. Use '.' for repo root.
        depth: Depth of the directory tree to inspect.

    Returns:
        A tree of the directory structure including files and directories, or an error message if the directory could not be inspected
    """
    if not runtime.context:
        raise ValueError(
            "Runtime context not found. Make sure context is passed when invoking the agent."
        )

    return await _inspect_directory_impl(
        directory_path=directory_path,
        sandbox_context=runtime.context.sandbox_context,
        depth=depth,
    )


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
        # Update index
        try:
            service = get_index_service()
            await service.update_files(runtime.context.sandbox_context, [file_path])
        except Exception as e:
            logger.warning(f"Index update failed after create_file: {e}")
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



@tool
async def search_code(query: str, top_k: int, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Lexical search over code chunks using FTS; returns file paths, line ranges, and snippets.
    """
    if top_k is None:
        top_k = 10
    try:
        service = get_index_service()
        results = await service.search_code_lexical(query, k=top_k)
        if not results:
            return "No results"
        lines = []
        for r in results:
            lines.append(f"{r['path']}:{r['start_line']}-{r['end_line']} :: {r.get('snippet','')}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error lexical searching code: {e}")
        return f"Error searching code: {e}"


@tool
async def search_symbols(query: str, top_k: int, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Search symbols (functions/classes/methods) by name or docstring. Returns definitions and line ranges.
    """
    if top_k is None:
        top_k = 20
    try:
        service = get_index_service()
        results = await service.search_symbols(query, k=top_k)
        if not results:
            return "No symbols found"
        lines = []
        for r in results:
            lines.append(f"{r['type']} {r['qualname']} @ {r['path']}:{r['lineno']}-{r['end_lineno']}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        return f"Error searching symbols: {e}"


@tool
async def semantic_search(query: str, top_k: int, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Semantic search over code chunks using embeddings. Returns top-k chunks with similarity scores.
    """
    if top_k is None:
        top_k = 10
    try:
        service = get_index_service()
        results = await service.semantic_search(query, k=top_k)
        if not results:
            return "No results"
        lines = []
        for r in results:
            score = f"{r['score']:.3f}"
            lines.append(f"{r['path']}:{r['start_line']}-{r['end_line']} :: score={score}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error semantic searching code: {e}")
        return f"Error semantic searching code: {e}"


@tool
async def get_symbol_definition(qualname: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Fetch the file path and line range for a fully-qualified symbol name.
    """
    try:
        service = get_index_service()
        res = await service.get_symbol_definition(qualname)
        if not res:
            return "Symbol not found"
        return f"{res['type']} {qualname} @ {res['path']}:{res['lineno']}-{res['end_lineno']}"
    except Exception as e:
        logger.error(f"Error getting symbol definition: {e}")
        return f"Error getting symbol definition: {e}"


@tool
async def find_usages(symbol: str, top_k: int, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Find approximate call sites/usages for a symbol (by name or qualname).
    """
    if top_k is None:
        top_k = 50
    try:
        service = get_index_service()
        results = await service.find_usages(symbol, k=top_k)
        if not results:
            return "No usages found"
        lines = []
        for r in results:
            lines.append(f"{r['path']}:{r['lineno']} :: {r['call_type']}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error finding usages: {e}")
        return f"Error finding usages: {e}"


@tool
async def get_code_region(path: str, start_line: int, end_line: int, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Return the exact text for a file region by line numbers using the index's chunk store when available.
    """
    try:
        service = get_index_service()
        content = await service.get_file_region(path, start_line, end_line)
        if content is None:
            return "Region not found in index"
        return content
    except Exception as e:
        logger.error(f"Error getting code region: {e}")
        return f"Error getting code region: {e}"
