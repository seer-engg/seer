#!/usr/bin/env python3
"""
CLI for the Seer Eval Agent and Supervisor Agent.

Usage:
    seer-eval run                    # Start interactive eval agent loop
    seer-eval run --thread-id <uuid> # Resume existing thread
    seer-eval new-supervisor         # Start interactive supervisor chat
"""
import asyncio
import base64
import json
import mimetypes
import sys
import uuid
import traceback
from pathlib import Path
from typing import Optional, Any, List

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from rich.prompt import Prompt

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Load .env before importing seer modules
load_dotenv()

console = Console()

# Global verbose flag
VERBOSE = False


def process_file_input(user_input: str) -> Optional[str]:
    """
    Process file path(s) from user input and encode them as base64.
    
    Returns a JSON string with file data if files were successfully read,
    or None if no valid files were found.
    """
    # Parse potential file paths (space-separated, or single path)
    potential_paths = user_input.strip().split()
    
    files_data: List[dict] = []
    
    for path_str in potential_paths:
        # Expand ~ and resolve path
        file_path = Path(path_str).expanduser().resolve()
        
        if not file_path.exists():
            console.print(f"[yellow]Warning: File not found: {path_str}[/yellow]")
            continue
        
        if not file_path.is_file():
            console.print(f"[yellow]Warning: Not a file: {path_str}[/yellow]")
            continue
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            # Default based on extension
            ext = file_path.suffix.lower()
            mime_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.json': 'application/json',
                '.csv': 'text/csv',
            }
            mime_type = mime_map.get(ext, 'application/octet-stream')
        
        try:
            # Read and encode file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Check file size (limit to 10MB)
            if len(file_data) > 10 * 1024 * 1024:
                console.print(f"[yellow]Warning: File too large (>10MB): {path_str}[/yellow]")
                continue
            
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            
            files_data.append({
                'filename': file_path.name,
                'content_type': mime_type,
                'data': encoded_data,
                'path': str(file_path),
            })
            
            console.print(f"[green]✓ Loaded file: {file_path.name} ({mime_type})[/green]")
            
        except Exception as e:
            console.print(f"[red]Error reading file {path_str}: {e}[/red]")
            continue
    
    if files_data:
        return json.dumps({
            'type': 'file_upload',
            'files': files_data,
            'message': f"Uploaded {len(files_data)} file(s)",
        })
    
    return None


async def handle_interrupts(
    graph,
    config: RunnableConfig,
    progress=None,
    task=None,
) -> dict:
    """
    Handle LangGraph interrupts in a loop.
    
    When the graph encounters an interrupt(), it pauses and returns.
    This function checks for pending interrupts, prompts the user,
    and resumes the graph until completion.
    
    Returns the final state when no more interrupts are pending.
    """
    while True:
        # Check the current state for interrupts
        state = graph.get_state(config)
        
        # Check if there are pending interrupts
        # Interrupts are stored in state.tasks with interrupts field
        has_interrupt = False
        interrupt_payload = None
        
        for task_state in state.tasks:
            if hasattr(task_state, 'interrupts') and task_state.interrupts:
                has_interrupt = True
                # Get the first interrupt payload
                interrupt_payload = task_state.interrupts[0].value if task_state.interrupts else None
                break
        
        if not has_interrupt:
            # No more interrupts, return the final values
            return state.values
        
        # Stop the progress spinner while prompting
        # Note: task can be 0 (first task ID), so check `is not None`
        if progress is not None and task is not None:
            progress.stop()
        
        # Display the interrupt to the user
        if interrupt_payload and isinstance(interrupt_payload, dict):
            interrupt_type = interrupt_payload.get('type', 'input_required')
            
            if interrupt_type == 'missing_config':
                field = interrupt_payload.get('field', 'unknown')
                env_var = interrupt_payload.get('env_var', field.upper())
                instructions = interrupt_payload.get('instructions', f'Please provide a value for {field}')
                
                console.print(f"\n[bold yellow]⚠ Missing Configuration[/bold yellow]")
                console.print(Panel(
                    f"[bold]{env_var}[/bold] is required but not set.\n\n"
                    f"[dim]{instructions}[/dim]",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
                console.print()  # Extra newline for visibility
                
                user_input = Prompt.ask(
                    f"[bold]Enter value for [cyan]{env_var}[/cyan][/bold] (or 'exit' to quit)",
                    console=console
                )
                console.print()  # Newline after input
            else:
                # Generic interrupt
                console.print(f"\n[bold yellow]Input Required[/bold yellow]")
                if isinstance(interrupt_payload, dict):
                    console.print(Panel(
                        json.dumps(interrupt_payload, indent=2, default=str),
                        border_style="yellow",
                        box=box.ROUNDED
                    ))
                else:
                    console.print(f"[dim]{interrupt_payload}[/dim]")
                
                user_input = Prompt.ask("Your response (or 'exit' to quit)", console=console)
        else:
            # Unknown interrupt format
            console.print(f"\n[bold yellow]Input Required[/bold yellow]")
            console.print(f"[dim]Payload: {interrupt_payload}[/dim]")
            user_input = Prompt.ask("Your response (or 'exit' to quit)", console=console)
        
        # Check for exit
        if user_input.lower() in ('exit', 'quit', 'stop', 'cancel'):
            console.print("[yellow]Exiting...[/yellow]")
            return state.values
        
        # Restart progress if it was running
        if progress is not None and task is not None:
            progress.start()
        
        # Resume the graph with the user's input
        try:
            await graph.ainvoke(Command(resume=user_input), config=config)
        except Exception as e:
            if VERBOSE:
                console.print(f"\n[bold red]Error during resume:[/bold red] {e}")
                console.print(traceback.format_exc())
            raise


# Lazy imports to avoid import errors when displaying help
def get_graph():
    """Lazy import of the eval agent graph."""
    from agents.eval_agent.graph import build_graph
    return build_graph()


def create_compiled_graph(checkpointer=None):
    """Create and compile the eval agent graph."""
    graph = get_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def get_supervisor_graph():
    """Lazy import of the supervisor agent graph."""
    from agents.supervisor.graph import build_graph
    return build_graph()


def create_supervisor_compiled_graph(checkpointer=None):
    """Create and compile the supervisor agent graph."""
    graph = get_supervisor_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def format_context(context) -> str:
    """Format agent context for display."""
    lines = []
    if hasattr(context, 'agent_name') and context.agent_name:
        lines.append(f"**Agent Name:** {context.agent_name}")
    if hasattr(context, 'mcp_services') and context.mcp_services:
        lines.append(f"**MCP Services:** {', '.join(context.mcp_services)}")
    if hasattr(context, 'functional_requirements') and context.functional_requirements:
        lines.append("\n**Functional Requirements:**")
        for i, req in enumerate(context.functional_requirements, 1):
            lines.append(f"  {i}. {req}")
    return '\n'.join(lines)


def format_dataset_example(example) -> str:
    """Format a dataset example for display."""
    return example.to_markdown()


@click.group()
@click.version_option(version="0.1.4", prog_name="seer-eval")
@click.option('--verbose', '-v', is_flag=True, help='Show full error tracebacks for debugging')
def cli(verbose: bool):
    """
    🔮 Seer Eval Agent CLI
    
    Evaluate AI agents through automated test generation and execution.
    
    \b
    Commands:
      run            - Start interactive eval agent loop (continuous)
      new-supervisor - Start interactive supervisor chat for database operations
    
    \b
    Examples:
      # Start the interactive eval agent loop
      seer-eval run
      
      # Resume an existing thread
      seer-eval run --thread-id <uuid>
      
      # Start a supervisor chat session
      seer-eval new-supervisor
      
      # Debug with full tracebacks
      seer-eval -v run
    """
    global VERBOSE
    VERBOSE = verbose


@cli.command()
@click.option('--thread-id', '-t', default=None, help='Thread ID (auto-generated if not provided)')
def run(thread_id: Optional[str]):
    """
    Start an interactive eval agent loop.
    
    This is a continuous agent loop that runs until you type 'exit'.
    Each iteration asks for a step (alignment, plan, testing, finalize).
    
    \b
    Steps:
      alignment - Align agent expectations with user requirements
                  (prompts for description, repo, and user-id)
      plan      - Generate test cases based on aligned expectations
      testing   - Execute generated test cases
      finalize  - Finalize the evaluation
    
    \b
    Commands during session:
      exit, quit, bye  - Exit the session
      clear            - Clear the screen
    
    \b
    Example:
      seer-eval run
      seer-eval run --thread-id <uuid>  # Resume existing thread
    """
    asyncio.run(_run(thread_id))


async def _run(thread_id: Optional[str]):
    """Async implementation of run command (continuous agent loop)."""
    thread_id = thread_id or str(uuid.uuid4())
    
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Eval Agent - Interactive Loop[/bold cyan]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]\n"
        "[dim]Type 'exit' to quit, 'clear' to clear screen[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    # Create the eval agent with checkpointer
    memory = MemorySaver()
    eval_agent = create_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    valid_steps = ['alignment', 'plan', 'testing', 'finalize']
    
    # Continuous agent loop
    while True:
        try:
            # Ask for step (mandatory)
            console.print("[bold]Available steps:[/bold] alignment, plan, testing, finalize")
            step = Prompt.ask(
                "[bold cyan]Select step[/bold cyan]",
                console=console,
                choices=valid_steps + ['exit', 'quit', 'bye', 'clear'],
                show_choices=False
            )
            
            # Handle special commands
            if step.lower().strip() in ('exit', 'quit', 'bye', 'q'):
                console.print("\n[cyan]👋 Goodbye![/cyan]")
                break
            
            if step.lower().strip() == 'clear':
                console.clear()
                console.print(Panel.fit(
                    "[bold cyan]🔮 Seer Eval Agent - Interactive Loop[/bold cyan]\n\n"
                    f"[dim]Thread ID: {thread_id}[/dim]\n"
                    "[dim]Type 'exit' to quit, 'clear' to clear screen[/dim]",
                    border_style="cyan"
                ))
                continue
            
            # Prepare inputs based on step
            inputs = {"step": step}
            
            # For alignment step, ask for additional inputs
            if step == 'alignment':
                console.print("\n[bold]Alignment Configuration[/bold]")
                console.print("─" * 40)
                
                description = Prompt.ask(
                    "[bold]Description[/bold] (what does your agent do?)",
                    console=console
                )
                if description.lower().strip() in ('exit', 'quit', 'bye'):
                    console.print("\n[cyan]👋 Goodbye![/cyan]")
                    break
                
                repo = Prompt.ask(
                    "[bold]GitHub Repository[/bold] (owner/repo format)",
                    console=console
                )
                if repo.lower().strip() in ('exit', 'quit', 'bye'):
                    console.print("\n[cyan]👋 Goodbye![/cyan]")
                    break
                
                user_id = Prompt.ask(
                    "[bold]User ID[/bold] (for authentication context, press Enter to skip)",
                    console=console,
                    default=""
                )
                if user_id.lower().strip() in ('exit', 'quit', 'bye'):
                    console.print("\n[cyan]👋 Goodbye![/cyan]")
                    break
                
                inputs["messages"] = [{"type": "human", "content": description}]
                inputs["input_context"] = {
                    "integrations": {
                        "github": {"name": repo}
                    }
                }
                if user_id.strip():
                    inputs["input_context"]["user_id"] = user_id.strip()
                
                console.print()
                console.print(f"[dim]Description: {description}[/dim]")
                console.print(f"[dim]Repository: {repo}[/dim]")
                if user_id.strip():
                    console.print(f"[dim]User ID: {user_id}[/dim]")
            
            console.print()
            console.print(f"[bold]Running step: {step}[/bold]")
            console.print("─" * 40)
            
            # Run the eval agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                prog_task = progress.add_task(f"Running {step}...", total=None)
                
                try:
                    # Initial invocation
                    await eval_agent.ainvoke(inputs, config=runnable_config)
                    
                    # Handle any interrupts
                    results = await handle_interrupts(
                        eval_agent,
                        runnable_config,
                        progress=progress,
                        task=prog_task,
                    )
                    progress.update(prog_task, completed=True)
                except KeyboardInterrupt:
                    progress.stop()
                    console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    progress.stop()
                    console.print(f"\n[bold red]Error in {step}:[/bold red] {e}")
                    if VERBOSE:
                        console.print("\n[bold red]Full Traceback:[/bold red]")
                        console.print(traceback.format_exc())
                    else:
                        console.print("[dim]Use -v flag for full traceback: seer-eval -v run[/dim]")
                    continue
            
            # Display results based on step
            console.print()
            _display_step_results(step, results)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except EOFError:
            console.print("\n[cyan]👋 Goodbye![/cyan]")
            break


def _display_step_results(step: str, results: dict):
    """Display results based on the step that was run."""
    if step == 'alignment':
        context = results.get('context')
        if context:
            # Build display content
            content_lines = [
                f"[bold]Agent:[/bold] {context.agent_name if hasattr(context, 'agent_name') else 'Unknown'}",
                f"[bold]Services:[/bold] {', '.join(context.mcp_services) if hasattr(context, 'mcp_services') and context.mcp_services else 'None'}"
            ]
            
            # Add functional requirements if available
            if hasattr(context, 'functional_requirements') and context.functional_requirements:
                content_lines.append("")
                content_lines.append("[bold]Functional Requirements:[/bold]")
                for i, req in enumerate(context.functional_requirements, 1):
                    content_lines.append(f"  {i}. {req}")
            
            console.print(Panel(
                "\n".join(content_lines),
                title="[bold green]✓ Alignment Complete[/bold green]",
                border_style="green",
                box=box.ROUNDED
            ))
        else:
            console.print("[green]✓ Alignment step completed[/green]")
    
    elif step == 'plan':
        examples = results.get('dataset_examples', [])
        console.print(f"[green]✓[/green] Generated {len(examples)} test case(s)")
        
        if examples:
            for example in examples:
                console.print(Panel(
                    Markdown(format_dataset_example(example)),
                    border_style="blue",
                    box=box.ROUNDED
                ))
    
    elif step == 'testing':
        latest_results = results.get('latest_results', [])
        passed = sum(1 for r in latest_results if hasattr(r, 'passed') and r.passed)
        total = len(latest_results)
        
        if total > 0:
            status_color = "green" if passed == total else "yellow" if passed > 0 else "red"
            console.print(Panel(
                f"[bold]Tests Passed:[/bold] {passed}/{total}",
                title=f"[bold {status_color}]Testing Results[/bold {status_color}]",
                border_style=status_color,
                box=box.ROUNDED
            ))
        else:
            console.print("[green]✓ Testing step completed[/green]")
        
        # Check for missing config
        missing_config = results.get('missing_config', [])
        if missing_config:
            console.print(f"\n[bold yellow]⚠ Missing Configuration:[/bold yellow]")
            for config_key in missing_config:
                console.print(f"  • {config_key}")
    
    elif step == 'finalize':
        console.print("[green]✓ Finalize step completed[/green]")
        
        # Display any final summary if available
        if results.get('summary'):
            console.print(Panel(
                Markdown(results['summary']),
                title="[bold green]Evaluation Summary[/bold green]",
                border_style="green",
                box=box.ROUNDED
            ))
    
    else:
        console.print(f"[green]✓ Step '{step}' completed[/green]")


@cli.command('new-supervisor')
@click.option('--thread-id', '-t', default=None, help='Thread ID (auto-generated if not provided)')
@click.option('--db-uri', '-d', default=None, help='PostgreSQL connection URI (uses DATABASE_URI env var if not provided)')
def new_supervisor(thread_id: Optional[str], db_uri: Optional[str]):
    """
    Start an interactive chat session with the Supervisor agent.
    
    The Supervisor agent can help with database operations, schema exploration,
    and other PostgreSQL-related tasks.
    
    \b
    Commands during chat:
      exit, quit, bye  - Exit the chat session
      clear            - Clear the screen
    
    \b
    Example:
      seer-eval new-supervisor
      seer-eval new-supervisor --db-uri "postgresql://user:pass@host/db"
    """
    asyncio.run(_new_supervisor(thread_id, db_uri))


async def _new_supervisor(thread_id: Optional[str], db_uri: Optional[str]):
    """Async implementation of new-supervisor command."""
    thread_id = thread_id or str(uuid.uuid4())
    
    console.print(Panel.fit(
        "[bold magenta]🤖 Seer Supervisor Agent[/bold magenta]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]\n"
        "[dim]Type 'exit' to quit, 'clear' to clear screen[/dim]",
        border_style="magenta"
    ))
    
    if db_uri:
        console.print(f"[dim]Database: Connected via provided URI[/dim]")
    else:
        console.print(f"[dim]Database: Using DATABASE_URI from environment[/dim]")
    console.print()
    
    # Create the supervisor agent with checkpointer
    memory = MemorySaver()
    supervisor_agent = create_supervisor_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]", console=console)
            
            # Handle special commands
            if user_input.lower().strip() in ('exit', 'quit', 'bye', 'q'):
                console.print("\n[magenta]👋 Goodbye![/magenta]")
                break
            
            if user_input.lower().strip() == 'clear':
                console.clear()
                console.print(Panel.fit(
                    "[bold magenta]🤖 Seer Supervisor Agent[/bold magenta]\n\n"
                    f"[dim]Thread ID: {thread_id}[/dim]\n"
                    "[dim]Type 'exit' to quit, 'clear' to clear screen[/dim]",
                    border_style="magenta"
                ))
                continue
            
            if not user_input.strip():
                continue
            
            # Build input for the supervisor
            inputs = {
                "messages": [{"type": "human", "content": user_input}],
            }
            if db_uri:
                inputs["database_connection_string"] = db_uri
            
            # Run the supervisor agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                prog_task = progress.add_task("Thinking...", total=None)
                
                try:
                    # Initial invocation
                    await supervisor_agent.ainvoke(inputs, config=runnable_config)
                    
                    # Handle any interrupts (file requests, etc.)
                    results = await handle_supervisor_interrupts(
                        supervisor_agent,
                        runnable_config,
                        progress=progress,
                        task=prog_task,
                    )
                    progress.update(prog_task, completed=True)
                except KeyboardInterrupt:
                    progress.stop()
                    console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    progress.stop()
                    console.print(f"\n[bold red]Error:[/bold red] {e}")
                    if VERBOSE:
                        console.print("\n[bold red]Full Traceback:[/bold red]")
                        console.print(traceback.format_exc())
                    continue
            
            # Display the response
            response = results.get('response')
            if response:
                console.print()
                console.print(Panel(
                    Markdown(response),
                    title="[bold magenta]Supervisor[/bold magenta]",
                    border_style="magenta",
                    box=box.ROUNDED
                ))
                console.print()
            else:
                console.print("[dim]No response received.[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except EOFError:
            console.print("\n[magenta]👋 Goodbye![/magenta]")
            break


async def handle_supervisor_interrupts(
    graph,
    config: RunnableConfig,
    progress=None,
    task=None,
) -> dict:
    """
    Handle LangGraph interrupts for the supervisor agent.
    
    Handles file request interrupts and other supervisor-specific interrupts.
    Returns the final state when no more interrupts are pending.
    """
    while True:
        # Check the current state for interrupts
        state = graph.get_state(config)
        
        # Check if there are pending interrupts
        has_interrupt = False
        interrupt_payload = None
        
        for task_state in state.tasks:
            if hasattr(task_state, 'interrupts') and task_state.interrupts:
                has_interrupt = True
                interrupt_payload = task_state.interrupts[0].value if task_state.interrupts else None
                break
        
        if not has_interrupt:
            # No more interrupts, return the final values
            return state.values
        
        # Stop the progress spinner while prompting
        if progress is not None and task is not None:
            progress.stop()
        
        # Display the interrupt to the user
        if interrupt_payload and isinstance(interrupt_payload, dict):
            interrupt_type = interrupt_payload.get('type', 'input_required')
            
            if interrupt_type == 'file_request':
                # File request interrupt from supervisor
                description = interrupt_payload.get('description', 'Files requested')
                accepted_types = interrupt_payload.get('accepted_types', ['image', 'pdf'])
                message = interrupt_payload.get('message', description)
                
                console.print(f"\n[bold yellow]📎 File Request[/bold yellow]")
                console.print(Panel(
                    Markdown(message),
                    border_style="yellow",
                    box=box.ROUNDED
                ))
                console.print("[dim]Provide file path(s) separated by spaces, or type 'skip' to continue without files[/dim]")
                console.print()
                
                user_input = Prompt.ask(
                    "[bold]File path(s) or response[/bold]",
                    console=console
                )
                console.print()
                
                # Check if user provided file paths
                if user_input.lower().strip() not in ('skip', 'exit', 'quit', 'stop', 'cancel'):
                    file_response = process_file_input(user_input)
                    if file_response:
                        user_input = file_response
                
            elif interrupt_type == 'missing_config':
                field = interrupt_payload.get('field', 'unknown')
                env_var = interrupt_payload.get('env_var', field.upper())
                instructions = interrupt_payload.get('instructions', f'Please provide a value for {field}')
                
                console.print(f"\n[bold yellow]⚠ Missing Configuration[/bold yellow]")
                console.print(Panel(
                    f"[bold]{env_var}[/bold] is required but not set.\n\n"
                    f"[dim]{instructions}[/dim]",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
                console.print()
                
                user_input = Prompt.ask(
                    f"[bold]Enter value for [cyan]{env_var}[/cyan][/bold] (or 'exit' to quit)",
                    console=console
                )
                console.print()
            else:
                # Generic interrupt
                console.print(f"\n[bold yellow]Input Required[/bold yellow]")
                if isinstance(interrupt_payload, dict):
                    console.print(Panel(
                        json.dumps(interrupt_payload, indent=2, default=str),
                        border_style="yellow",
                        box=box.ROUNDED
                    ))
                else:
                    console.print(f"[dim]{interrupt_payload}[/dim]")
                
                user_input = Prompt.ask("Your response (or 'exit' to quit)", console=console)
        else:
            # Handle string payloads (like postgres write approval)
            if isinstance(interrupt_payload, str):
                # Check if it's a postgres write approval request
                if "PostgreSQL Write Approval" in interrupt_payload or interrupt_payload.startswith("🔒"):
                    console.print(f"\n[bold yellow]🔒 Database Write Approval[/bold yellow]")
                    console.print(Panel(
                        Markdown(interrupt_payload),
                        border_style="yellow",
                        box=box.ROUNDED
                    ))
                    console.print()
                    user_input = Prompt.ask(
                        "[bold]Approve?[/bold] (yes/approve to proceed, no/reject to cancel)",
                        console=console
                    )
                else:
                    # Generic string payload - render as markdown
                    console.print(f"\n[bold yellow]Input Required[/bold yellow]")
                    console.print(Panel(
                        Markdown(interrupt_payload),
                        border_style="yellow",
                        box=box.ROUNDED
                    ))
                    user_input = Prompt.ask("Your response (or 'exit' to quit)", console=console)
            else:
                # Unknown interrupt format
                console.print(f"\n[bold yellow]Input Required[/bold yellow]")
                console.print(f"[dim]Payload: {interrupt_payload}[/dim]")
                user_input = Prompt.ask("Your response (or 'exit' to quit)", console=console)
        
        # Check for exit
        if user_input.lower() in ('exit', 'quit', 'stop', 'cancel'):
            console.print("[yellow]Exiting interrupt handler...[/yellow]")
            return state.values
        
        # Restart progress if it was running
        if progress is not None and task is not None:
            progress.start()
        
        # Resume the graph with the user's input
        try:
            await graph.ainvoke(Command(resume=user_input), config=config)
        except Exception as e:
            if VERBOSE:
                console.print(f"\n[bold red]Error during resume:[/bold red] {e}")
                console.print(traceback.format_exc())
            raise


@cli.command()
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'table']), default='table', help='Output format')
def config(fmt: str):
    """
    Show current configuration.
    
    Displays configuration values loaded from environment variables and .env file.
    """
    from shared.config import config as seer_config
    
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Configuration[/bold cyan]",
        border_style="cyan"
    ))
    
    # Configuration sections to display
    sections = {
        "API Keys": [
            ("openai_api_key", "OPENAI_API_KEY", True),  # (attr, env_var, is_secret)
            ("composio_api_key", "COMPOSIO_API_KEY", True),
            ("github_token", "GITHUB_TOKEN", True),
            ("langfuse_secret_key", "LANGFUSE_SECRET_KEY", True),
        ],
        "Evaluation Settings": [
            ("eval_n_rounds", "EVAL_N_ROUNDS", False),
            ("eval_n_test_cases", "EVAL_N_TEST_CASES", False),
            ("eval_pass_threshold", "EVAL_PASS_THRESHOLD", False),
            ("eval_reasoning_effort", "EVAL_REASONING_EFFORT", False),
        ],
        "External Services": [
            ("langfuse_base_url", "LANGFUSE_BASE_URL", False),
            ("neo4j_uri", "NEO4J_URI", False),
            ("database_uri", "DATABASE_URI", True),
        ],
    }
    
    if fmt == 'json':
        output = {}
        for section, items in sections.items():
            output[section] = {}
            for attr, env_var, is_secret in items:
                value = getattr(seer_config, attr, None)
                if is_secret and value:
                    output[section][attr] = "***" + value[-4:] if len(str(value)) > 4 else "***"
                else:
                    output[section][attr] = value
        console.print(json.dumps(output, indent=2, default=str))
    else:
        for section, items in sections.items():
            table = Table(title=section, box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Env Variable", style="dim")
            table.add_column("Value")
            table.add_column("Status", justify="center")
            
            for attr, env_var, is_secret in items:
                value = getattr(seer_config, attr, None)
                if value is None:
                    display_value = "[dim]not set[/dim]"
                    status = "[yellow]○[/yellow]"
                elif is_secret:
                    display_value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
                    status = "[green]●[/green]"
                else:
                    display_value = str(value)
                    status = "[green]●[/green]"
                
                table.add_row(attr, env_var, display_value, status)
            
            console.print(table)
            console.print()


@cli.command()
@click.argument('thread_id', required=True)
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'markdown']), default='markdown', help='Output format')
def export(thread_id: str, fmt: str):
    """
    Export results from a previous run.
    
    Retrieves and formats the results from a completed evaluation session.
    
    \b
    Example:
      seer-eval export <thread-id>
      seer-eval export <thread-id> --format json
    """
    console.print(f"[yellow]Export functionality requires database checkpointer.[/yellow]")
    console.print(f"[dim]Thread ID: {thread_id}[/dim]")
    console.print("\n[dim]Note: When using MemorySaver, state is only available during the session.[/dim]")
    console.print("[dim]Set DATABASE_URI in your .env to enable persistent state.[/dim]")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

