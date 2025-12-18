#!/usr/bin/env python3
"""
CLI for the Seer Eval Agent.

Usage:
    seer-eval align "Evaluate my agent that syncs GitHub PRs to Asana" --repo seer-engg/my-agent
    seer-eval plan --thread-id <uuid>
    seer-eval test --thread-id <uuid>
    seer-eval run "Evaluate my agent..." --repo seer-engg/my-agent  # Full pipeline
"""
import asyncio
import json
import sys
import uuid
import traceback
from pathlib import Path
from typing import Optional, Any

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
      align   - Align agent expectations with user requirements
      plan    - Generate test cases based on aligned expectations
      test    - Execute generated test cases
      run     - Full pipeline: align → plan → test
    
    \b
    Examples:
      # Start with alignment
      seer-eval align "Evaluate my GitHub-Asana bot" --repo owner/repo
      
      # Continue with planning (use thread-id from previous step)
      seer-eval plan --thread-id <uuid>
      
      # Or run the full pipeline
      seer-eval run "Evaluate my bot" --repo owner/repo --user-id me@example.com
      
      # Debug with full tracebacks
      seer-eval -v run "My agent" --repo owner/repo
    """
    global VERBOSE
    VERBOSE = verbose


@cli.command()
@click.argument('description', required=True)
@click.option('--repo', '-r', help='GitHub repository (owner/repo format)', required=True)
@click.option('--user-id', '-u', default=None, help='User ID for authentication context')
@click.option('--thread-id', '-t', default=None, help='Thread ID (auto-generated if not provided)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
def align(description: str, repo: str, user_id: Optional[str], thread_id: Optional[str], output: Optional[str]):
    """
    Align agent expectations with user requirements.
    
    This step analyzes your description and generates functional requirements
    and identifies required MCP services (GitHub, Asana, etc.).
    
    \b
    Example:
      seer-eval align "My agent syncs GitHub PRs to Asana tickets" --repo seer-engg/my-agent
    """
    asyncio.run(_align(description, repo, user_id, thread_id, output))


async def _align(description: str, repo: str, user_id: Optional[str], thread_id: Optional[str], output: Optional[str]):
    """Async implementation of align command."""
    thread_id = thread_id or str(uuid.uuid4())
    
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Eval Agent - Alignment[/bold cyan]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]",
        border_style="cyan"
    ))
    
    console.print(f"\n[bold]Description:[/bold] {description}")
    console.print(f"[bold]Repository:[/bold] {repo}")
    if user_id:
        console.print(f"[bold]User ID:[/bold] {user_id}")
    console.print()
    
    # Build input
    inputs = {
        "messages": [{"type": "human", "content": description}],
        "step": "alignment",
        "input_context": {
            "integrations": {
                "github": {"name": repo}
            }
        }
    }
    if user_id:
        inputs["input_context"]["user_id"] = user_id
    
    # Run alignment
    memory = MemorySaver()
    eval_agent = create_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Running alignment...", total=None)
        
        try:
            # Initial invocation
            await eval_agent.ainvoke(inputs, config=runnable_config)
            
            # Handle any interrupts (prompts for missing config, etc.)
            results = await handle_interrupts(
                eval_agent,
                runnable_config,
                progress=progress,
                task=prog_task,
            )
            progress.update(prog_task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error:[/bold red] {e}")
            if VERBOSE:
                console.print("\n[bold red]Full Traceback:[/bold red]")
                console.print(traceback.format_exc())
            else:
                console.print("[dim]Use -v flag for full traceback: seer-eval -v align ...[/dim]")
            sys.exit(1)
    
    # Display results
    console.print("\n[bold green]✓ Alignment Complete[/bold green]\n")
    
    context = results.get('context')
    if context:
        console.print(Panel(
            Markdown(format_context(context)),
            title="[bold]Agent Context[/bold]",
            border_style="green",
            box=box.ROUNDED
        ))
    
    # Show thread ID for next steps
    console.print(f"\n[bold cyan]Next Steps:[/bold cyan]")
    console.print(f"  Continue with planning:")
    console.print(f"  [dim]seer-eval plan --thread-id {thread_id}[/dim]\n")
    
    # Save output if requested
    if output:
        output_data = {
            "thread_id": thread_id,
            "step": "alignment",
            "context": context.model_dump() if context else None
        }
        Path(output).write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"[dim]Results saved to {output}[/dim]")


@cli.command()
@click.option('--thread-id', '-t', required=True, help='Thread ID from alignment step')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
def plan(thread_id: str, output: Optional[str]):
    """
    Generate test cases based on aligned expectations.
    
    Requires a thread-id from a previous alignment step.
    
    \b
    Example:
      seer-eval plan --thread-id <uuid-from-align>
    """
    asyncio.run(_plan(thread_id, output))


async def _plan(thread_id: str, output: Optional[str]):
    """Async implementation of plan command."""
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Eval Agent - Planning[/bold cyan]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]",
        border_style="cyan"
    ))
    
    inputs = {"step": "plan"}
    memory = MemorySaver()
    eval_agent = create_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Generating test cases...", total=None)
        
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
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error:[/bold red] {e}")
            if VERBOSE:
                console.print("\n[bold red]Full Traceback:[/bold red]")
                console.print(traceback.format_exc())
            else:
                console.print("[dim]Use -v flag for full traceback: seer-eval -v plan ...[/dim]")
            sys.exit(1)
    
    console.print("\n[bold green]✓ Planning Complete[/bold green]\n")
    
    # Display generated test cases
    examples = results.get('dataset_examples', [])
    if examples:
        console.print(f"[bold]Generated {len(examples)} test case(s):[/bold]\n")
        for example in examples:
            console.print(Panel(
                Markdown(format_dataset_example(example)),
                border_style="blue",
                box=box.ROUNDED
            ))
    else:
        console.print("[yellow]No test cases generated.[/yellow]")
    
    # Show next steps
    console.print(f"\n[bold cyan]Next Steps:[/bold cyan]")
    console.print(f"  Continue with testing:")
    console.print(f"  [dim]seer-eval test --thread-id {thread_id}[/dim]\n")
    
    # Save output if requested
    if output:
        output_data = {
            "thread_id": thread_id,
            "step": "plan",
            "dataset_examples": [e.model_dump() for e in examples]
        }
        Path(output).write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"[dim]Results saved to {output}[/dim]")


@cli.command()
@click.option('--thread-id', '-t', required=True, help='Thread ID from planning step')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
def test(thread_id: str, output: Optional[str]):
    """
    Execute generated test cases against the target agent.
    
    Requires a thread-id from a previous planning step.
    
    \b
    Example:
      seer-eval test --thread-id <uuid-from-plan>
    """
    asyncio.run(_test(thread_id, output))


async def _test(thread_id: str, output: Optional[str]):
    """Async implementation of test command."""
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Eval Agent - Testing[/bold cyan]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]",
        border_style="cyan"
    ))
    
    inputs = {"step": "testing"}
    memory = MemorySaver()
    eval_agent = create_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Running tests...", total=None)
        
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
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error:[/bold red] {e}")
            if VERBOSE:
                console.print("\n[bold red]Full Traceback:[/bold red]")
                console.print(traceback.format_exc())
            else:
                console.print("[dim]Use -v flag for full traceback: seer-eval -v test ...[/dim]")
            sys.exit(1)
    
    console.print("\n[bold green]✓ Testing Complete[/bold green]\n")
    
    # Display results
    latest_results = results.get('latest_results', [])
    if latest_results:
        table = Table(title="Test Results", box=box.ROUNDED)
        table.add_column("Test ID", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Passed", justify="center")
        table.add_column("Reasoning")
        
        for result in latest_results:
            score = result.score if hasattr(result, 'score') else 0.0
            passed = result.passed if hasattr(result, 'passed') else False
            reasoning = result.judge_reasoning[:50] + "..." if hasattr(result, 'judge_reasoning') and len(result.judge_reasoning) > 50 else (result.judge_reasoning if hasattr(result, 'judge_reasoning') else "N/A")
            
            score_style = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
            passed_icon = "✓" if passed else "✗"
            passed_style = "green" if passed else "red"
            
            table.add_row(
                result.dataset_example.example_id if hasattr(result, 'dataset_example') else "N/A",
                f"[{score_style}]{score:.2f}[/{score_style}]",
                f"[{passed_style}]{passed_icon}[/{passed_style}]",
                reasoning
            )
        
        console.print(table)
    else:
        console.print("[yellow]No test results available.[/yellow]")
    
    # Check for missing config
    missing_config = results.get('missing_config', [])
    if missing_config:
        console.print(f"\n[bold yellow]⚠ Missing Configuration:[/bold yellow]")
        for config_key in missing_config:
            console.print(f"  • {config_key}")
        console.print("\n[dim]Set these in your .env file or as environment variables.[/dim]")
    
    # Save output if requested
    if output:
        output_data = {
            "thread_id": thread_id,
            "step": "testing",
            "results": [r.model_dump() for r in latest_results] if latest_results else []
        }
        Path(output).write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"\n[dim]Results saved to {output}[/dim]")


@cli.command()
@click.argument('description', required=True)
@click.option('--repo', '-r', help='GitHub repository (owner/repo format)', required=True)
@click.option('--user-id', '-u', default=None, help='User ID for authentication context')
@click.option('--thread-id', '-t', default=None, help='Thread ID (auto-generated if not provided)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
@click.option('--skip-testing', is_flag=True, help='Stop after planning (don\'t run tests)')
def run(description: str, repo: str, user_id: Optional[str], thread_id: Optional[str], output: Optional[str], skip_testing: bool):
    """
    Run the full evaluation pipeline: align → plan → test.
    
    \b
    Example:
      seer-eval run "Evaluate my GitHub-Asana sync bot" --repo seer-engg/my-agent
      seer-eval run "Test my agent" --repo owner/repo --skip-testing  # Stop after plan
    """
    asyncio.run(_run(description, repo, user_id, thread_id, output, skip_testing))


async def _run(description: str, repo: str, user_id: Optional[str], thread_id: Optional[str], output: Optional[str], skip_testing: bool):
    """Async implementation of run command (full pipeline)."""
    thread_id = thread_id or str(uuid.uuid4())
    
    console.print(Panel.fit(
        "[bold cyan]🔮 Seer Eval Agent - Full Pipeline[/bold cyan]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]",
        border_style="cyan"
    ))
    
    console.print(f"\n[bold]Description:[/bold] {description}")
    console.print(f"[bold]Repository:[/bold] {repo}")
    if user_id:
        console.print(f"[bold]User ID:[/bold] {user_id}")
    console.print()
    
    memory = MemorySaver()
    eval_agent = create_compiled_graph(memory)
    runnable_config = RunnableConfig(configurable={"thread_id": thread_id})
    
    all_results = {}
    
    # Step 1: Alignment
    console.print("\n[bold]Step 1/3: Alignment[/bold]")
    console.print("─" * 40)
    
    alignment_inputs = {
        "messages": [{"type": "human", "content": description}],
        "step": "alignment",
        "input_context": {
            "integrations": {
                "github": {"name": repo}
            }
        }
    }
    if user_id:
        alignment_inputs["input_context"]["user_id"] = user_id
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Running alignment...", total=None)
        try:
            # Initial invocation
            await eval_agent.ainvoke(alignment_inputs, config=runnable_config)
            
            # Handle any interrupts (prompts for missing config, etc.)
            alignment_results = await handle_interrupts(
                eval_agent,
                runnable_config,
                progress=progress,
                task=prog_task,
            )
            progress.update(prog_task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error in alignment:[/bold red] {e}")
            if VERBOSE:
                console.print("\n[bold red]Full Traceback:[/bold red]")
                console.print(traceback.format_exc())
            else:
                console.print("[dim]Use -v flag for full traceback: seer-eval -v run ...[/dim]")
            sys.exit(1)
    
    context = alignment_results.get('context')
    if context:
        console.print(f"[green]✓[/green] Agent: {context.agent_name if hasattr(context, 'agent_name') else 'Unknown'}")
        if hasattr(context, 'mcp_services') and context.mcp_services:
            console.print(f"[green]✓[/green] Services: {', '.join(context.mcp_services)}")
    all_results['alignment'] = alignment_results
    
    # Check if user exited during alignment
    if alignment_results.get('should_exit'):
        console.print("\n[yellow]Pipeline stopped due to missing configuration.[/yellow]")
        return
    
    # Step 2: Planning
    console.print("\n[bold]Step 2/3: Planning[/bold]")
    console.print("─" * 40)
    
    plan_inputs = {"step": "plan"}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Generating test cases...", total=None)
        try:
            # Initial invocation
            await eval_agent.ainvoke(plan_inputs, config=runnable_config)
            
            # Handle any interrupts
            plan_results = await handle_interrupts(
                eval_agent,
                runnable_config,
                progress=progress,
                task=prog_task,
            )
            progress.update(prog_task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error in planning:[/bold red] {e}")
            if VERBOSE:
                console.print("\n[bold red]Full Traceback:[/bold red]")
                console.print(traceback.format_exc())
            else:
                console.print("[dim]Use -v flag for full traceback: seer-eval -v run ...[/dim]")
            sys.exit(1)
    
    examples = plan_results.get('dataset_examples', [])
    console.print(f"[green]✓[/green] Generated {len(examples)} test case(s)")
    all_results['plan'] = plan_results
    
    # Check if user exited during planning
    if plan_results.get('should_exit'):
        console.print("\n[yellow]Pipeline stopped due to missing configuration.[/yellow]")
        return
    
    if skip_testing:
        console.print("\n[yellow]Skipping testing (--skip-testing flag set)[/yellow]")
    else:
        # Step 3: Testing
        console.print("\n[bold]Step 3/3: Testing[/bold]")
        console.print("─" * 40)
        
        test_inputs = {"step": "testing"}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            prog_task = progress.add_task("Running tests...", total=None)
            try:
                # Initial invocation
                await eval_agent.ainvoke(test_inputs, config=runnable_config)
                
                # Handle any interrupts
                test_results = await handle_interrupts(
                    eval_agent,
                    runnable_config,
                    progress=progress,
                    task=prog_task,
                )
                progress.update(prog_task, completed=True)
            except Exception as e:
                progress.stop()
                console.print(f"[bold red]Error in testing:[/bold red] {e}")
                if VERBOSE:
                    console.print("\n[bold red]Full Traceback:[/bold red]")
                    console.print(traceback.format_exc())
                else:
                    console.print("[dim]Use -v flag for full traceback: seer-eval -v run ...[/dim]")
                sys.exit(1)
        
        latest_results = test_results.get('latest_results', [])
        passed = sum(1 for r in latest_results if hasattr(r, 'passed') and r.passed)
        total = len(latest_results)
        
        if total > 0:
            console.print(f"[green]✓[/green] Tests complete: {passed}/{total} passed")
        
        # Check for missing config
        missing_config = test_results.get('missing_config', [])
        if missing_config:
            console.print(f"\n[bold yellow]⚠ Missing Configuration:[/bold yellow]")
            for config_key in missing_config:
                console.print(f"  • {config_key}")
        
        all_results['testing'] = test_results
    
    # Summary
    console.print("\n" + "═" * 50)
    console.print("[bold green]✓ Pipeline Complete[/bold green]")
    console.print("═" * 50)
    console.print(f"\n[dim]Thread ID: {thread_id}[/dim]")
    
    # Show test cases
    if examples:
        console.print(f"\n[bold]Generated Test Cases:[/bold]")
        for example in examples:
            console.print(Panel(
                Markdown(format_dataset_example(example)),
                border_style="blue",
                box=box.ROUNDED
            ))
    
    # Save output if requested
    if output:
        output_data = {
            "thread_id": thread_id,
            "alignment": {
                "context": all_results.get('alignment', {}).get('context').model_dump() if all_results.get('alignment', {}).get('context') else None
            },
            "plan": {
                "dataset_examples": [e.model_dump() for e in examples]
            }
        }
        if not skip_testing and 'testing' in all_results:
            latest = all_results['testing'].get('latest_results', [])
            output_data["testing"] = {
                "results": [r.model_dump() for r in latest] if latest else []
            }
        
        Path(output).write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"\n[dim]Results saved to {output}[/dim]")


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

