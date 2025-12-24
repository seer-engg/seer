#!/usr/bin/env python3
"""
CLI for the Seer Eval Agent and Supervisor Agent.

Usage:
    uv run seer dev            # Start development environment (recommended: no installation)
    
    # If CLI is installed, you can use 'seer' directly instead of 'uv run seer'
"""
import json
import sys
import threading
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Load .env before importing seer modules
load_dotenv()

console = Console()

# Global verbose flag
VERBOSE = False

@click.group()
@click.version_option(version="0.1.4", prog_name="seer")
@click.option('--verbose', '-v', is_flag=True, help='Show full error tracebacks for debugging')
def cli(verbose: bool):
    """
    🔮 Seer - Multi-Agent System for Evaluating AI Agents
    
    Evaluate AI agents through automated test generation and execution.
    
    \b
    Commands:
      dev            - Start development environment with Docker Compose
    
    \b
    Examples:
      # Start development environment (recommended: no installation needed)
      uv run seer dev
          
      # Debug with full tracebacks
      uv run seer -v dev
      
      # If CLI is installed, you can use 'seer' directly
      seer dev
    """
    global VERBOSE
    VERBOSE = verbose


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
            ("github_token", "GITHUB_TOKEN", True),
        ],
        "Evaluation Settings": [
            ("eval_n_rounds", "EVAL_N_ROUNDS", False),
            ("eval_n_test_cases", "EVAL_N_TEST_CASES", False),
            ("eval_pass_threshold", "EVAL_PASS_THRESHOLD", False),
            ("eval_reasoning_effort", "EVAL_REASONING_EFFORT", False),
        ],
        "External Services": [
            ("neo4j_uri", "NEO4J_URI", False),
            ("database_url", "DATABASE_URL", True),
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
      uv run seer export <thread-id>  # Recommended: no installation needed
      uv run seer export <thread-id> --format json
      # Or if CLI is installed: seer export <thread-id>
    """
    console.print(f"[yellow]Export functionality requires database checkpointer.[/yellow]")
    console.print(f"[dim]Thread ID: {thread_id}[/dim]")
    console.print("\n[dim]Note: When using MemorySaver, state is only available during the session.[/dim]")
    console.print("[dim]Set DATABASE_URL in your .env to enable persistent state.[/dim]")


def _tail_docker_logs(project_root: Path, service: str, stop_event: threading.Event):
    """
    Tail Docker logs for a service in a background thread.
    
    Args:
        project_root: Path to project root (where docker-compose.yml is)
        service: Service name to tail logs for
        stop_event: Threading event to signal when to stop tailing
    """
    import subprocess
    import select
    import time
    
    try:
        # Start docker compose logs with follow flag
        process = subprocess.Popen(
            ["docker", "compose", "logs", "-f", "--tail=0", service],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Read lines until stop event is set
        # Use select for non-blocking reads (Unix only, but that's fine for Docker)
        while not stop_event.is_set():
            # Check if data is available (Unix/Linux/macOS)
            if sys.platform != "win32":
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if not ready:
                    continue
            
            line = process.stdout.readline()
            if not line:
                # Process ended or no more data
                if stop_event.is_set():
                    break
                # Small delay before checking again
                time.sleep(0.1)
                continue
            
            # Print log line with service prefix
            console.print(f"[dim][{service}][/dim] {line.rstrip()}")
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        # Silently fail - don't interrupt main flow
        pass


@cli.command()
@click.option('--frontend-url', default='http://localhost:5173', help='Frontend URL (default: http://localhost:5173)')
@click.option('--backend-url', default='http://localhost:8000', help='Backend URL (default: http://localhost:8000)')
@click.option('--no-browser', is_flag=True, help='Do not open browser automatically')
@click.option('--rebuild', is_flag=True, help='Force rebuild containers (stops, removes, rebuilds, then starts)')
def dev(frontend_url: str, backend_url: str, no_browser: bool, rebuild: bool):
    """
    Start development environment with Docker Compose.
    
    Starts Postgres, MLflow, and backend server, then opens the workflow editor
    in your browser connected to the local backend.
    
    \b
    This command:
      1. Starts Docker Compose services (postgres, mlflow, backend)
      2. Installs dependencies in Docker container (via uv sync during build)
      3. Tails Docker logs in real-time during startup
      4. Waits for backend to be ready (health check)
      5. Opens browser automatically to: <frontend-url>/workflows?backend=<backend-url>
    
    \b
    Note: Dependencies are installed inside Docker containers, not locally.
    Only the CLI tool needs to be installed locally (lightweight: click, rich).
    
    \b
    Code changes are reflected immediately via volume mounts and uvicorn --reload.
    Use --rebuild to force a full rebuild (useful when dependencies change).
    
    \b
    Example:
      uv run seer dev  # Recommended: no installation needed
      seer dev  # If CLI is installed
      seer dev --frontend-url http://localhost:3000
      seer dev --no-browser
      seer dev --rebuild  # Force rebuild containers
    """
    import subprocess
    import time
    import sys
    import os
    import shutil
    from pathlib import Path
    
    # Clear VIRTUAL_ENV if it points to Docker path to avoid uv warnings
    if os.environ.get('VIRTUAL_ENV', '').startswith('/app'):
        os.environ.pop('VIRTUAL_ENV', None)
    
    # Get project root (where docker-compose.yml is)
    project_root = Path(__file__).parent.parent
    docker_compose_file = project_root / "docker-compose.yml"
    
    # Check for broken .venv symlinks and clean if needed
    venv_path = project_root / ".venv"
    if venv_path.exists():
        python_symlink = venv_path / "bin" / "python3"
        if python_symlink.exists() and python_symlink.is_symlink():
            try:
                target = python_symlink.readlink()
                # Check if symlink target exists (resolved relative to symlink's directory)
                if not (venv_path / "bin" / target).exists() and not Path(target).exists():
                    console.print("[yellow]⚠ Detected broken .venv symlinks, cleaning...[/yellow]")
                    shutil.rmtree(venv_path)
                    console.print("[green]✓[/green] Cleaned broken .venv")
            except (OSError, ValueError):
                # Symlink is broken, clean it
                console.print("[yellow]⚠ Detected broken .venv, cleaning...[/yellow]")
                shutil.rmtree(venv_path)
                console.print("[green]✓[/green] Cleaned broken .venv")
    
    if not docker_compose_file.exists():
        console.print(f"[bold red]Error:[/bold red] docker-compose.yml not found at {docker_compose_file}")
        sys.exit(1)
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Starting Seer Development Environment[/bold cyan]\n\n"
        f"[dim]Frontend: {frontend_url}[/dim]\n"
        f"[dim]Backend: {backend_url}[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    # Handle rebuild option
    if rebuild:
        console.print("[bold]🔄 Force rebuild requested - stopping and removing containers...[/bold]")
        try:
            # Stop and remove containers
            # Unset VIRTUAL_ENV in subprocess env to avoid uv warnings
            env = os.environ.copy()
            if env.get('VIRTUAL_ENV', '').startswith('/app'):
                env.pop('VIRTUAL_ENV', None)
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=project_root,
                capture_output=True,
                text=True,
                env=env,
            )
            console.print("[green]✓[/green] Containers stopped and removed")
        except FileNotFoundError:
            console.print("[bold red]Error:[/bold red] docker-compose not found. Please install Docker Compose.")
            sys.exit(1)
    
    # Start Docker Compose with --build flag to ensure dependencies are up-to-date
    console.print("[bold]📦 Starting Docker Compose services...[/bold]")
    try:
        # Always use --build to ensure image is rebuilt if Dockerfile or dependencies changed
        # Volume mount ensures code changes don't require rebuild
        # Unset VIRTUAL_ENV in subprocess env to avoid uv warnings
        env = os.environ.copy()
        if env.get('VIRTUAL_ENV', '').startswith('/app'):
            env.pop('VIRTUAL_ENV', None)
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--build"],
            cwd=project_root,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            console.print(f"[bold red]Error starting Docker Compose:[/bold red]")
            console.print(result.stderr)
            sys.exit(1)
        console.print("[green]✓[/green] Services started")
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] docker-compose not found. Please install Docker Compose.")
        sys.exit(1)
    
    # Wait for backend to be ready
    console.print()
    console.print("[bold]⏳ Waiting for backend server to be ready...[/bold]")
    console.print("[dim]📋 Tailing Docker logs (langgraph-server)...[/dim]")
    console.print()
    
    max_attempts = 60
    attempt = 0
    import urllib.request
    import urllib.error
    
    # Start tailing Docker logs in background thread
    # This will continue running until interrupted or error
    stop_logs_event = threading.Event()
    logs_thread = threading.Thread(
        target=_tail_docker_logs,
        args=(project_root, "langgraph-server", stop_logs_event),
        daemon=True,
    )
    logs_thread.start()
    
    backend_ready = False
    try:
        # Wait for backend to be ready
        while attempt < max_attempts:
            try:
                health_url = f"{backend_url}/health"
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.getcode() == 200:
                        console.print()
                        console.print("[green]✓[/green] Backend server is ready!")
                        backend_ready = True
                        break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            
            attempt += 1
            time.sleep(2)
        
        if not backend_ready:
            console.print()
            console.print("[bold red]❌ Backend server failed to start[/bold red]")
            console.print("[yellow]Last 50 lines of logs:[/yellow]")
            console.print()
            # Unset VIRTUAL_ENV in subprocess env to avoid uv warnings
            env = os.environ.copy()
            if env.get('VIRTUAL_ENV', '').startswith('/app'):
                env.pop('VIRTUAL_ENV', None)
            subprocess.run(
                ["docker", "compose", "logs", "langgraph-server", "--tail=50"],
                cwd=project_root,
                env=env,
            )
            stop_logs_event.set()
            sys.exit(1)
        
        # Backend is ready - show success message and continue tailing logs
        workflow_url = f"{frontend_url}/workflows?backend={backend_url}"
        
        console.print()
        console.print(Panel.fit(
            "[bold green]🎉 Development environment is ready![/bold green]\n\n"
            "[bold]📊 Services:[/bold]\n"
            f"   • Backend API: {backend_url}\n"
            f"   • MLflow: http://localhost:5000\n"
            f"   • Postgres: localhost:5432\n\n"
            f"[bold]🌐 Workflow Editor:[/bold]\n"
            f"   {workflow_url}\n\n"
            "[dim]📋 Logs will continue streaming. Press Ctrl+C to stop.[/dim]",
            border_style="green"
        ))
        console.print()
        
        # Open browser
        if not no_browser:
            console.print("[bold]🌐 Opening workflow editor...[/bold]")
            try:
                if sys.platform == "darwin":
                    # macOS
                    subprocess.run(["open", workflow_url], check=False)
                elif sys.platform == "linux":
                    # Linux
                    subprocess.run(["xdg-open", workflow_url], check=False)
                elif sys.platform == "win32":
                    # Windows
                    subprocess.run(["start", workflow_url], check=False, shell=True)
                else:
                    console.print(f"[yellow]⚠ Could not automatically open browser.[/yellow]")
                    console.print(f"Please navigate to: {workflow_url}")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not open browser: {e}[/yellow]")
                console.print(f"Please navigate to: {workflow_url}")
        else:
            console.print(f"[dim]Browser opening skipped. Navigate to:[/dim]")
            console.print(f"[bold]{workflow_url}[/bold]")
        
        console.print()
        console.print("[dim]📝 To view logs: docker compose logs -f[/dim]")
        console.print("[dim]🛑 To stop: Press Ctrl+C or run 'docker compose down'[/dim]")
        console.print()
        
        # Keep tailing logs until interrupted
        # The log tailing thread will continue running
        # Wait for the thread to finish (which happens on Ctrl+C or error)
        try:
            while logs_thread.is_alive():
                logs_thread.join(timeout=1)
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]🛑 Stopping development environment...[/yellow]")
            stop_logs_event.set()
            logs_thread.join(timeout=2)
            console.print("[green]✓[/green] Stopped")
            
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]🛑 Stopping development environment...[/yellow]")
        stop_logs_event.set()
        logs_thread.join(timeout=2)
        console.print("[green]✓[/green] Stopped")
    except Exception as e:
        console.print()
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        stop_logs_event.set()
        logs_thread.join(timeout=2)
        raise


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

