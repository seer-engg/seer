#!/usr/bin/env python3
"""
Database Migration Helper Script (Python version for cross-platform support)

Usage:
    python scripts/migrate.py              # Run migrations
    python scripts/migrate.py create       # Create a new migration
    python scripts/migrate.py rollback     # Rollback one migration
    python scripts/migrate.py history      # Show migration history
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_colored(message: str, color: str):
    """Print colored message"""
    print(f"{color}{message}{Colors.NC}")


def is_in_docker() -> bool:
    """Check if running inside Docker container"""
    return os.path.exists('/.dockerenv')


def has_docker_compose() -> bool:
    """Check if docker-compose is available"""
    try:
        subprocess.run(['docker-compose', '--version'],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_alembic(args: list[str]) -> int:
    """Run alembic command in appropriate environment"""
    in_docker = is_in_docker()
    docker_available = has_docker_compose()

    if in_docker:
        # Already in Docker, run directly
        cmd = ['uv', 'run', 'alembic'] + args
    elif docker_available:
        # Run via docker-compose
        print_colored("Running in Docker container...", Colors.BLUE)
        cmd = ['docker-compose', 'exec', 'langgraph-server', 'uv', 'run', 'alembic'] + args
    else:
        # Run locally
        print_colored("Running locally...", Colors.BLUE)
        cmd = ['uv', 'run', 'alembic'] + args

    result = subprocess.run(cmd)
    return result.returncode


def ensure_db_ready():
    """Ensure database is ready"""
    print_colored("Checking database connection...", Colors.BLUE)

    if has_docker_compose() and not is_in_docker():
        # Start postgres if not running
        subprocess.run(['docker-compose', 'up', '-d', 'postgres'],
                      capture_output=True)
        print_colored("Waiting for PostgreSQL to be ready...", Colors.YELLOW)
        time.sleep(3)


def cmd_upgrade():
    """Run all pending migrations"""
    print_colored("🔄 Running database migrations...", Colors.GREEN)
    ensure_db_ready()
    exitcode = run_alembic(['upgrade', 'head'])
    if exitcode == 0:
        print_colored("✅ Migrations completed successfully!", Colors.GREEN)
    return exitcode


def cmd_create(message: str = "migration"):
    """Create a new migration"""
    print_colored(f"📝 Creating new migration: {message}", Colors.GREEN)
    ensure_db_ready()
    exitcode = run_alembic(['revision', '--autogenerate', '-m', message])
    if exitcode == 0:
        print_colored("✅ Migration file created!", Colors.GREEN)
        print_colored("📍 Review the generated migration in alembic/versions/", Colors.YELLOW)
    return exitcode


def cmd_rollback(steps: str = "-1"):
    """Rollback migrations"""
    print_colored(f"⬇️  Rolling back {steps} migration(s)...", Colors.YELLOW)
    ensure_db_ready()
    exitcode = run_alembic(['downgrade', steps])
    if exitcode == 0:
        print_colored("✅ Rollback completed!", Colors.GREEN)
    return exitcode


def cmd_history():
    """Show migration history"""
    print_colored("📜 Migration history:", Colors.BLUE)
    return run_alembic(['history', '--verbose'])


def cmd_current():
    """Show current migration status"""
    print_colored("📍 Current migration status:", Colors.BLUE)
    return run_alembic(['current'])


def cmd_reset():
    """Reset database (DANGER!)"""
    print_colored("⚠️  WARNING: This will DROP all tables and re-run migrations!", Colors.RED)
    print_colored("This action cannot be undone!", Colors.RED)

    response = input("Are you sure? Type 'YES' to confirm: ")
    if response == "YES":
        print_colored("Dropping all tables...", Colors.YELLOW)
        ensure_db_ready()
        run_alembic(['downgrade', 'base'])
        print_colored("Re-running migrations...", Colors.YELLOW)
        exitcode = run_alembic(['upgrade', 'head'])
        if exitcode == 0:
            print_colored("✅ Database reset complete!", Colors.GREEN)
        return exitcode
    else:
        print_colored("Cancelled.", Colors.BLUE)
        return 0


def show_help():
    """Show help message"""
    print("""Database Migration Helper

Usage: python scripts/migrate.py [command] [options]

Commands:
  upgrade, up, migrate          Run all pending migrations (default)
  create, new, revision [name]  Create a new migration
  rollback, down [steps]        Rollback migrations (default: -1)
  history, log, show            Show migration history
  current, status               Show current migration status
  reset, drop                   Drop all tables and re-run migrations (DANGER!)
  help, -h, --help              Show this help message

Examples:
  python scripts/migrate.py                          # Run migrations
  python scripts/migrate.py create add_user_fields   # Create new migration
  python scripts/migrate.py rollback                 # Rollback last migration
  python scripts/migrate.py history                  # Show migration history
""")


def main():
    """Main entry point"""
    command = sys.argv[1] if len(sys.argv) > 1 else 'upgrade'

    commands = {
        'upgrade': cmd_upgrade,
        'up': cmd_upgrade,
        'migrate': cmd_upgrade,
        'create': lambda: cmd_create(sys.argv[2] if len(sys.argv) > 2 else 'migration'),
        'new': lambda: cmd_create(sys.argv[2] if len(sys.argv) > 2 else 'migration'),
        'revision': lambda: cmd_create(sys.argv[2] if len(sys.argv) > 2 else 'migration'),
        'rollback': lambda: cmd_rollback(sys.argv[2] if len(sys.argv) > 2 else '-1'),
        'down': lambda: cmd_rollback(sys.argv[2] if len(sys.argv) > 2 else '-1'),
        'downgrade': lambda: cmd_rollback(sys.argv[2] if len(sys.argv) > 2 else '-1'),
        'history': cmd_history,
        'log': cmd_history,
        'show': cmd_history,
        'current': cmd_current,
        'status': cmd_current,
        'reset': cmd_reset,
        'drop': cmd_reset,
        'help': lambda: show_help() or 0,
        '-h': lambda: show_help() or 0,
        '--help': lambda: show_help() or 0,
    }

    if command in commands:
        exitcode = commands[command]()
        sys.exit(exitcode if exitcode is not None else 0)
    else:
        print_colored(f"Unknown command: {command}", Colors.RED)
        print("Run 'python scripts/migrate.py help' for usage information")
        sys.exit(1)


if __name__ == '__main__':
    main()
