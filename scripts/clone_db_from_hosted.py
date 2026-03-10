#!/usr/bin/env python3
"""
Clone a hosted database (dev or main) to local Postgres using pg_dump/pg_restore.

Source DB URL  -> fetched from AWS SSM at /{source-env}/database_url (requires AWS creds).
Local DB URL   -> read from DATABASE_URL in .env file.

Both can be overridden with explicit CLI flags.

Usage:
    uv run scripts/clone_db_from_hosted.py --source-env dev
    uv run scripts/clone_db_from_hosted.py --source-env main --dry-run
    uv run scripts/clone_db_from_hosted.py --source-env dev --drop-existing --yes

    # Override URLs directly instead of using SSM / .env
    uv run scripts/clone_db_from_hosted.py \\
      --source-db-url "postgres://..." \\
      --local-db-url "postgres://..."

Examples:
    # Dry run - preview commands
    uv run scripts/clone_db_from_hosted.py --source-env dev --dry-run

    # Full clone from dev
    uv run scripts/clone_db_from_hosted.py --source-env dev

    # Clone main and drop existing local data
    uv run scripts/clone_db_from_hosted.py --source-env main --drop-existing --yes

    # Exclude large tables
    uv run scripts/clone_db_from_hosted.py --source-env dev \\
      --exclude-tables "llm_usage_records,session_recordings"
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Add src/ to path so we can import seer modules (AwsSsmSettingsSource, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DbConnectionInfo:
    """Parsed database connection information."""

    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_url(cls, url: str) -> "DbConnectionInfo":
        """Parse a postgres:// URL into connection components."""
        parsed = urlparse(url)
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Expected postgres:// or postgresql://")

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=parsed.username or "postgres",
            password=parsed.password or "",
            database=(parsed.path or "").lstrip("/") or "postgres",
        )

    def to_env_dict(self) -> dict[str, str]:
        """Return environment variables for pg_dump/pg_restore."""
        return {
            "PGHOST": self.host,
            "PGPORT": str(self.port),
            "PGUSER": self.user,
            "PGPASSWORD": self.password,
            "PGDATABASE": self.database,
        }

    def display_safe(self) -> str:
        """Return a display-safe connection string (password masked)."""
        masked_pw = "*" * min(8, len(self.password)) if self.password else ""
        return f"postgresql://{self.user}:{masked_pw}@{self.host}:{self.port}/{self.database}"


# ---------------------------------------------------------------------------
# Tool availability check
# ---------------------------------------------------------------------------


def check_tool_availability() -> tuple[bool, list[str]]:
    """Check if pg_dump and pg_restore are available."""
    missing = []
    for tool in ["pg_dump", "pg_restore"]:
        if shutil.which(tool) is None:
            missing.append(tool)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# URL resolution (following clone_user_from_prod.py pattern)
# ---------------------------------------------------------------------------


def resolve_source_db_url(source_env: str) -> str:
    """
    Fetch the source DATABASE_URL from AWS SSM at /{source_env}/database_url.
    Requires valid AWS credentials in the environment.
    """
    # pylint: disable=import-outside-toplevel  # Reason: seer imports need sys.path set up first
    from seer.config import SeerConfig
    from seer.utilities.aws.parameter_store import AwsSsmSettingsSource

    print(f"Fetching source DB URL from AWS SSM /{source_env}/database_url ...")
    ssm = AwsSsmSettingsSource(SeerConfig, ssm_path_prefix=f"/{source_env}/")
    params = ssm()
    url = params.get("database_url")
    if not url:
        raise ValueError(
            f"database_url not found in AWS SSM under /{source_env}/. "
            "Check your AWS credentials and that the parameter exists."
        )
    return url


def resolve_local_db_url() -> str:
    """Read DATABASE_URL from the project .env file."""
    try:
        # pylint: disable=import-outside-toplevel  # Reason: optional import, only needed here
        from dotenv import dotenv_values
    except ImportError as exc:
        raise ImportError("python-dotenv not installed. Run: uv add python-dotenv") from exc

    env_file = Path(__file__).parent.parent / ".env"
    vals = dotenv_values(str(env_file))
    url = vals.get("DATABASE_URL")
    if not url:
        raise ValueError(
            f"DATABASE_URL not found in {env_file}. "
            "Add it or use --local-db-url to specify manually."
        )
    return url


# ---------------------------------------------------------------------------
# User confirmation
# ---------------------------------------------------------------------------


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Prompt user for confirmation.
    Returns True if user confirms, False otherwise.
    """
    suffix = " [y/N]: " if not default else " [Y/n]: "
    try:
        response = input(message + suffix).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted by user.")
        return False


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def build_pg_dump_command(
    output_file: str,
    exclude_tables: list[str],
    data_only: bool,
    schema_only: bool,
) -> list[str]:
    """Build the pg_dump command with appropriate flags."""
    cmd = [
        "pg_dump",
        "--format=custom",  # Use custom format for pg_restore compatibility
        "--verbose",  # Show progress
        "--no-owner",  # Don't output owner commands
        "--no-acl",  # Don't output ACL (grant/revoke) commands
        f"--file={output_file}",
    ]

    # Add exclusions
    for table in exclude_tables:
        cmd.append(f"--exclude-table={table}")

    # Data/schema only options (mutually exclusive)
    if data_only:
        cmd.append("--data-only")
    elif schema_only:
        cmd.append("--schema-only")

    return cmd


def build_pg_restore_command(
    conn: DbConnectionInfo,
    input_file: str,
    data_only: bool,
    schema_only: bool,
) -> list[str]:
    """Build the pg_restore command with appropriate flags."""
    cmd = [
        "pg_restore",
        "--verbose",  # Show progress
        "--no-owner",  # Don't restore owner
        "--no-acl",  # Don't restore ACL
        f"--dbname={conn.database}",
    ]

    # Data/schema only options
    if data_only:
        cmd.append("--data-only")
    elif schema_only:
        cmd.append("--schema-only")

    cmd.append(input_file)

    return cmd


def drop_schema_cascade(conn: DbConnectionInfo, dry_run: bool) -> int:
    """
    Drop and recreate public schema using CASCADE to remove all objects atomically.

    This avoids pg_restore --clean failures caused by local-only tables (e.g. tables
    added by local migrations that don't exist in the source dump) whose FK constraints
    block the DROP of tables that ARE in the dump.
    """
    sql = "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
    cmd = [
        "psql",
        f"--dbname={conn.database}",
        "--no-password",
        "-c",
        sql,
    ]
    env = {**os.environ, **conn.to_env_dict()}

    print("\nDropping and recreating public schema (CASCADE)...")
    print(f"Command: psql --dbname={conn.database} -c \"{sql}\"")

    if dry_run:
        print("[DRY RUN] Would execute schema drop/recreate")
        return 0

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"ERROR: Schema drop/recreate failed with exit code {result.returncode}", file=sys.stderr)
    else:
        print("Schema cleared successfully.")

    return result.returncode


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_pg_dump(
    conn: DbConnectionInfo,
    output_file: str,
    exclude_tables: list[str],
    data_only: bool,
    schema_only: bool,
    dry_run: bool,
) -> int:
    """
    Execute pg_dump to create a database dump.
    Returns exit code (0 = success).
    """
    cmd = build_pg_dump_command(output_file, exclude_tables, data_only, schema_only)
    env = {**os.environ, **conn.to_env_dict()}

    print(f"\n{'=' * 70}")
    print("DUMPING SOURCE DATABASE")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")

    if dry_run:
        print("[DRY RUN] Would execute pg_dump")
        return 0

    print("\nExecuting pg_dump...")
    result = subprocess.run(cmd, env=env, check=False)

    if result.returncode != 0:
        print(f"ERROR: pg_dump failed with exit code {result.returncode}", file=sys.stderr)
    else:
        # Show file size
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Dump complete: {size_mb:.2f} MB")

    return result.returncode


def run_pg_restore(
    conn: DbConnectionInfo,
    input_file: str,
    data_only: bool,
    schema_only: bool,
    dry_run: bool,
) -> int:
    """
    Execute pg_restore to restore database from dump.
    Returns exit code (0 = success).
    """
    cmd = build_pg_restore_command(conn, input_file, data_only, schema_only)
    env = {**os.environ, **conn.to_env_dict()}

    print(f"\n{'=' * 70}")
    print("RESTORING TO LOCAL DATABASE")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Would execute pg_restore")
        return 0

    print("\nExecuting pg_restore...")
    result = subprocess.run(cmd, env=env, check=False)

    if result.returncode != 0:
        # pg_restore often returns non-zero even on partial success
        print(f"WARNING: pg_restore exited with code {result.returncode}", file=sys.stderr)
        print("Some errors may be expected (e.g., objects already exist)")
    else:
        print("Restore complete!")

    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    """Main entry point."""

    # Check tool availability
    available, missing = check_tool_availability()
    if not available:
        print(f"ERROR: Required tools not found: {', '.join(missing)}", file=sys.stderr)
        print("Install PostgreSQL client tools (pg_dump, pg_restore) to proceed.")
        return 1

    # Validate mutually exclusive options
    if args.data_only and args.schema_only:
        print("ERROR: --data-only and --schema-only are mutually exclusive", file=sys.stderr)
        return 1

    # Resolve database URLs
    try:
        source_url = args.source_db_url or resolve_source_db_url(args.source_env)
        local_url = args.local_db_url or resolve_local_db_url()
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Parse connection info
    try:
        source_conn = DbConnectionInfo.from_url(source_url)
        local_conn = DbConnectionInfo.from_url(local_url)
    except ValueError as exc:
        print(f"ERROR: Invalid database URL - {exc}", file=sys.stderr)
        return 1

    # Parse exclude tables
    exclude_tables: list[str] = []
    if args.exclude_tables:
        exclude_tables = [t.strip() for t in args.exclude_tables.split(",") if t.strip()]

    # Display connection info
    print(f"\n{'=' * 70}")
    print("DATABASE CLONE CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Source:      {source_conn.display_safe()}")
    print(f"Destination: {local_conn.display_safe()}")
    print(f"Source env:  {args.source_env}")
    if exclude_tables:
        print(f"Excluded:    {', '.join(exclude_tables)}")
    if args.data_only:
        print("Mode:        Data only (no schema)")
    elif args.schema_only:
        print("Mode:        Schema only (no data)")
    else:
        print("Mode:        Full (schema + data)")
    if args.drop_existing:
        print("Drop:        Will drop existing objects before restore")
    print(f"{'=' * 70}")

    # Safety confirmation
    if not args.dry_run and not args.yes:
        if args.drop_existing:
            msg = (
                f"\nWARNING: This will DROP existing objects in {local_conn.database}!\n"
                "Are you sure you want to proceed?"
            )
        else:
            msg = f"\nThis will restore data into {local_conn.database}. Proceed?"

        if not confirm_action(msg):
            print("Aborted.")
            return 1

    # Create temporary file for dump
    dump_file = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".dump",
            prefix=f"seer_{args.source_env}_",
            delete=False,
        ) as f:
            dump_file = f.name

        print(f"\nUsing temporary dump file: {dump_file}")

        # Step 1: pg_dump
        exit_code = run_pg_dump(
            conn=source_conn,
            output_file=dump_file,
            exclude_tables=exclude_tables,
            data_only=args.data_only,
            schema_only=args.schema_only,
            dry_run=args.dry_run,
        )

        if exit_code != 0 and not args.dry_run:
            print("\nDump failed. Aborting restore.", file=sys.stderr)
            return exit_code

        # Step 2: If dropping existing, nuke the schema first so local-only tables
        # (e.g. from unapplied local migrations) can't block the restore via FK deps.
        if args.drop_existing and not args.data_only:
            exit_code = drop_schema_cascade(local_conn, dry_run=args.dry_run)
            if exit_code != 0 and not args.dry_run:
                print("\nSchema drop failed. Aborting restore.", file=sys.stderr)
                return exit_code

        # Step 3: pg_restore
        exit_code = run_pg_restore(
            conn=local_conn,
            input_file=dump_file,
            data_only=args.data_only,
            schema_only=args.schema_only,
            dry_run=args.dry_run,
        )

        # Summary
        print(f"\n{'=' * 70}")
        if args.dry_run:
            print("[DRY RUN] Clone would complete successfully")
        elif exit_code == 0:
            print("DATABASE CLONE COMPLETE!")
        else:
            print("Clone completed with warnings (see output above)")
        print(f"{'=' * 70}")

        return 0 if args.dry_run else exit_code

    finally:
        # Cleanup
        if dump_file and os.path.exists(dump_file):
            if args.dry_run:
                print(f"\n[DRY RUN] Would clean up: {dump_file}")
            else:
                try:
                    os.remove(dump_file)
                    print(f"\nCleaned up temporary file: {dump_file}")
                except OSError as e:
                    print(f"Warning: Could not remove temp file {dump_file}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clone a hosted database (dev/main) to local Postgres using pg_dump/pg_restore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source environment
    parser.add_argument(
        "--source-env",
        choices=["dev", "main"],
        default="dev",
        help="AWS SSM environment prefix for source DB (default: dev -> /dev/database_url)",
    )

    # URL overrides
    parser.add_argument(
        "--source-db-url",
        default=None,
        help="Override source DB URL directly (skips AWS SSM lookup)",
    )
    parser.add_argument(
        "--local-db-url",
        default=None,
        help="Override local DB URL directly (skips .env DATABASE_URL lookup)",
    )

    # Table exclusion
    parser.add_argument(
        "--exclude-tables",
        default=None,
        help="Comma-separated list of tables to exclude from dump (e.g., 'large_logs,audit_trail')",
    )

    # Dump/restore modes
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only dump/restore data, not schema (assumes schema already exists)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only dump/restore schema, not data",
    )

    # Destructive options
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing database objects before restore (use with caution!)",
    )

    # Safety flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be executed without running them",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts (use in scripts)",
    )

    sys.exit(main(parser.parse_args()))
