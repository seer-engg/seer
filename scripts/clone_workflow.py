#!/usr/bin/env python3
# pylint: disable=duplicate-code  # Reason: _strip_auth_from_spec is intentionally duplicated in standalone scripts to keep them self-contained
"""
Interactive workflow cloner - clone workflows between users via direct DB access.

Features:
- Browse all users and their workflows interactively
- Clone workflows to your account with name deduplication
- Export/import workflows across environments (prod → local)
- Works with dev/prod databases via DATABASE_URL env var

Usage:
    uv run scripts/clone_workflow.py browse
    uv run scripts/clone_workflow.py clone --dry-run
    uv run scripts/clone_workflow.py export --workflow-id wf_139 -o /tmp/wf.json
    uv run scripts/clone_workflow.py import /tmp/wf.json --user-index 0

Environment:
    DATABASE_URL: PostgreSQL connection string (from your .env or config)
"""

import copy
import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional

# Add src to Python path before importing local packages
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# pylint: disable=wrong-import-position
# reason: sys.path must be modified before importing local packages in standalone scripts
import asyncclick as click
from tortoise.expressions import Q

from seer.database import close_db, init_db
from seer.database.models import User
from seer.database.workflow_models import (
    Workflow,
    WorkflowVersion,
    WorkflowVersionStatus,
    parse_workflow_public_id,
)
from seer.logger import get_logger
# pylint: enable=wrong-import-position

logger = get_logger("scripts.clone_workflow")


# ────────────────────────────────────────────────────────────────────────
# Interactive Browser
# ────────────────────────────────────────────────────────────────────────


class WorkflowBrowser:
    """Interactive browser for users and workflows."""

    def __init__(self):
        self.current_user: Optional[User] = None
        self.workflows: list[Workflow] = []
        self.workflow_idx = 0

    async def list_users(self) -> list[User]:
        """Fetch all users from database."""
        return await User.all().order_by("-created_at")

    async def list_workflows(self, user: User) -> list[Workflow]:
        """Fetch all workflows for a user."""
        return await Workflow.filter(user=user).order_by("-updated_at")

    def print_users(self, users: list[User]) -> None:
        """Print user list with index."""
        print("\n" + "=" * 70)
        print("AVAILABLE USERS")
        print("=" * 70)
        if not users:
            print("No users found in database.")
            return
        for i, user in enumerate(users):
            marker = " >>>" if user == self.current_user else "    "
            name = f"{user.first_name} {user.last_name}" if user.first_name and user.last_name else "Untitled"
            wf_count = len(self.workflows) if user == self.current_user else 0
            print(f"{marker} [{i}] {user.user_id[:20]}... | {name} | {user.email or 'no email'}")
            print(f"       workflows: {wf_count}")
        print("=" * 70)

    def print_workflows(self) -> None:
        """Print workflows for current user."""
        if not self.current_user:
            print("\nNo user selected. Use 'u <index>' to select a user.")
            return
        print("\n" + "=" * 70)
        print(f"WORKFLOWS FOR {self.current_user.email or self.current_user.user_id}")
        print("=" * 70)
        if not self.workflows:
            print("No workflows found for this user.")
            print("=" * 70)
            return
        for i, wf in enumerate(self.workflows):
            marker = " >>>" if wf.id == self.workflows[self.workflow_idx].id else "    "
            print(f"{marker} [{i}] {wf.workflow_id} | {wf.name}")
        print("=" * 70)

    def select_workflow(self, index: int) -> bool:
        """Select a workflow by index."""
        if 0 <= index < len(self.workflows):
            self.workflow_idx = index
            return True
        return False


# ────────────────────────────────────────────────────────────────────────
# Core Logic
# ────────────────────────────────────────────────────────────────────────


def _strip_auth_from_spec(spec: dict) -> None:
    """
    Strip user-specific auth from a workflow spec in-place.

    Removes provider_connection_id from triggers, connection_id from tool/agent
    nodes, and auth from mcp nodes.
    """
    for trigger in spec.get("triggers") or []:
        trigger.get("provider_config", {}).pop("provider_connection_id", None)

    for node in spec.get("nodes") or []:
        node_type = node.get("type")
        if node_type == "tool":
            node.pop("connection_id", None)
        elif node_type == "agent":
            for tool in node.get("inputs", {}).get("tools") or []:
                if isinstance(tool, dict):
                    tool.pop("connection_id", None)
        elif node_type == "mcp":
            node["auth"] = None


async def _dedupe_name(base_name: str, user: User) -> str:
    """Return a unique workflow name for the given user."""
    counter = 1
    name = base_name
    while await Workflow.filter(user=user, name=name).exists():
        name = f"{base_name} ({counter})"
        counter += 1
    return name


async def clone_workflow(source_workflow_id: int, target_user: User, new_name: Optional[str] = None) -> Workflow:
    """Clone a workflow to a target user, stripping auth and deduplicating the name."""
    source = await Workflow.get(id=source_workflow_id)
    source_version = await WorkflowVersion.get(
        workflow=source, version_number=0, status=WorkflowVersionStatus.DRAFT
    )

    base_name = new_name or f"{source.name} (clone)"
    counter = 1
    while await Workflow.filter(user=target_user, name__startswith=base_name).exists():
        base_name = f"{new_name or source.name} ({counter})"
        counter += 1

    new_workflow = await Workflow.create(user=target_user, name=base_name)

    new_spec = copy.deepcopy(source_version.spec)
    _strip_auth_from_spec(new_spec)
    new_spec_hash = sha256(json.dumps(new_spec, sort_keys=True).encode()).hexdigest()

    await WorkflowVersion.create(
        workflow=new_workflow,
        spec=new_spec,
        version_number=0,
        status=WorkflowVersionStatus.DRAFT,
        spec_hash=new_spec_hash,
        created_by=target_user,
        updated_by=target_user,
    )

    logger.info("Cloned workflow %s to %s", source.workflow_id, new_workflow.workflow_id)
    return new_workflow


# ────────────────────────────────────────────────────────────────────────
# CLI Helpers
# ────────────────────────────────────────────────────────────────────────


async def _init_db_or_exit() -> None:
    """Initialize DB connection or exit with a user-friendly error."""
    try:
        await init_db()
    except Exception as e:  # pylint: disable=broad-exception-caught  # reason: surface any DB error as CLI message
        if not os.environ.get("DATABASE_URL"):
            click.secho(
                "Error: DATABASE_URL is not set. Set it via:\n"
                "  DATABASE_URL=your_url uv run scripts/clone_workflow.py\n"
                "  or use --database-url 'your_url'",
                fg="red",
            )
        else:
            click.secho(f"Failed to connect to database: {e}", fg="red")
        sys.exit(1)


async def _prompt_index(prompt: str, count: int) -> int:
    """Prompt until the user enters a valid index in [0, count)."""
    while True:
        choice = await click.prompt(prompt, type=str)
        if choice.lower() == "q":
            click.echo("Exiting.")
            sys.exit(0)
        try:
            idx = int(choice)
            if 0 <= idx < count:
                return idx
            click.secho(f"Invalid index: {idx}", fg="red")
        except ValueError:
            click.secho("Enter a number", fg="yellow")


async def _select_user(browser: WorkflowBrowser, users: list[User], user_index: Optional[int]) -> User:
    """Select a user by index, auto-select if only one, or prompt interactively."""
    if user_index is not None:
        if not 0 <= user_index < len(users):
            click.secho(f"Invalid user index: {user_index}", fg="red")
            sys.exit(1)
        return users[user_index]
    if len(users) == 1:
        click.echo(f"Auto-selected user: {users[0].user_id[:20]}...")
        return users[0]
    browser.print_users(users)
    idx = await _prompt_index("Select user", len(users))
    return users[idx]


async def _select_export_source(browser: WorkflowBrowser, workflow_id: Optional[str]) -> Workflow:
    """Resolve the source workflow for export: by ID or interactively."""
    if workflow_id:
        try:
            pk = parse_workflow_public_id(workflow_id)
        except ValueError:
            click.secho(f"Invalid workflow ID format: '{workflow_id}' (expected e.g. wf_139)", fg="red")
            sys.exit(1)
        source = await Workflow.get_or_none(id=pk)
        if not source:
            click.secho(f"Workflow '{workflow_id}' not found.", fg="red")
            sys.exit(1)
        return source

    users = await browser.list_users()
    if not users:
        click.secho("No users found.", fg="red")
        sys.exit(1)
    browser.print_users(users)
    user_idx = await _prompt_index("Select user index", len(users))
    browser.current_user = users[user_idx]

    browser.workflows = await browser.list_workflows(browser.current_user)
    browser.print_workflows()
    if not browser.workflows:
        click.secho("No workflows found for this user.", fg="red")
        sys.exit(1)
    wf_idx = await _prompt_index("Select workflow index", len(browser.workflows))
    return browser.workflows[wf_idx]


# ────────────────────────────────────────────────────────────────────────
# CLI Commands
# ────────────────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--database-url",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string (defaults to DATABASE_URL env var)",
)
def main(database_url: Optional[str]):
    """Interactive workflow cloner - browse, clone, export, and import workflows.

    Examples:
        uv run scripts/clone_workflow.py browse
        uv run scripts/clone_workflow.py clone --dry-run
        uv run scripts/clone_workflow.py export --workflow-id wf_139 -o /tmp/wf.json
        uv run scripts/clone_workflow.py import /tmp/wf.json --user-index 0
    """
    if database_url:
        os.environ["DATABASE_URL"] = database_url


@main.command()
@click.option("--user-index", type=int, help="User index to clone from")
@click.option("--target-user", type=int, help="Target user index to clone to")
@click.option("--name", help="Custom name for cloned workflow")
@click.option("--workflow-index", type=int, help="Workflow index to clone")
@click.option("--list-only", is_flag=True, help="Just list users and exit")
@click.option("--dry-run", is_flag=True, help="Show what would be cloned without cloning")
# pylint: disable=too-many-positional-arguments  # reason: click maps each option to a positional parameter
async def clone(
    user_index: Optional[int],
    target_user: Optional[int],
    name: Optional[str],
    workflow_index: Optional[int],
    list_only: bool,
    dry_run: bool,
):
    """Interactive workflow cloning session."""
    click.echo("Initializing database connection...")
    await _init_db_or_exit()

    try:  # pylint: disable=too-many-branches  # reason: interactive CLI flow has multiple selection branches
        browser = WorkflowBrowser()
        users = await browser.list_users()
        click.echo(f"\nFound {len(users)} users in database.")

        if list_only:
            browser.print_users(users)
            return

        browser.current_user = await _select_user(browser, users, user_index)
        browser.workflows = await browser.list_workflows(browser.current_user)
        browser.print_workflows()

        if workflow_index is not None:
            if not 0 <= workflow_index < len(browser.workflows):
                click.secho(f"Invalid workflow index: {workflow_index}", fg="red")
                sys.exit(1)
            browser.workflow_idx = workflow_index
            click.echo(f"Selected workflow: {browser.workflows[workflow_index].workflow_id}")
        elif len(browser.workflows) == 1:
            click.echo(f"Auto-selected workflow: {browser.workflows[0].workflow_id}")
        else:
            browser.workflow_idx = await _prompt_index("Select workflow to clone", len(browser.workflows))

        if target_user is None and len(users) > 1:
            browser.print_users(users)
            target_user = await _prompt_index("Select target user", len(users))

        target_user_obj = users[target_user] if target_user is not None else browser.current_user

        source_workflow = browser.workflows[browser.workflow_idx]
        click.echo(f"\n{'=' * 70}")
        click.echo(f"CLONING: {source_workflow.workflow_id} -> {target_user_obj.user_id[:20]}...")
        if dry_run:
            click.secho("[DRY RUN] - No changes will be made", fg="yellow")
        click.echo(f"{'=' * 70}")

        if dry_run:
            click.echo(f"Would create: {name or source_workflow.name + ' (clone)'}")
            return

        new_workflow = await clone_workflow(source_workflow.id, target_user_obj, new_name=name)
        click.secho(f"\nSUCCESS! Cloned to: {new_workflow.workflow_id} ({new_workflow.name})", fg="green")

    except Exception as e:  # pylint: disable=broad-exception-caught  # reason: surface any error as CLI message
        logger.exception("Cloning failed")
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    finally:
        await close_db()


@main.command()
@click.option("--user", help="Filter by user email or name (partial match)")
@click.option("--workflow", help="Filter workflows by name (partial match)")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
async def browse(user: Optional[str], workflow: Optional[str], fmt: str):
    """Browse all users and workflows (non-interactive)."""
    try:
        await init_db()
        users = await User.all().order_by("-created_at")
        result = []

        for u in users:
            user_workflows = await Workflow.filter(user=u).order_by("-updated_at")

            if user:
                matches = (
                    user.lower() in (u.email or "").lower()
                    or user.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                    or user.lower() in u.user_id.lower()
                )
                if not matches:
                    continue

            for wf in user_workflows:
                if workflow and workflow.lower() not in wf.name.lower():
                    continue
                result.append({
                    "user_id": u.user_id[:30],
                    "email": u.email,
                    "name": f"{u.first_name or ''} {u.last_name or ''}".strip(),
                    "workflow_id": wf.workflow_id,
                    "workflow_name": wf.name,
                    "updated_at": wf.updated_at.isoformat(),
                })

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            if not result:
                click.echo("No workflows found matching filters.")
                return
            click.echo(f"\n{'=' * 80}")
            click.echo(f"{'USER':<35} | {'WORKFLOW':<45}")
            click.echo(f"{'=' * 80}")
            for r in result:
                user_str = f"{r['name']} ({r['user_id'][:20]}...)"
                wf_str = f"{r['workflow_id']} | {r['workflow_name']}"
                click.echo(f"{user_str:<35} | {wf_str:<45}")
            click.echo(f"{'=' * 80}\n")
            click.echo(f"Total: {len(result)} workflows")

    finally:
        await close_db()


@main.command()
@click.argument("user_id")
@click.option("--email", is_flag=True, help="Search by email instead of user_id")
@click.option("--name", is_flag=True, help="Search by name instead of user_id")
async def find_user(user_id: str, email: bool, name: bool):
    """Find a user by user_id, email, or name."""
    try:
        await init_db()
        query = User.all()
        if email:
            query = query.filter(email__icontains=user_id)
        elif name:
            query = query.filter(Q(first_name__icontains=user_id) | Q(last_name__icontains=user_id))
        else:
            query = query.filter(user_id__icontains=user_id)

        users = await query
        if not users:
            click.secho(f"No user found matching '{user_id}'", fg="red")
            sys.exit(1)
        for u in users:
            name_str = f"{u.first_name or ''} {u.last_name or ''}".strip()
            click.echo(f"[{u.id}] {u.user_id[:30]}... | {name_str} | {u.email}")

    except Exception as e:  # pylint: disable=broad-exception-caught  # reason: surface any error as CLI message
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    finally:
        await close_db()


@main.command("export")
@click.option("--workflow-id", help="Workflow ID string (e.g. wf_139) to export directly")
@click.option("--output", "-o", help="Output file path (default: <workflow_name>.json)")
async def export_workflow(workflow_id: Optional[str], output: Optional[str]):
    """Export a workflow spec to a JSON file.

    Examples:
        DATABASE_URL=$PROD_URL uv run scripts/clone_workflow.py export --workflow-id wf_139 -o /tmp/wf.json
    """
    await _init_db_or_exit()
    try:
        browser = WorkflowBrowser()
        source = await _select_export_source(browser, workflow_id)

        source_version = await WorkflowVersion.get_or_none(
            workflow=source, version_number=0, status=WorkflowVersionStatus.DRAFT
        )
        if not source_version:
            click.secho(f"No draft version found for {source.workflow_id}.", fg="red")
            sys.exit(1)

        spec = copy.deepcopy(source_version.spec)
        _strip_auth_from_spec(spec)

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_workflow_id": source.workflow_id,
            "source_workflow_name": source.name,
            "spec": spec,
        }

        out_path = output or f"{source.name.replace('/', '_')}.json"
        Path(out_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        click.secho(f"Exported '{source.name}' ({source.workflow_id}) → {out_path}", fg="green")

    except Exception as e:  # pylint: disable=broad-exception-caught  # reason: surface any error as CLI message
        logger.exception("Export failed")
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    finally:
        await close_db()


@main.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", help="Override workflow name")
@click.option("--user-index", type=int, help="Target user index (skip interactive selection)")
@click.option("--dry-run", is_flag=True, help="Show what would be created without writing")
async def import_workflow(file: str, name: Optional[str], user_index: Optional[int], dry_run: bool):
    """Import a workflow from an exported JSON file.

    Examples:
        DATABASE_URL=$LOCAL_URL uv run scripts/clone_workflow.py import /tmp/wf.json --user-index 0
    """
    payload = json.loads(Path(file).read_text(encoding="utf-8"))
    if "spec" not in payload:
        click.secho("Invalid export file: missing 'spec' key.", fg="red")
        sys.exit(1)

    await _init_db_or_exit()
    try:
        browser = WorkflowBrowser()
        users = await browser.list_users()
        if not users:
            click.secho("No users found in target database.", fg="red")
            sys.exit(1)

        target_user = await _select_user(browser, users, user_index)
        final_name = await _dedupe_name(name or payload.get("source_workflow_name", "Imported Workflow"), target_user)

        click.echo(f"\n{'=' * 70}")
        click.echo(f"IMPORTING: '{final_name}' → {target_user.email or target_user.user_id[:20]}")
        click.echo(f"Source: {payload.get('source_workflow_id')} (exported {payload.get('exported_at', 'unknown')})")
        if dry_run:
            click.secho("[DRY RUN] - No changes will be made", fg="yellow")
        click.echo(f"{'=' * 70}")

        if dry_run:
            return

        spec = payload["spec"]
        spec_hash = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        new_workflow = await Workflow.create(user=target_user, name=final_name)
        await WorkflowVersion.create(
            workflow=new_workflow,
            spec=spec,
            version_number=0,
            status=WorkflowVersionStatus.DRAFT,
            spec_hash=spec_hash,
            created_by=target_user,
            updated_by=target_user,
        )
        click.secho(f"\nSUCCESS! Created: {new_workflow.workflow_id} ({new_workflow.name})", fg="green")

    except Exception as e:  # pylint: disable=broad-exception-caught  # reason: surface any error as CLI message
        logger.exception("Import failed")
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    try:
        main()  # pylint: disable=no-value-for-parameter  # reason: click injects arguments from decorators
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
