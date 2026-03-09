#!/usr/bin/env python3
# pylint: disable=too-many-lines,duplicate-code  # Reason: TABLE_CONFIGS list can't be shortened; _strip_auth_from_spec intentionally duplicated in standalone scripts
"""
Clone a production user's data to local Postgres.

All FK references to the source user are remapped to the local developer's
user (default: postgres id=1). Encrypted fields are copied verbatim — they
only work locally if LOCAL_ENCRYPTION_KEY matches production.

Prod DB URL  → fetched from AWS SSM at /{prod-env}/database_url (requires AWS creds).
Local DB URL → read from DATABASE_URL in .env file.

Both can be overridden with explicit CLI flags.

Usage:
    uv run scripts/clone_user_from_prod.py \\
      --source-user <clerk_user_id_or_pg_id> \\
      [--prod-env main]          # SSM path prefix, default: main
      [--local-user-id 1]        # local postgres user to remap into, default: 1
      [--dry-run]                # print counts, don't insert
      [--truncate-existing]      # wipe previously cloned data first
      [--include-credentials]    # also clone oauth_connections, integration_resources, integration_secrets

    # Override URLs directly instead of using SSM / .env
    uv run scripts/clone_user_from_prod.py \\
      --source-user user_2abc123 \\
      --prod-db-url "postgres://..." \\
      --local-db-url "postgres://..."

Examples:
    # Dry run — preview row counts
    uv run scripts/clone_user_from_prod.py --source-user user_2abc123 --dry-run

    # Full clone
    uv run scripts/clone_user_from_prod.py --source-user user_2abc123

    # Wipe and re-clone
    uv run scripts/clone_user_from_prod.py --source-user user_2abc123 --truncate-existing

    # Checkpoint data (LangGraph) is cloned automatically — no extra flag needed.
"""

import argparse
import asyncio
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src/ to path so we can import seer modules (AwsSsmSettingsSource, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import asyncpg
except ImportError:
    print("asyncpg not found. Install with: uv add asyncpg", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Table configuration
# ---------------------------------------------------------------------------


@dataclass
# pylint: disable=too-many-instance-attributes  # Reason: deliberate data container — all 13 fields are used by fetch/remap/insert logic
class TableConfig:
    """Describes how to fetch, remap, and insert rows for one table."""

    table: str
    pk_col: str = "id"
    pk_type: str = "int"  # "int" or "uuid"
    # WHERE clause source:
    #   "user_id"           → WHERE fetch_col = source_user_id
    #   "id_map:<table>"    → WHERE fetch_col = ANY(id_map[<table>].keys())
    fetch_col: str = "user_id"
    fetch_source: str = "user_id"
    # Columns that store the owning user's FK — always remapped to local_user_id
    user_cols: list[str] = field(default_factory=list)
    # Nullable user FK columns — remapped to local_user_id only when non-null
    nullable_user_cols: list[str] = field(default_factory=list)
    # {col_name: referenced_table} for integer FK columns
    int_fk_cols: dict[str, str] = field(default_factory=dict)
    # {col_name: referenced_table} for UUID FK columns
    uuid_fk_cols: dict[str, str] = field(default_factory=dict)
    # bytea columns — passed through as raw bytes (asyncpg handles natively)
    binary_cols: list[str] = field(default_factory=list)
    # pgvector columns — fetched as ::text, inserted as ::vector
    vector_cols: list[str] = field(default_factory=list)
    # encrypted columns — copied verbatim, only work if encryption key matches
    encrypted_cols: list[str] = field(default_factory=list)
    # True if this table has a workflow_run_id CharField (e.g. "run_123") to remap
    workflow_run_id_str: bool = False


# Tables skipped by default — tokens are encrypted with prod key and won't work locally
CREDENTIAL_TABLES: frozenset[str] = frozenset({
    "oauth_connections",
    "integration_resources",
    "integration_secrets",
})


# Insertion order: parents before children (respects all FK dependencies)
TABLE_CONFIGS: list[TableConfig] = [
    # ── User-owned, no parent FKs ──────────────────────────────────────────
    TableConfig(
        table="user_settings",
        user_cols=["user_id"],
    ),
    TableConfig(
        table="billing_profiles",
        fetch_col="owner_user_id",
        user_cols=["owner_user_id"],
    ),
    # ── Children of billing_profiles ───────────────────────────────────────
    TableConfig(
        table="billing_subscriptions",
        fetch_col="billing_profile_id",
        fetch_source="id_map:billing_profiles",
        int_fk_cols={"billing_profile_id": "billing_profiles"},
    ),
    TableConfig(
        table="overage_settings",
        fetch_col="billing_profile_id",
        fetch_source="id_map:billing_profiles",
        int_fk_cols={"billing_profile_id": "billing_profiles"},
    ),
    # ── OAuth / integrations ───────────────────────────────────────────────
    TableConfig(
        table="oauth_connections",
        user_cols=["user_id"],
        encrypted_cols=["access_token_enc", "refresh_token_enc"],
    ),
    TableConfig(
        table="integration_resources",
        user_cols=["user_id"],
        int_fk_cols={"oauth_connection_id": "oauth_connections"},
    ),
    TableConfig(
        table="integration_secrets",
        user_cols=["user_id"],
        int_fk_cols={
            "oauth_connection_id": "oauth_connections",
            "resource_id": "integration_resources",
        },
        encrypted_cols=["value_enc"],
    ),
    # ── Browser ────────────────────────────────────────────────────────────
    TableConfig(
        table="browser_profiles",
        pk_type="uuid",
        user_cols=["user_id"],
        encrypted_cols=["session_state_enc"],
    ),
    # ── Knowledge base hierarchy ───────────────────────────────────────────
    TableConfig(
        table="knowledge_bases",
        user_cols=["user_id"],
    ),
    TableConfig(
        table="knowledge_documents",
        fetch_col="knowledge_base_id",
        fetch_source="id_map:knowledge_bases",
        int_fk_cols={"knowledge_base_id": "knowledge_bases"},
    ),
    TableConfig(
        table="knowledge_chunks",
        fetch_col="knowledge_base_id",
        fetch_source="id_map:knowledge_bases",
        int_fk_cols={
            "knowledge_base_id": "knowledge_bases",
            "document_id": "knowledge_documents",
        },
        vector_cols=["embedding"],
    ),
    # ── Usage ──────────────────────────────────────────────────────────────
    TableConfig(
        table="usage_counters",
        user_cols=["user_id"],
    ),
    # ── Workflow hierarchy ─────────────────────────────────────────────────
    TableConfig(
        table="workflows",
        user_cols=["user_id"],
    ),
    TableConfig(
        table="workflow_versions",
        fetch_col="workflow_id",
        fetch_source="id_map:workflows",
        int_fk_cols={"workflow_id": "workflows"},
        nullable_user_cols=["created_by_id", "updated_by_id"],
    ),
    TableConfig(
        table="workflow_records",
        user_cols=["user_id"],
    ),
    # ── Triggers (must come before workflow_runs that reference them) ───────
    TableConfig(
        table="trigger_subscriptions",
        user_cols=["user_id"],
        int_fk_cols={"workflow_id": "workflows"},
    ),
    TableConfig(
        table="trigger_events",
        fetch_col="subscription_id",
        fetch_source="id_map:trigger_subscriptions",
        int_fk_cols={"subscription_id": "trigger_subscriptions"},
    ),
    # ── Workflow runs and children ─────────────────────────────────────────
    TableConfig(
        table="workflow_runs",
        user_cols=["user_id"],
        int_fk_cols={
            "workflow_id": "workflows",
            "workflow_version_id": "workflow_versions",
            "subscription_id": "trigger_subscriptions",
            "trigger_event_id": "trigger_events",
        },
    ),
    TableConfig(
        table="workflow_files",
        user_cols=["user_id"],
        int_fk_cols={"workflow_run_id": "workflow_runs"},
    ),
    # ── Chat sessions ──────────────────────────────────────────────────────
    TableConfig(
        table="workflow_chat_sessions",
        user_cols=["user_id"],
        int_fk_cols={"workflow_id": "workflows"},
    ),
    TableConfig(
        table="workflow_discovery_chat_sessions",
        user_cols=["user_id"],
        int_fk_cols={"created_workflow_id": "workflows"},
    ),
    TableConfig(
        table="workflow_proposals",
        fetch_col="created_by_id",
        user_cols=["created_by_id"],
        int_fk_cols={
            "workflow_id": "workflows",
            "session_id": "workflow_chat_sessions",
        },
    ),
    TableConfig(
        table="workflow_chat_messages",
        fetch_col="session_id",
        fetch_source="id_map:workflow_chat_sessions",
        int_fk_cols={
            "session_id": "workflow_chat_sessions",
            "proposal_id": "workflow_proposals",
        },
    ),
    # ── LLM usage and overage ──────────────────────────────────────────────
    TableConfig(
        table="llm_usage_records",
        user_cols=["user_id"],
        workflow_run_id_str=True,
    ),
    TableConfig(
        table="overage_usage_records",
        fetch_col="overage_settings_id",
        fetch_source="id_map:overage_settings",
        int_fk_cols={
            "overage_settings_id": "overage_settings",
            "llm_usage_record_id": "llm_usage_records",
        },
    ),
    # ── Session recordings ─────────────────────────────────────────────────
    TableConfig(
        table="session_recordings",
        pk_type="uuid",
        user_cols=["user_id"],
        uuid_fk_cols={"browser_profile_id": "browser_profiles"},
        binary_cols=["events_compressed"],
        workflow_run_id_str=True,
    ),
    # ── Templates (nullable user FK) ───────────────────────────────────────
    TableConfig(
        table="workflow_templates",
        fetch_col="created_by_id",
        nullable_user_cols=["created_by_id"],
    ),
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


async def setup_conn(conn: asyncpg.Connection) -> None:
    """Register JSON codecs so jsonb columns are decoded as Python objects."""
    await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def resolve_user(conn: asyncpg.Connection, source: str) -> asyncpg.Record:
    """Resolve source user from clerk ID (user_xxx) or postgres integer PK."""
    if source.isdigit():
        row = await conn.fetchrow("SELECT id, user_id, email FROM users WHERE id = $1", int(source))
    else:
        row = await conn.fetchrow("SELECT id, user_id, email FROM users WHERE user_id = $1", source)
    if not row:
        raise ValueError(f"User not found: {source!r}")
    return row


async def get_column_names(conn: asyncpg.Connection, table: str) -> list[str]:
    """Return ordered column names for a table from information_schema."""
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 "
        "ORDER BY ordinal_position",
        table,
    )
    return [r["column_name"] for r in rows]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


async def fetch_rows(
    conn: asyncpg.Connection,
    config: TableConfig,
    source_user_id: int,
    id_map: dict[str, dict],
) -> list[asyncpg.Record]:
    """Fetch all matching rows from the source (prod) database."""
    col_names = await get_column_names(conn, config.table)
    if not col_names:
        print(f"  WARNING: table {config.table!r} not found in prod DB, skipping.")
        return []

    # Cast pgvector columns to text so asyncpg doesn't choke on unknown type
    select_parts = [
        f"{col}::text AS {col}" if col in config.vector_cols else col
        for col in col_names
    ]
    select_clause = ", ".join(select_parts)

    if config.fetch_source == "user_id":
        sql = f"SELECT {select_clause} FROM {config.table} WHERE {config.fetch_col} = $1"
        return list(await conn.fetch(sql, source_user_id))

    # "id_map:<parent_table>" — fetch rows whose parent FK is one of the already-mapped IDs
    _, parent_table = config.fetch_source.split(":", 1)
    parent_ids = list(id_map.get(parent_table, {}).keys())
    if not parent_ids:
        return []
    sql = f"SELECT {select_clause} FROM {config.table} WHERE {config.fetch_col} = ANY($1::int[])"
    return list(await conn.fetch(sql, parent_ids))


# ---------------------------------------------------------------------------
# Remap
# ---------------------------------------------------------------------------


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


def remap_row(  # pylint: disable=too-complex  # Reason: 5 sequential flat remap passes (pk, user FKs, int FKs, uuid FKs, run_id string) — not real branching complexity
    row: asyncpg.Record,
    config: TableConfig,
    id_map: dict[str, dict],
    local_user_id: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Translate a prod row into local IDs.

    Returns (old_pk, remapped_data_dict). The old_pk is used to populate
    id_map so downstream tables can find the new local ID.
    """
    d = dict(row)
    old_pk = d[config.pk_col]

    # Preserve prod PKs so thread_id / run_id references stay consistent.
    # (UUID values come back as uuid.UUID objects; stringify for downstream use)
    if config.pk_type == "uuid":
        d[config.pk_col] = str(old_pk)

    # Required user FK columns → always local_user_id
    for col in config.user_cols:
        d[col] = local_user_id

    # Nullable user FK columns → local_user_id only when set
    for col in config.nullable_user_cols:
        if d.get(col) is not None:
            d[col] = local_user_id

    # Integer FK columns → look up new local ID in id_map; set None if unmapped
    for col, ref_table in config.int_fk_cols.items():
        old_val = d.get(col)
        if old_val is not None:
            d[col] = id_map.get(ref_table, {}).get(old_val)  # None if not cloned

    # UUID FK columns (e.g. browser_profile_id on session_recordings)
    for col, ref_table in config.uuid_fk_cols.items():
        old_val = d.get(col)
        if old_val is not None:
            d[col] = id_map.get(ref_table, {}).get(str(old_val))

    # workflow_run_id is stored as "run_<int>" string, not a real FK column
    if config.workflow_run_id_str:
        run_id_str = d.get("workflow_run_id")
        if run_id_str:
            match = re.fullmatch(r"run_(\d+)", str(run_id_str))
            if match:
                old_run_id = int(match.group(1))
                new_run_id = id_map.get("workflow_runs", {}).get(old_run_id)
                d["workflow_run_id"] = f"run_{new_run_id}" if new_run_id else None

    # Strip prod connection IDs from workflow specs — they don't exist locally
    if config.table == "workflow_versions" and d.get("spec"):
        _strip_auth_from_spec(d["spec"])

    return old_pk, d


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


async def insert_row(
    conn: asyncpg.Connection,
    config: TableConfig,
    data: dict[str, Any],
) -> Any:
    """
    Insert one remapped row into the local DB. Returns the new PK.

    pgvector columns use a ::vector cast; all others use plain $N parameters.
    """
    cols = list(data.keys())
    values = list(data.values())

    placeholders = [
        f"${i + 1}::vector" if (cols[i] in config.vector_cols and values[i] is not None) else f"${i + 1}"
        for i in range(len(cols))
    ]

    sql = (
        f"INSERT INTO {config.table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(placeholders)}) "
        f"RETURNING {config.pk_col}"
    )
    result = await conn.fetchval(sql, *values)
    return str(result) if config.pk_type == "uuid" else result


# ---------------------------------------------------------------------------
# Checkpoint cloning (LangGraph AsyncPostgresSaver tables)
# ---------------------------------------------------------------------------


async def clone_checkpoint_tables(
    prod_conn: asyncpg.Connection,
    local_conn: asyncpg.Connection,
    thread_ids: list[str],
    dry_run: bool,
) -> int:
    """
    Clone LangGraph checkpoint rows for the given thread_ids from prod to local.

    Handles checkpoints, checkpoint_blobs, checkpoint_writes.
    checkpoint_migrations is skipped — it stores schema versioning only, not user data.

    Returns total number of rows cloned (or that would be cloned in dry-run).
    """
    if not thread_ids:
        print("  (no thread_ids found — skipping checkpoint tables)")
        return 0

    checkpoint_tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
    total = 0

    for table in checkpoint_tables:
        col_names = await get_column_names(prod_conn, table)
        if not col_names:
            print(f"  {table}: not found in prod DB, skipping.")
            continue

        rows = await prod_conn.fetch(
            f"SELECT * FROM {table} WHERE thread_id = ANY($1::text[])",
            thread_ids,
        )

        if dry_run:
            print(f"  {table}: {len(rows)} rows  [dry-run]")
            total += len(rows)
            continue

        if not rows:
            print(f"  {table}: 0 rows")
            continue

        col_list = ", ".join(col_names)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(col_names)))
        sql = (
            f"INSERT INTO {table} ({col_list}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

        inserted = 0
        try:
            async with local_conn.transaction():
                for row in rows:
                    values = [row[col] for col in col_names]
                    await local_conn.execute(sql, *values)
                    inserted += 1
            print(f"  {table}: {inserted} rows inserted")
            total += inserted
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: one bad checkpoint table must not abort the full clone
            print(f"  {table}: ERROR — {exc}")
            traceback.print_exc()

    return total


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


async def truncate_user_data(  # pylint: disable=too-complex  # Reason: sequential FK-ordered deletion across many tables; splitting would obscure the deletion order
    conn: asyncpg.Connection, local_user_id: int) -> None:
    """
    Delete all previously cloned rows for local_user_id in reverse FK order.
    Uses subqueries for child-only tables that have no direct user FK.
    """
    print(f"Truncating existing data for local user id={local_user_id} ...")

    # ── Collect thread_ids BEFORE deleting app rows ───────────────────────
    _all_thread_ids: list[str] = []
    for _t in ("workflow_runs", "workflow_chat_sessions", "workflow_discovery_chat_sessions"):
        try:
            _rows = await conn.fetch(
                f"SELECT thread_id FROM {_t} WHERE user_id = $1 AND thread_id IS NOT NULL",
                local_user_id,
            )
            _all_thread_ids.extend(r["thread_id"] for r in _rows)
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: thread_id column may not exist in older local envs
            pass

    if _all_thread_ids:
        for _ckpt in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            try:
                _status = await conn.execute(
                    f"DELETE FROM {_ckpt} WHERE thread_id = ANY($1::text[])",
                    _all_thread_ids,
                )
                _count = _status.split()[-1] if _status else "0"
                if _count != "0":
                    print(f"  {_ckpt}: deleted {_count} rows")
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: checkpoint tables may not exist in fresh local envs
                pass

    queries: list[tuple[str, str]] = [
        # Deepest leaves first
        ("workflow_templates", "DELETE FROM workflow_templates WHERE created_by_id = $1"),
        ("session_recordings", "DELETE FROM session_recordings WHERE user_id = $1"),
        (
            "overage_usage_records",
            "DELETE FROM overage_usage_records WHERE overage_settings_id IN "
            "(SELECT id FROM overage_settings WHERE billing_profile_id IN "
            "(SELECT id FROM billing_profiles WHERE owner_user_id = $1))",
        ),
        ("llm_usage_records", "DELETE FROM llm_usage_records WHERE user_id = $1"),
        (
            "workflow_chat_messages",
            "DELETE FROM workflow_chat_messages WHERE session_id IN "
            "(SELECT id FROM workflow_chat_sessions WHERE user_id = $1)",
        ),
        ("workflow_proposals", "DELETE FROM workflow_proposals WHERE created_by_id = $1"),
        ("workflow_discovery_chat_sessions", "DELETE FROM workflow_discovery_chat_sessions WHERE user_id = $1"),
        ("workflow_chat_sessions", "DELETE FROM workflow_chat_sessions WHERE user_id = $1"),
        ("workflow_files", "DELETE FROM workflow_files WHERE user_id = $1"),
        ("workflow_runs", "DELETE FROM workflow_runs WHERE user_id = $1"),
        (
            "trigger_events",
            "DELETE FROM trigger_events WHERE subscription_id IN "
            "(SELECT id FROM trigger_subscriptions WHERE user_id = $1)",
        ),
        ("trigger_subscriptions", "DELETE FROM trigger_subscriptions WHERE user_id = $1"),
        ("workflow_records", "DELETE FROM workflow_records WHERE user_id = $1"),
        (
            "workflow_versions",
            "DELETE FROM workflow_versions WHERE workflow_id IN "
            "(SELECT id FROM workflows WHERE user_id = $1)",
        ),
        ("workflows", "DELETE FROM workflows WHERE user_id = $1"),
        ("usage_counters", "DELETE FROM usage_counters WHERE user_id = $1"),
        (
            "knowledge_chunks",
            "DELETE FROM knowledge_chunks WHERE knowledge_base_id IN "
            "(SELECT id FROM knowledge_bases WHERE user_id = $1)",
        ),
        (
            "knowledge_documents",
            "DELETE FROM knowledge_documents WHERE knowledge_base_id IN "
            "(SELECT id FROM knowledge_bases WHERE user_id = $1)",
        ),
        ("knowledge_bases", "DELETE FROM knowledge_bases WHERE user_id = $1"),
        ("browser_profiles", "DELETE FROM browser_profiles WHERE user_id = $1"),
        ("integration_secrets", "DELETE FROM integration_secrets WHERE user_id = $1"),
        ("integration_resources", "DELETE FROM integration_resources WHERE user_id = $1"),
        ("oauth_connections", "DELETE FROM oauth_connections WHERE user_id = $1"),
        (
            "overage_settings",
            "DELETE FROM overage_settings WHERE billing_profile_id IN "
            "(SELECT id FROM billing_profiles WHERE owner_user_id = $1)",
        ),
        (
            "billing_subscriptions",
            "DELETE FROM billing_subscriptions WHERE billing_profile_id IN "
            "(SELECT id FROM billing_profiles WHERE owner_user_id = $1)",
        ),
        ("billing_profiles", "DELETE FROM billing_profiles WHERE owner_user_id = $1"),
        ("user_settings", "DELETE FROM user_settings WHERE user_id = $1"),
    ]

    for table, sql in queries:
        status = await conn.execute(sql, local_user_id)
        count = status.split()[-1] if status else "0"
        if count != "0":
            print(f"  {table}: deleted {count} rows")


# ---------------------------------------------------------------------------
# URL resolution (config / SSM / .env)
# ---------------------------------------------------------------------------


def resolve_prod_db_url(prod_env: str) -> str:
    """
    Fetch the production DATABASE_URL from AWS SSM at /{prod_env}/database_url.
    Requires valid AWS credentials in the environment (IAM role, profile, etc.).
    """
    # pylint: disable=import-outside-toplevel  # Reason: seer imports need sys.path set up first
    from seer.config import SeerConfig
    from seer.utilities.aws.parameter_store import AwsSsmSettingsSource

    print(f"Fetching prod DB URL from AWS SSM /{prod_env}/database_url ...")
    ssm = AwsSsmSettingsSource(SeerConfig, ssm_path_prefix=f"/{prod_env}/")
    params = ssm()
    url = params.get("database_url")
    if not url:
        raise ValueError(
            f"database_url not found in AWS SSM under /{prod_env}/. "
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
# Main
# ---------------------------------------------------------------------------


async def main(  # pylint: disable=too-complex,too-many-statements,too-many-locals,too-many-branches  # Reason: CLI entrypoint — URL resolve → connect → validate → clone loop → summary; splitting would hurt readability
    args: argparse.Namespace,
) -> int:
    # ── Resolve DB URLs ────────────────────────────────────────────────────
    try:
        prod_url: str = args.prod_db_url or resolve_prod_db_url(args.prod_env)
        local_url: str = args.local_db_url or resolve_local_db_url()
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Prod DB:  {prod_url[:60]}{'...' if len(prod_url) > 60 else ''}")
    print(f"Local DB: {local_url[:60]}{'...' if len(local_url) > 60 else ''}")

    prod_conn: asyncpg.Connection = await asyncpg.connect(prod_url)
    local_conn: asyncpg.Connection = await asyncpg.connect(local_url)
    await setup_conn(prod_conn)
    await setup_conn(local_conn)

    try:
        # ── Resolve users ──────────────────────────────────────────────────
        source_user = await resolve_user(prod_conn, args.source_user)
        source_user_id: int = source_user["id"]
        print(
            f"\nSource user: postgres_id={source_user_id}  "
            f"clerk_id={source_user['user_id']}  email={source_user['email']}"
        )

        local_user = await local_conn.fetchrow("SELECT id, email FROM users WHERE id = $1", args.local_user_id)
        if not local_user:
            print(
                f"ERROR: local user id={args.local_user_id} not found. "
                "Run the local seed script first.",
                file=sys.stderr,
            )
            return 1
        print(f"Local user: postgres_id={local_user['id']}  email={local_user['email']}")

        # ── Encryption warning ─────────────────────────────────────────────
        print(
            "\nWARNING: Encrypted fields (access_token_enc, refresh_token_enc, "
            "value_enc, session_state_enc) are copied verbatim.\n"
            "         They only work locally if LOCAL_ENCRYPTION_KEY matches production.\n"
        )

        # ── Optional pre-truncate ──────────────────────────────────────────
        if args.truncate_existing:
            if args.dry_run:
                print("[dry-run] Would truncate existing data for local user.\n")
            else:
                await truncate_user_data(local_conn, args.local_user_id)
                print()

        # ── Clone tables ───────────────────────────────────────────────────
        # id_map[table] = {old_pk: new_pk}
        # Pre-seed with the user mapping so child tables can remap user FKs.
        id_map: dict[str, dict] = {"users": {source_user_id: args.local_user_id}}

        print(f"Cloning {len(TABLE_CONFIGS)} tables:")
        total_rows = 0
        total_errors = 0

        for config in TABLE_CONFIGS:
            if not args.include_credentials and config.table in CREDENTIAL_TABLES:
                id_map[config.table] = {}  # empty map so FK children know to null out these FKs
                print(f"  {config.table}: skipped (use --include-credentials to clone)")
                continue

            rows = await fetch_rows(prod_conn, config, source_user_id, id_map)
            total_rows += len(rows)

            if args.dry_run:
                print(f"  {config.table}: {len(rows)} rows  [dry-run]")
                continue

            if not rows:
                id_map[config.table] = {}
                print(f"  {config.table}: 0 rows")
                continue

            old_ids: list[Any] = []
            new_ids: list[Any] = []

            try:
                async with local_conn.transaction():
                    for row in rows:
                        old_pk, data = remap_row(row, config, id_map, args.local_user_id)
                        new_pk = await insert_row(local_conn, config, data)
                        old_ids.append(old_pk if config.pk_type == "int" else str(old_pk))
                        new_ids.append(new_pk)

                id_map[config.table] = dict(zip(old_ids, new_ids))
                print(f"  {config.table}: {len(rows)} rows inserted")

            except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: catch any DB error so one bad table doesn't abort the whole clone
                print(f"  {config.table}: ERROR — {exc}")
                traceback.print_exc()
                id_map[config.table] = {}  # children will see empty map and skip
                total_errors += 1

        # ── Advance sequences after explicit-PK inserts ───────────────
        if not args.dry_run:
            print("\nAdvancing Postgres sequences ...")
            for config in TABLE_CONFIGS:
                if config.pk_type != "int":
                    continue
                try:
                    await local_conn.execute(
                        f"SELECT setval("
                        f"pg_get_serial_sequence('{config.table}', '{config.pk_col}'), "
                        f"COALESCE((SELECT MAX({config.pk_col}) FROM {config.table}), 1)"
                        f")"
                    )
                except Exception:  # pylint: disable=broad-exception-caught  # Reason: table may not exist in local env
                    pass

        # ── Clone LangGraph checkpoint tables ─────────────────────────────
        print("\nCollecting thread_ids for checkpoint cloning ...")
        thread_ids: list[str] = []
        _tid_conn = prod_conn if args.dry_run else local_conn
        _tid_user = source_user_id if args.dry_run else args.local_user_id
        for _table in ("workflow_runs", "workflow_chat_sessions", "workflow_discovery_chat_sessions"):
            try:
                _rows = await _tid_conn.fetch(
                    f"SELECT thread_id FROM {_table} WHERE user_id = $1 AND thread_id IS NOT NULL",
                    _tid_user,
                )
                thread_ids.extend(r["thread_id"] for r in _rows)
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: thread_id column may not exist in all envs
                pass
        thread_ids = list(set(thread_ids))
        print(f"  Found {len(thread_ids)} unique thread_id(s).")

        print("\nCloning checkpoint tables:")
        checkpoint_rows = await clone_checkpoint_tables(prod_conn, local_conn, thread_ids, args.dry_run)

        # ── Summary ────────────────────────────────────────────────────────
        if args.dry_run:
            print(f"\n[dry-run] Total rows that would be cloned: {total_rows} app rows + {checkpoint_rows} checkpoint rows")
        else:
            print(
                f"\nDone.  {total_rows} app rows + {checkpoint_rows} checkpoint rows cloned",
                end="",
            )
            print(f"  ({total_errors} table errors)" if total_errors else ".")
            print(f"\nLog in locally as user id={args.local_user_id} to browse the cloned data.")

        return 0 if total_errors == 0 else 1

    finally:
        await prod_conn.close()
        await local_conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clone a production user's data to local Postgres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-user",
        required=True,
        help="Clerk user ID (user_xxx) or postgres integer ID",
    )
    parser.add_argument(
        "--prod-env",
        default="main",
        help="AWS SSM environment prefix for prod secrets (default: main → /main/database_url)",
    )
    parser.add_argument(
        "--local-user-id",
        type=int,
        default=1,
        help="Local postgres user ID to remap all FKs to (default: 1)",
    )
    parser.add_argument(
        "--prod-db-url",
        default=None,
        help="Override prod DB URL directly (skips AWS SSM lookup)",
    )
    parser.add_argument(
        "--local-db-url",
        default=None,
        help="Override local DB URL directly (skips .env DATABASE_URL lookup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts per table without inserting anything",
    )
    parser.add_argument(
        "--truncate-existing",
        action="store_true",
        help="Delete all existing rows for local-user-id before cloning",
    )
    parser.add_argument(
        "--include-credentials",
        action="store_true",
        help=(
            "Also clone oauth_connections, integration_resources, and integration_secrets. "
            "Only useful if LOCAL_ENCRYPTION_KEY matches production — otherwise tokens are unreadable."
        ),
    )

    sys.exit(asyncio.run(main(parser.parse_args())))
