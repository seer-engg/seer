#!/usr/bin/env python3
"""
One-shot migration script that copies legacy workflow_records into the new
workflows/workflow_versions/workflow_drafts tables introduced in
4_20260106052819_workflow_version.

The script:
  1. Runs entirely inside a single database transaction.
  2. Reuses the original workflow IDs so existing foreign keys remain valid.
  3. Seeds a RELEASED workflow version per workflow and marks it as published.
  4. Initializes workflow drafts so the builder API keeps functioning.
  5. Backfills workflow_runs.workflow_version_id for existing runs.

Usage:
    python scripts/migrate_workflow_records.py

Make sure DATABASE_URL (and DB_SCHEMA if applicable) are set before running.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from tortoise import Tortoise
from tortoise.transactions import in_transaction

# Ensure project modules are importable when running as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seer.database import TORTOISE_ORM  # noqa: E402

logger = logging.getLogger("workflow_migration")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _hash_spec(spec: Dict[str, Any]) -> str:
    """Stable hash used to deduplicate versions."""
    serialized = json.dumps(spec or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _jsonb(value: Any) -> str | None:
    """Encode Python values for JSONB columns."""
    if value is None:
        return None
    return json.dumps(value)


def _normalize_json(
    raw_value: Any,
    *,
    default_factory: Callable[[], Any],
    context: str,
) -> Any:
    """
    Ensure legacy JSON is loaded as a native Python dict/list.

    Legacy workflow_records rows may store JSON as strings. This helper loads them
    while staying resilient to bad data.
    """
    if raw_value is None:
        return default_factory()
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return default_factory()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s JSON; using default.", context)
            return default_factory()
    # Unexpected shape (e.g., numbers/bools) – fall back but log once.
    logger.warning("Unexpected %s payload (%r); using default.", context, raw_value)
    return default_factory()


async def migrate_workflows() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    try:
        async with in_transaction("default") as connection:
            existing_workflows = await connection.execute_query_dict(
                "SELECT COUNT(*) AS count FROM workflows"
            )
            if existing_workflows and existing_workflows[0]["count"] > 0:
                logger.error(
                    "Workflows table already contains rows (%s). Aborting migration.",
                    existing_workflows[0]["count"],
                )
                return

            legacy_records = await connection.execute_query_dict(
                """
                SELECT
                    id,
                    user_id,
                    name,
                    description,
                    spec,
                    version,
                    tags,
                    meta,
                    last_compile_ok,
                    created_at,
                    updated_at
                FROM workflow_records
                ORDER BY id
                FOR UPDATE
                """
            )

            if not legacy_records:
                logger.info("No workflow_records rows found; nothing to migrate.")
                return

            inserted_count = 0
            run_updates = 0
            for record in legacy_records:
                workflow_id = record["id"]
                spec = _normalize_json(record["spec"], default_factory=dict, context="spec")
                spec_json = _jsonb(spec)
                revision = record["version"] or 1

                tags = _normalize_json(record["tags"], default_factory=list, context="tags")
                tags_json = _jsonb(tags)
                meta = _normalize_json(record.get("meta"), default_factory=dict, context="meta")
                meta["last_compile_ok"] = bool(record.get("last_compile_ok", False))
                meta_json = _jsonb(meta)

                await connection.execute_query(
                    """
                    INSERT INTO workflows (
                        id,
                        user_id,
                        name,
                        description,
                        tags,
                        meta,
                        created_at,
                        updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    [
                        workflow_id,
                        record["user_id"],
                        record["name"],
                        record.get("description"),
                        tags_json,
                        meta_json,
                        record["created_at"],
                        record["updated_at"],
                    ],
                )

                await connection.execute_query(
                    """
                    INSERT INTO workflow_drafts (
                        workflow_id,
                        spec,
                        revision,
                        updated_by_id
                    )
                    VALUES ($1, $2, $3, $4)
                    """,
                    [
                        workflow_id,
                        spec_json,
                        revision,
                        record["user_id"],
                    ],
                )

                version_result = await connection.execute_query_dict(
                    """
                    INSERT INTO workflow_versions (
                        workflow_id,
                        status,
                        spec,
                        created_from_draft_revision,
                        created_by_id,
                        manifest,
                        spec_hash,
                        version_number,
                        created_at
                    )
                    VALUES ($1, 'RELEASED', $2, $3, $4, NULL, $5, $6, $7)
                    RETURNING id
                    """,
                    [
                        workflow_id,
                        spec_json,
                        revision,
                        record["user_id"],
                        _hash_spec(spec),
                        revision,
                        record["updated_at"],
                    ],
                )
                version_id = version_result[0]["id"]

                await connection.execute_query(
                    """
                    UPDATE workflows
                    SET published_version_id = $1
                    WHERE id = $2
                    """,
                    [version_id, workflow_id],
                )

                updated_runs = await connection.execute_query_dict(
                    """
                    UPDATE workflow_runs
                    SET workflow_version_id = $1
                    WHERE workflow_id = $2 AND (workflow_version_id IS NULL)
                    RETURNING id
                    """,
                    [version_id, workflow_id],
                )
                run_updates += len(updated_runs)

                inserted_count += 1

            await connection.execute_query(
                "SELECT setval(pg_get_serial_sequence('workflows', 'id'), (SELECT COALESCE(MAX(id), 0) FROM workflows))"
            )
            await connection.execute_query(
                "SELECT setval(pg_get_serial_sequence('workflow_versions', 'id'), (SELECT COALESCE(MAX(id), 0) FROM workflow_versions))"
            )
            await connection.execute_query(
                "SELECT setval(pg_get_serial_sequence('workflow_drafts', 'id'), (SELECT COALESCE(MAX(id), 0) FROM workflow_drafts))"
            )

            logger.info(
                "Successfully migrated %s workflows. Updated %s workflow_runs.",
                inserted_count,
                run_updates,
            )
    finally:
        await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(migrate_workflows())
