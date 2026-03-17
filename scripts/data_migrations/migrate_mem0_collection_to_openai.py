#!/usr/bin/env python3
"""
Migrate Mem0 memories from a legacy collection into the OpenAI-backed collection.

This script reads raw payloads from the source pgvector table, regenerates
embeddings with the currently configured OpenAI Mem0 embedder, and inserts the
rows into the target collection while preserving ids, timestamps, scope fields,
and custom metadata.

Usage:
    uv run python scripts/data_migrations/migrate_mem0_collection_to_openai.py

    uv run python scripts/data_migrations/migrate_mem0_collection_to_openai.py \
      --source-collection nexus_user_memories \
      --target-collection nexus_user_memories_openai \
      --batch-size 100

    uv run python scripts/data_migrations/migrate_mem0_collection_to_openai.py --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import uuid

import psycopg2
from psycopg2.extras import Json, execute_values

# Add src/ to path so we can import seer modules when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mem0 import Memory

from seer.config import config
from seer.services.memory.mem0_client import _build_mem0_config, _parse_database_url

TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class MigrationStats:
    """Track migration outcomes."""

    scanned: int = 0
    prepared: int = 0
    inserted: int = 0
    skipped_invalid: int = 0
    skipped_existing: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-collection",
        default="nexus_user_memories",
        help="Existing pgvector table containing 384-dimension memories.",
    )
    parser.add_argument(
        "--target-collection",
        default=config.mem0_collection_name,
        help="OpenAI-backed pgvector table to write migrated memories into.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to embed and insert per batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many source rows to migrate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect source/target collections without creating embeddings or writing data.",
    )
    return parser.parse_args()


def _validate_table_name(name: str) -> str:
    if not TABLE_NAME_RE.match(name):
        raise ValueError(f"Invalid collection name: {name!r}")
    return name


def _get_db_connection():
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to run the memory migration")

    db_params = _parse_database_url(config.database_url)
    return psycopg2.connect(**db_params)


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def _row_count(conn, table_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return int(cur.fetchone()[0])


def _vector_type(conn, table_name: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s
              AND a.attname = 'vector'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (table_name,),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None


def _build_target_client(target_collection: str) -> Memory:
    mem0_config = _build_mem0_config()
    mem0_config["vector_store"]["config"]["collection_name"] = target_collection
    return Memory.from_config(mem0_config)


def _iter_source_rows(conn, source_collection: str, batch_size: int):
    with conn.cursor(name="mem0_migrate_source") as cur:
        cur.itersize = batch_size
        cur.execute(
            f"""
            SELECT id::text, payload
            FROM {source_collection}
            ORDER BY COALESCE(payload->>'created_at', ''), id
            """
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield rows


def _insert_batch(target_client: Memory, target_collection: str, batch_rows: list[tuple[str, list[float], dict]]) -> int:
    prepared_rows = [
        (str(uuid.UUID(memory_id)), embedding, Json(payload))
        for memory_id, embedding, payload in batch_rows
    ]

    inserted = execute_values(
        target_client.vector_store.cur,
        f"""
        INSERT INTO {target_collection} (id, vector, payload)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
        RETURNING id
        """,
        prepared_rows,
        template="(%s::uuid, %s, %s)",
        fetch=True,
    )
    target_client.vector_store.conn.commit()
    return len(inserted)


def _migrate(args: argparse.Namespace) -> None:
    source_collection = _validate_table_name(args.source_collection)
    target_collection = _validate_table_name(args.target_collection)

    if source_collection == target_collection:
        raise ValueError("Source and target collections must be different")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if config.mem0_embedder_provider.lower() != "openai":
        raise RuntimeError(
            "This migration expects MEM0_EMBEDDER_PROVIDER=openai for the target collection"
        )

    source_conn = _get_db_connection()
    try:
        if not _table_exists(source_conn, source_collection):
            raise RuntimeError(f"Source collection {source_collection!r} does not exist")

        print(f"Source collection: {source_collection}")
        print(f"Source vector type: {_vector_type(source_conn, source_collection)}")
        print(f"Source rows: {_row_count(source_conn, source_collection)}")
        print(f"Target collection: {target_collection}")
        print(f"Target embedder: {config.mem0_embedder_provider}/{config.mem0_embedder_model}")
        print(f"Target dimensions: {config.mem0_embedding_dims}")

        if args.dry_run:
            if _table_exists(source_conn, target_collection):
                print(f"Target vector type: {_vector_type(source_conn, target_collection)}")
                print(f"Target rows: {_row_count(source_conn, target_collection)}")
            else:
                print("Target collection does not exist yet.")
            print("Dry run complete.")
            return

        target_client = _build_target_client(target_collection)
        stats = MigrationStats()
        limit_remaining = args.limit

        print(f"Target vector type: {_vector_type(source_conn, target_collection)}")

        for source_rows in _iter_source_rows(source_conn, source_collection, args.batch_size):
            batch_payloads: list[tuple[str, list[float], dict]] = []

            for memory_id, payload in source_rows:
                if limit_remaining is not None and limit_remaining <= 0:
                    break

                stats.scanned += 1
                if limit_remaining is not None:
                    limit_remaining -= 1

                if not isinstance(payload, dict):
                    stats.skipped_invalid += 1
                    continue

                memory_text = payload.get("data")
                if not isinstance(memory_text, str) or not memory_text.strip():
                    stats.skipped_invalid += 1
                    continue

                try:
                    uuid.UUID(memory_id)
                except ValueError:
                    stats.skipped_invalid += 1
                    continue

                embedding = target_client.embedding_model.embed(memory_text, memory_action="add")
                batch_payloads.append((memory_id, embedding, payload))
                stats.prepared += 1

            if batch_payloads:
                inserted = _insert_batch(target_client, target_collection, batch_payloads)
                stats.inserted += inserted
                stats.skipped_existing += len(batch_payloads) - inserted
                print(
                    f"Migrated batch: prepared={len(batch_payloads)} inserted={inserted} "
                    f"skipped_existing={len(batch_payloads) - inserted}"
                )

            if limit_remaining is not None and limit_remaining <= 0:
                break

        print("Migration complete.")
        print(f"  scanned:          {stats.scanned}")
        print(f"  prepared:         {stats.prepared}")
        print(f"  inserted:         {stats.inserted}")
        print(f"  skipped_existing: {stats.skipped_existing}")
        print(f"  skipped_invalid:  {stats.skipped_invalid}")

    finally:
        source_conn.close()


if __name__ == "__main__":
    _migrate(_parse_args())
