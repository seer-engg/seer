# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
Shared database fixtures for tests using real PostgreSQL via Testcontainers.

Provides:
- Session-scoped database initialization with schema generation
- Function-scoped transaction isolation for test independence
- Shared Tortoise ORM config builder

Used by both integration tests (via db_engine) and E2E tests (via db_session).
"""
from typing import Any, AsyncGenerator, Dict

import pytest


def build_tortoise_config(database_url: str) -> Dict[str, Any]:
    """
    Build Tortoise ORM config pointing to a PostgreSQL instance.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        Tortoise ORM configuration dictionary
    """
    from urllib.parse import urlparse

    parsed = urlparse(database_url)
    credentials = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "database": (parsed.path or "").lstrip("/") or "seer_test",
        "minsize": 1,
        "maxsize": 5,
    }

    return {
        "connections": {
            "default": {
                "engine": "tortoise.backends.asyncpg",
                "credentials": credentials,
            },
        },
        "apps": {
            "models": {
                "models": [
                    "seer.database.models",
                    "seer.database.models_oauth",
                    "seer.database.models_integrations",
                    "seer.database.models_browser",
                    "seer.database.models_browser_recording",
                    "seer.database.workflow_models",
                    "seer.database.chat_models",
                    "seer.database.subscription_models",
                    "seer.database.usage_models",
                    "seer.database.knowledge_models",
                    "seer.database.memory_models",
                    "seer.database.template_models",
                    "seer.database.overage_models",
                    "seer.database.profile_models",
                    "seer.database.organization_models",
                    "seer.database.email_models",
                    "seer.database.learning_models",
                    "aerich.models",
                ],
                "default_connection": "default",
            },
        },
        "use_tz": True,
        "timezone": "UTC",
    }


@pytest.fixture(scope="session")
def db_initialized(database_url: str) -> str:
    """
    Initialize database schema once per session.

    Tries aerich migrations first (validates migrations are correct),
    falls back to generate_schemas if migrations fail.

    Args:
        database_url: PostgreSQL connection URL from container

    Returns:
        str: The database URL (for downstream fixtures)
    """
    import subprocess
    import os
    from pathlib import Path

    # Set DATABASE_URL for aerich to use
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url

    # Get project root
    project_root = Path(__file__).parents[2]

    # Run aerich upgrade to apply all migrations
    result = subprocess.run(
        ["uv", "run", "aerich", "upgrade"],
        env=env,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        # Fallback: generate schemas directly if migrations fail
        import asyncio
        from tortoise import Tortoise

        async def init_db():
            config = build_tortoise_config(database_url)
            await Tortoise.init(config=config)
            await Tortoise.generate_schemas()
            await Tortoise.close_connections()

        asyncio.get_event_loop().run_until_complete(init_db())

    return database_url


@pytest.fixture(scope="function")
async def db_engine(db_initialized: str) -> AsyncGenerator[None, None]:
    """
    Function-scoped database session with truncation-based cleanup.

    Replaces the old SQLite in-memory fixture. Uses real PostgreSQL with
    table truncation after each test for isolation.

    Why truncation instead of transaction rollback:
      PostgreSQL aborts the entire transaction on IntegrityError. Business
      logic like persist_event() catches IntegrityError and continues
      querying — this works in production (auto-commit mode) but fails
      inside a test transaction wrapper. Truncation avoids this.

    Args:
        db_initialized: Database URL (ensures migrations ran)

    Yields:
        None: Test runs with full PostgreSQL semantics
    """
    from tortoise import Tortoise
    from tortoise import connections as tortoise_connections

    # Reset Tortoise internal state. We cannot call close_connections()
    # because the previous test's event loop may already be gone, causing
    # asyncpg RuntimeError. Instead, we force-clear internal state and
    # let the old pool be garbage-collected.
    try:
        # Terminate pools immediately (no graceful close that needs event loop)
        for name in list(tortoise_connections._get_storage().keys()):
            conn = tortoise_connections.get(name)
            pool = getattr(conn, "_pool", None)
            if pool is not None:
                try:
                    pool.terminate()
                except Exception:
                    pass
        tortoise_connections._get_storage().clear()
    except Exception:
        pass
    Tortoise.apps = {}
    Tortoise._inited = False

    config = build_tortoise_config(db_initialized)
    await Tortoise.init(config=config)

    yield

    # Truncate all application tables for isolation.
    # Uses pg_tables to discover all tables in the public schema,
    # excluding system tables (aerich, pg_*).
    try:
        conn = Tortoise.get_connection("default")
        _, rows = await conn.execute_query(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename != 'aerich'"
        )
        tables = [r["tablename"] for r in rows]
        if tables:
            table_list = ", ".join(f'"{t}"' for t in tables)
            await conn.execute_query(f"TRUNCATE {table_list} CASCADE;")
    except Exception:
        pass

    try:
        await Tortoise.close_connections()
    except Exception:
        pass


@pytest.fixture(scope="function")
async def db_session(db_initialized: str) -> AsyncGenerator[None, None]:
    """
    Alias for db_engine — used by E2E tests.

    Provides the same transaction-rollback isolation as db_engine.
    Kept as a separate name for E2E test compatibility.
    """
    from tortoise import Tortoise
    from tortoise.transactions import in_transaction

    config = build_tortoise_config(db_initialized)
    await Tortoise.init(config=config)

    async with in_transaction("default") as transaction:
        try:
            yield
        finally:
            await transaction.rollback()

    await Tortoise.close_connections()


@pytest.fixture(scope="function")
async def clean_db_session(db_initialized: str) -> AsyncGenerator[None, None]:
    """
    Alternative fixture that truncates tables instead of using transactions.

    Use this for tests that need to commit transactions (e.g., testing
    transaction behavior) but still want a clean state.

    Args:
        db_initialized: Database URL (ensures migrations ran)

    Yields:
        None: Test runs with truncated tables
    """
    from tortoise import Tortoise

    config = build_tortoise_config(db_initialized)
    await Tortoise.init(config=config)

    yield

    # Truncate all tables after test
    conn = Tortoise.get_connection("default")
    tables = [
        "workflow_runs", "trigger_subscriptions", "trigger_events",
        "workflows", "users", "organizations", "billing_subscriptions",
        "usage_records", "knowledge_documents", "chat_sessions",
        "oauth_connections", "api_key_connections",
    ]

    for table in tables:
        try:
            await conn.execute_query(f"TRUNCATE TABLE {table} CASCADE;")
        except Exception:
            pass

    await Tortoise.close_connections()


__all__ = [
    "build_tortoise_config",
    "db_initialized",
    "db_engine",
    "db_session",
    "clean_db_session",
]
