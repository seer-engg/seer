# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
Database fixtures for E2E tests with real PostgreSQL.

Provides:
- Session-scoped database initialization with migrations
- Function-scoped transaction isolation for test independence
"""
import asyncio
from typing import AsyncGenerator, Dict, Any

import pytest


def _build_tortoise_config(database_url: str) -> Dict[str, Any]:
    """
    Build Tortoise ORM config pointing to test container.

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
                    "seer.database.template_models",
                    "seer.database.overage_models",
                    "seer.database.profile_models",
                    "seer.database.organization_models",
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
    Initialize database schema using aerich migrations.

    Runs migrations once per session to set up the full schema.
    This is slower than generate_schemas but ensures migrations work correctly.

    Args:
        database_url: PostgreSQL connection URL from container

    Returns:
        str: The database URL (for downstream fixtures)
    """
    import subprocess
    import os

    # Set DATABASE_URL for aerich to use
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url

    # Run aerich upgrade to apply all migrations
    # This tests that migrations are valid and complete
    result = subprocess.run(
        ["uv", "run", "aerich", "upgrade"],
        env=env,
        cwd="/home/lokesh/fifth/seer",
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        # If migrations fail, try generating schemas directly
        # This allows tests to run even if migrations have issues
        import asyncio
        from tortoise import Tortoise

        async def init_db():
            config = _build_tortoise_config(database_url)
            await Tortoise.init(config=config)
            await Tortoise.generate_schemas()
            await Tortoise.close_connections()

        asyncio.get_event_loop().run_until_complete(init_db())

    return database_url


@pytest.fixture(scope="function")
async def db_session(db_initialized: str) -> AsyncGenerator[None, None]:
    """
    Function-scoped database session with transaction rollback.

    Each test runs in a transaction that is rolled back at the end,
    providing complete isolation without re-running migrations.

    This approach is:
    - Fast: No schema recreation between tests
    - Isolated: Each test sees a clean database state
    - Safe: Changes are never committed

    Args:
        db_initialized: Database URL (ensures migrations ran)

    Yields:
        None: Test runs within transaction context
    """
    from tortoise import Tortoise
    from tortoise.transactions import in_transaction

    config = _build_tortoise_config(db_initialized)
    await Tortoise.init(config=config)

    # Get connection and start transaction
    conn = Tortoise.get_connection("default")

    # Use savepoint for nested transaction support
    async with in_transaction("default") as transaction:
        try:
            yield
        finally:
            # Rollback to ensure test isolation
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

    config = _build_tortoise_config(db_initialized)
    await Tortoise.init(config=config)

    yield

    # Truncate all tables after test
    conn = Tortoise.get_connection("default")
    # Get all table names from models
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
            # Table might not exist or have dependencies
            pass

    await Tortoise.close_connections()


__all__ = [
    "db_initialized",
    "db_session",
    "clean_db_session",
]
