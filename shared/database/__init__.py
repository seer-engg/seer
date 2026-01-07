"""Database initialization with SQLModel."""
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
import subprocess

from shared.logger import get_logger
from shared.config import config

# Import all models to ensure they're registered
from shared.database import models  # noqa

logger = get_logger("shared.database")


async def run_alembic_migrations() -> None:
    """Run Alembic migrations."""
    if not config.AUTO_APPLY_DATABASE_MIGRATIONS:
        logger.info("⏭️  Auto-migrations disabled. Skipping Alembic.")
        return

    try:
        logger.info("🔄 Running Alembic migrations...")
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("✅ Alembic migrations completed successfully")
        if result.stdout:
            logger.debug(f"Alembic output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Alembic migration failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"❌ Alembic error: {e}")
        raise


@asynccontextmanager
async def db_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler for database."""
    logger.info("🚀 Initializing database...")

    # Run migrations or create tables
    if config.AUTO_APPLY_DATABASE_MIGRATIONS:
        await run_alembic_migrations()
    else:
        logger.info("📋 Auto-migrations disabled. Creating tables with SQLModel...")
        await init_db()
        logger.info("✅ Tables created successfully")

    logger.info("✅ Database ready")

    try:
        yield
    finally:
        from shared.database.base import close_db
        logger.info("🔌 Closing database connections...")
        await close_db()
        logger.info("✅ Database closed")


# Re-export database functions for backward compatibility
from shared.database.base import init_db, close_db  # noqa: E402, F401

# Re-export models for backward compatibility
from shared.database.models import (
    User,
    UserPublic,
    Project,
    Workflow,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowRecord,
    WorkflowRun,
    WorkflowChatSession,
    WorkflowChatMessage,
    WorkflowProposal,
    TriggerSubscription,
    TriggerEvent,
    OAuthConnection,
    IntegrationResource,
    IntegrationSecret,
    WorkflowRunStatus,
    WorkflowRunSource,
    WorkflowVersionStatus,
    TriggerEventStatus,
    make_workflow_public_id,
    parse_workflow_public_id,
    make_run_public_id,
    parse_run_public_id,
)

__all__ = [
    "db_lifespan",
    "init_db",
    "close_db",
    "User",
    "UserPublic",
    "Project",
    "Workflow",
    "WorkflowDraft",
    "WorkflowVersion",
    "WorkflowRecord",
    "WorkflowRun",
    "WorkflowChatSession",
    "WorkflowChatMessage",
    "WorkflowProposal",
    "TriggerSubscription",
    "TriggerEvent",
    "OAuthConnection",
    "IntegrationResource",
    "IntegrationSecret",
    "WorkflowRunStatus",
    "WorkflowRunSource",
    "WorkflowVersionStatus",
    "TriggerEventStatus",
    "make_workflow_public_id",
    "parse_workflow_public_id",
    "make_run_public_id",
    "parse_run_public_id",
]
