from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from shared.logger import get_logger
from shared.database.config import TORTOISE_ORM
from shared.database.workflow_models import (
    WorkflowRecord,
    WorkflowRun,
    WorkflowChatSession,
    WorkflowChatMessage,
    WorkflowProposal,
)
from shared.config import config

logger = get_logger("shared.database")


async def run_migrations() -> None:
    """Run Aerich migrations to update database schema.
    
    This function runs automatically on startup to ensure existing users
    upgrading from older versions get their databases migrated seamlessly.
    """
    try:
        from aerich import Command
        
        logger.info("Running database migrations...")
        # Command handles Tortoise initialization internally
        async with Command(tortoise_config=TORTOISE_ORM, app='models') as command:
            await command.upgrade()
        logger.info("✅ Database migrations applied successfully")
    except ImportError:
        logger.warning(
            "⚠️ Aerich not available. Migrations skipped. "
            "Install aerich to enable automatic migrations: pip install aerich"
        )
    except Exception as e:
        logger.error(
            f"❌ Migration failed: {e}. "
            "Database schema may be out of sync. Please fix migrations before starting the application.",
            exc_info=True,
        )
        raise  # Fail fast - migrations are critical


async def init_db() -> None:
    """Initialize Tortoise ORM with the configured settings."""
    
    # Run migrations first (Command handles Tortoise initialization)
    if config.AUTO_APPLY_DATABASE_MIGRATIONS:
        logger.info("Auto-applying database migrations...")
        await run_migrations()
    else:
        logger.info("Database migrations will not be applied automatically. Please run migrations manually.")
    
    # Initialize Tortoise for the application (Command closes connections on exit)
    await Tortoise.init(config=TORTOISE_ORM)


async def close_db() -> None:
    """Close all ORM connections."""
    await Tortoise.close_connections()


@asynccontextmanager
async def db_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler for database management."""
    logger.info("Initializing database connections")
    await init_db()
    try:
        yield
    finally:
        logger.info("Closing database connections")
        await close_db()


__all__ = [
    "db_lifespan",
    "init_db",
    "close_db",
    "WorkflowRecord",
    "WorkflowRun",
    "WorkflowChatSession", "WorkflowChatMessage", "WorkflowProposal"
]


