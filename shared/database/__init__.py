from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from shared.database.config import TORTOISE_ORM
from shared.database.models_integrations import IntegrationResource, IntegrationSecret
from shared.database.workflow_models import (
    Workflow,
    WorkflowChatMessage,
    WorkflowChatSession,
    WorkflowDraft,
    WorkflowProposal,
    WorkflowRun,
    WorkflowVersion,
)
from shared.logger import get_logger

logger = get_logger("shared.database")


async def init_db() -> None:
    """Initialize Tortoise ORM with the configured settings."""

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
    "Workflow",
    "WorkflowDraft",
    "WorkflowVersion",
    "WorkflowRun",
    "WorkflowChatSession",
    "WorkflowChatMessage",
    "WorkflowProposal",
    "IntegrationResource",
    "IntegrationSecret",
]
