from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from shared.logger import get_logger
from shared.database.config import TORTOISE_ORM
from shared.database.workflow_models import (
    Workflow,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowRun,
    WorkflowChatSession,
    WorkflowChatMessage,
    WorkflowProposal,
    parse_workflow_public_id,
    WorkflowRunStatus,
    WorkflowRunSource,
    TriggerEventStatus,
    make_workflow_public_id,
    WorkflowVersionStatus,
    parse_run_public_id,
    TriggerSubscription
)
from shared.database.models_integrations import IntegrationResource, IntegrationSecret
from shared.database.models import User

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
    "parse_workflow_public_id",
    "Workflow",
    "WorkflowDraft",
    "WorkflowVersion",
    "WorkflowRun",
    "WorkflowChatSession",
    "WorkflowChatMessage",
    "WorkflowProposal",
    "IntegrationResource",
    "IntegrationSecret",
    "User",
    "WorkflowRunStatus",
    "WorkflowRunSource",
    "TriggerEventStatus",
    "make_workflow_public_id",
    "WorkflowVersionStatus",
    "parse_run_public_id",
    "TriggerSubscription",
]
