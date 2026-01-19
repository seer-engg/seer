from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM
from seer.database.models import (
    User,
    UserPublic,
)
from seer.database.models_oauth import OAuthConnection
from seer.database.models_integrations import IntegrationResource, IntegrationSecret
from seer.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
    Workflow,
    WorkflowChatMessage,
    WorkflowChatSession,
    WorkflowDraft,
    WorkflowProposal,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
    make_run_public_id,
    make_workflow_public_id,
    parse_run_public_id,
    parse_workflow_public_id,
)
from seer.database.subscription_models import (
    BillingProfile,
    BillingProfileType,
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
    StripeWebhookEvent,
    StripeWebhookEventStatus,
)
from seer.logger import get_logger

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
    # Config
    "TORTOISE_ORM",
    # Core models
    "User",
    "UserPublic",
    # OAuth & Integrations
    "OAuthConnection",
    "IntegrationResource",
    "IntegrationSecret",
    # Workflow domain
    "Workflow",
    "WorkflowDraft",
    "WorkflowVersion",
    "WorkflowRun",
    "WorkflowChatSession",
    "WorkflowChatMessage",
    "WorkflowProposal",
    "WorkflowVersionStatus",
    "WorkflowRunStatus",
    "WorkflowRunSource",
    "TriggerSubscription",
    "TriggerEvent",
    "TriggerEventStatus",
    "make_workflow_public_id",
    "parse_workflow_public_id",
    "make_run_public_id",
    "parse_run_public_id",
    # Subscription domain
    "BillingProfile",
    "BillingProfileType",
    "BillingSubscription",
    "SubscriptionTier",
    "SubscriptionStatus",
    "StripeWebhookEvent",
    "StripeWebhookEventStatus",
]
