from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from src.core.config import logger
from src.db.config import DB_GENERATE_SCHEMAS, TORTOISE_ORM


async def init_db() -> None:
    """Initialize Tortoise ORM with the configured settings."""
    await Tortoise.init(config=TORTOISE_ORM)
    if DB_GENERATE_SCHEMAS:
        logger.warning(
            "DB_GENERATE_SCHEMAS is enabled – generating schemas at startup. "
            "Disable in production and rely on migrations instead.",
        )
        await Tortoise.generate_schemas()


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


__all__ = ["db_lifespan", "init_db", "close_db"]


