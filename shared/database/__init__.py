from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from tortoise import Tortoise

from shared.logger import get_logger
from shared.database.config import DB_GENERATE_SCHEMAS, TORTOISE_ORM

logger = get_logger("shared.database")


async def init_db() -> None:
    """Initialize Tortoise ORM with the configured settings."""
    # Ensure schema exists if using a custom schema (not 'public')
    # This must happen BEFORE Tortoise.init() to avoid schema errors
    schema = os.getenv("DB_SCHEMA", "public")
    if schema != "public":
        from shared.database.config import DB_CREDENTIALS
        import asyncpg
        try:
            # Create schema if it doesn't exist
            conn = await asyncpg.connect(
                host=DB_CREDENTIALS["host"],
                port=DB_CREDENTIALS["port"],
                user=DB_CREDENTIALS["user"],
                password=DB_CREDENTIALS["password"],
                database=DB_CREDENTIALS["database"],
            )
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            await conn.close()
            logger.info(f"Ensured schema '{schema}' exists")
        except Exception as e:
            logger.warning(f"Could not create schema '{schema}': {e}. Using 'public' schema.")
            # Note: If schema creation fails, server_settings will still try to use the schema
            # This may cause errors, but it's better than silently failing
    
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


