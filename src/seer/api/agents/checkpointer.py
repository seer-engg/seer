"""PostgreSQL checkpointer management for LangGraph."""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.checkpointer")


@asynccontextmanager
async def open_checkpointer(
    dsn: str,
    *,
    max_size: int = 10,
    pipeline: bool = False,  # pylint: disable=unused-argument  # Reason: Reserved for future psycopg feature
    prepare_threshold: int = 0,
):
    """
    Open a pooled AsyncPostgresSaver checkpointer for the caller's lifespan.

    The returned saver keeps its underlying psycopg pool open until the context
    exits, ensuring LangGraph graphs can call methods like ainvoke/aget_state_history
    safely across the entire runtime of the workflow.
    """
    pool = AsyncConnectionPool(
        conninfo=dsn,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": prepare_threshold,
        },
    )

    await pool.open()
    saver = AsyncPostgresSaver(pool)

    # Safe to call more than once; ensures checkpoint tables exist.
    await saver.setup()

    try:
        yield saver
    finally:
        await pool.close()


# Global checkpointer instance and context manager
_checkpointer: Optional[AsyncPostgresSaver] = None  # pylint: disable=invalid-name  # Reason: Module-level singleton cache
_checkpointer_cm: Optional[AsyncContextManager[AsyncPostgresSaver]] = None  # pylint: disable=invalid-name  # Reason: Module-level singleton cache
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer() -> Optional[AsyncPostgresSaver]:
    """
    Get or create the async PostgreSQL checkpointer.

    Uses connection pooling for efficient database access.
    Returns None if DATABASE_URL is not configured or initialization fails.
    """
    global _checkpointer, _checkpointer_cm  # pylint: disable=global-statement  # Reason: Singleton caching pattern

    if _checkpointer is not None:
        return _checkpointer

    async with _checkpointer_lock:
        # Double-check after acquiring lock
        if _checkpointer is not None:
            return _checkpointer

        if not config.DATABASE_URL:
            logger.warning("DATABASE_URL not configured, workflows will run without checkpointing")
            return None

        logger.info("Initializing AsyncPostgresSaver checkpointer")
        try:
            global _checkpointer_cm  # pylint: disable=global-statement  # Reason: Singleton caching pattern
            _checkpointer_cm = open_checkpointer(config.DATABASE_URL)

            _checkpointer = await _checkpointer_cm.__aenter__()  # pylint: disable=no-member,unnecessary-dunder-call  # Reason: Async context manager protocol

            # Verify checkpointer has required methods before returning
            if not hasattr(_checkpointer, 'get_next_version'):
                logger.warning("Checkpointer missing get_next_version method - this may cause issues")
                # Don't fail here - let graph_builder handle it gracefully

            return _checkpointer
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Initialization must handle any error gracefully
            logger.error(f"Failed to initialize checkpointer: {e}", exc_info=True)  # pylint: disable=logging-fstring-interpolation  # Reason: Error context needed
            _checkpointer = None
            _checkpointer_cm = None
            return None


async def close_checkpointer():
    """Close the checkpointer connection pool."""
    global _checkpointer, _checkpointer_cm  # pylint: disable=global-statement  # Reason: Singleton caching pattern

    if _checkpointer_cm is not None:
        try:
            # Exit the context manager properly
            logger.info("Closing checkpointer connection")
            await _checkpointer_cm.__aexit__(None, None, None)
            _checkpointer = None
            _checkpointer_cm = None
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Cleanup must handle any error
            logger.warning(f"Error closing checkpointer: {e}")  # pylint: disable=logging-fstring-interpolation  # Reason: Error context needed
            _checkpointer = None
            _checkpointer_cm = None


async def _is_checkpointer_healthy(checkpointer: AsyncPostgresSaver) -> bool:
    """Check if checkpointer connection is healthy."""
    try:
        # Try a simple operation to test connection
        # Use a minimal config that won't fail on empty state
        test_config = {"configurable": {"thread_id": "__health_check__"}}
        # Add timeout to prevent hanging
        await asyncio.wait_for(
            checkpointer.aget_tuple(test_config),
            timeout=5.0  # 5 second timeout
        )
        return True
    except asyncio.TimeoutError:
        logger.warning("Checkpointer health check timed out")
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Health check must handle any error
        logger.debug(f"Checkpointer health check failed: {e}")  # pylint: disable=logging-fstring-interpolation  # Reason: Error context needed
        return False


async def _recreate_checkpointer() -> Optional[AsyncPostgresSaver]:
    """Recreate checkpointer if connection is stale."""
    global _checkpointer, _checkpointer_cm  # pylint: disable=global-statement  # Reason: Singleton caching pattern

    # Close existing checkpointer
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)  # pylint: disable=no-member  # Reason: Async context manager protocol
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Cleanup must handle any error
            logger.warning(f"Error closing stale checkpointer: {e}")  # pylint: disable=logging-fstring-interpolation  # Reason: Error context needed

    _checkpointer = None
    _checkpointer_cm = None

    # Recreate
    return await get_checkpointer()


async def get_checkpointer_with_retry() -> Optional[AsyncPostgresSaver]:
    """Get checkpointer with automatic reconnection on failure."""
    checkpointer = await get_checkpointer()

    if checkpointer is None:
        return None

    # Check health
    if not await _is_checkpointer_healthy(checkpointer):
        logger.warning("Checkpointer connection is stale, recreating...")
        checkpointer = await _recreate_checkpointer()

    return checkpointer


@asynccontextmanager
async def checkpointer_lifespan():
    """
    Context manager for checkpointer lifecycle.

    Use with FastAPI lifespan for proper initialization/cleanup.
    """
    checkpointer = None
    try:
        # Initialize checkpointer on startup
        checkpointer = await get_checkpointer()
        yield checkpointer
    finally:
        # Cleanup on shutdown
        await close_checkpointer()
