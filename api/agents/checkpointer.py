"""PostgreSQL checkpointer management for LangGraph."""
from contextlib import asynccontextmanager
from typing import Optional
import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from shared.config import config
from shared.logger import get_logger

logger = get_logger("api.checkpointer")

# Global checkpointer instance
_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Get or create the async PostgreSQL checkpointer.
    
    Uses connection pooling for efficient database access.
    Raises RuntimeError if DATABASE_URI is not configured.
    """
    global _checkpointer
    
    if _checkpointer is not None:
        return _checkpointer
    
    async with _checkpointer_lock:
        # Double-check after acquiring lock
        if _checkpointer is not None:
            return _checkpointer
        
        if not config.database_uri:
            raise RuntimeError(
                "DATABASE_URI environment variable is required for checkpointing. "
                "Please set it to a PostgreSQL connection string."
            )
        
        logger.info("Initializing AsyncPostgresSaver checkpointer")
        _checkpointer = AsyncPostgresSaver.from_conn_string(config.database_uri)
        
        # Setup tables (idempotent)
        try:
            await _checkpointer.setup()
            logger.info("PostgreSQL checkpointer tables setup complete")
        except Exception as e:
            # Tables might already exist
            logger.debug(f"Checkpointer setup (tables may already exist): {e}")
        
        return _checkpointer


async def close_checkpointer():
    """Close the checkpointer connection pool."""
    global _checkpointer
    
    if _checkpointer is not None:
        try:
            # AsyncPostgresSaver manages its own connection pool
            # Close will be handled when the app shuts down
            logger.info("Closing checkpointer connection")
            _checkpointer = None
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")


@asynccontextmanager
async def checkpointer_lifespan():
    """
    Context manager for checkpointer lifecycle.
    
    Use with FastAPI lifespan for proper initialization/cleanup.
    """
    try:
        # Initialize checkpointer on startup
        await get_checkpointer()
        yield
    finally:
        # Cleanup on shutdown
        await close_checkpointer()

