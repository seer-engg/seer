"""PostgreSQL checkpointer management for LangGraph."""
from contextlib import asynccontextmanager
from typing import Optional, Any
import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from shared.config import config
from shared.logger import get_logger

logger = get_logger("api.checkpointer")

# Global checkpointer instance and context manager
_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_cm: Optional[Any] = None  # Store context manager to keep it alive
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer() -> Optional[AsyncPostgresSaver]:
    """
    Get or create the async PostgreSQL checkpointer.
    
    Uses connection pooling for efficient database access.
    Returns None if DATABASE_URL is not configured or initialization fails.
    """
    global _checkpointer, _checkpointer_cm
    
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
            # AsyncPostgresSaver.from_conn_string() returns a context manager
            # We need to enter it to get the actual checkpointer instance
            global _checkpointer_cm
            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(config.DATABASE_URL)
            
            # Enter the context manager to get the actual checkpointer instance
            _checkpointer = await _checkpointer_cm.__aenter__()
            
            # Setup tables (idempotent)
            try:
                await _checkpointer.setup()
                logger.info("PostgreSQL checkpointer tables setup complete")
            except Exception as e:
                # Tables might already exist
                logger.debug(f"Checkpointer setup (tables may already exist): {e}")
            
            # Verify checkpointer has required methods before returning
            if not hasattr(_checkpointer, 'get_next_version'):
                logger.warning("Checkpointer missing get_next_version method - this may cause issues")
                # Don't fail here - let graph_builder handle it gracefully
            
            return _checkpointer
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}", exc_info=True)
            _checkpointer = None
            _checkpointer_cm = None
            return None


async def close_checkpointer():
    """Close the checkpointer connection pool."""
    global _checkpointer, _checkpointer_cm
    
    if _checkpointer_cm is not None:
        try:
            # Exit the context manager properly
            logger.info("Closing checkpointer connection")
            await _checkpointer_cm.__aexit__(None, None, None)
            _checkpointer = None
            _checkpointer_cm = None
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")
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
    except Exception as e:
        logger.debug(f"Checkpointer health check failed: {e}")
        return False


async def _recreate_checkpointer() -> Optional[AsyncPostgresSaver]:
    """Recreate checkpointer if connection is stale."""
    global _checkpointer, _checkpointer_cm
    
    # Close existing checkpointer
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error closing stale checkpointer: {e}")
    
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
    try:
        # Initialize checkpointer on startup
        await get_checkpointer()
        yield
    finally:
        # Cleanup on shutdown
        await close_checkpointer()

