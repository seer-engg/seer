"""Global event loop reference for cross-thread async scheduling."""
import asyncio
import logging
import threading
from typing import Coroutine, Optional

# Global constant for main event loop reference (application singleton)
_MAIN_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Store reference to main event loop during application startup.

    This allows background threads and thread pool workers to schedule
    coroutines on the main loop using asyncio.run_coroutine_threadsafe().

    Args:
        loop: The main event loop (typically from FastAPI/uvicorn)
    """
    global _MAIN_EVENT_LOOP  # pylint: disable=global-statement  # Reason: application singleton pattern
    _MAIN_EVENT_LOOP = loop


def get_main_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """
    Get reference to main event loop.

    Returns:
        The main event loop, or None if not yet initialized
    """
    return _MAIN_EVENT_LOOP


def schedule_async_task(
    coro: Coroutine,
    logger: logging.Logger,
    error_message: str = "Failed to schedule async task",
) -> None:
    """
    Schedule an async coroutine to run in a thread-safe manner.

    Handles three scenarios:
    1. Running event loop exists (async context) → create_task()
    2. Main thread without running loop (sync context) → run_until_complete()
    3. Thread pool executor → run_coroutine_threadsafe() on main loop

    Args:
        coro: The coroutine to schedule
        logger: Logger instance for error reporting
        error_message: Custom error message prefix for logging
    """
    try:
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(coro)
        except RuntimeError:
            if threading.current_thread() is threading.main_thread():
                loop = asyncio.get_event_loop()
                loop.run_until_complete(coro)
            else:
                main_loop = get_main_event_loop()
                if main_loop is not None:
                    asyncio.run_coroutine_threadsafe(coro, main_loop)
                else:
                    # Close the coroutine to prevent "never awaited" warning
                    coro.close()
                    logger.error(
                        "%s: Main event loop not available - cannot schedule from thread pool",
                        error_message
                    )
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: scheduling failures should not break execution
        logger.error("%s: %s", error_message, e, exc_info=True)
