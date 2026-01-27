"""Global event loop reference for cross-thread async scheduling."""
import asyncio
from typing import Optional

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
