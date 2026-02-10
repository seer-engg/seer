"""Unit tests for event loop scheduling utilities."""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from seer.core.event_loop import (
    get_main_event_loop,
    schedule_async_task,
    set_main_event_loop,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def sample_coroutine():
    """Fixture providing a sample coroutine for testing."""
    async def sample_coro():
        await asyncio.sleep(0.001)
        return "completed"
    return sample_coro


@pytest.mark.asyncio
async def test_schedule_async_task_from_async_context(mock_logger, sample_coroutine):
    """Test scheduling from an async context uses create_task."""
    # Track if coroutine was executed
    executed = []

    async def track_coro():
        await asyncio.sleep(0.001)
        executed.append(True)

    # Schedule the task
    schedule_async_task(
        coro=track_coro(),
        logger=mock_logger,
        error_message="Test error",
    )

    # Wait for task to complete
    await asyncio.sleep(0.01)

    # Verify task was executed
    assert len(executed) == 1
    assert executed[0] is True

    # Verify no errors were logged
    mock_logger.error.assert_not_called()


def test_schedule_async_task_from_main_thread_sync_context(mock_logger):
    """Test scheduling from main thread sync context uses run_until_complete."""
    # Track if coroutine was executed
    executed = []

    async def track_coro():
        await asyncio.sleep(0.001)
        executed.append(True)

    # Ensure we're on main thread and no running loop
    assert threading.current_thread() is threading.main_thread()
    try:
        asyncio.get_running_loop()
        pytest.fail("Expected no running loop")
    except RuntimeError:
        pass  # Expected

    # Schedule the task
    schedule_async_task(
        coro=track_coro(),
        logger=mock_logger,
        error_message="Test error",
    )

    # Verify task was executed
    assert len(executed) == 1
    assert executed[0] is True

    # Verify no errors were logged
    mock_logger.error.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_async_task_from_thread_pool_with_main_loop(mock_logger):
    """Test scheduling from thread pool uses run_coroutine_threadsafe."""
    # Set up main event loop
    main_loop = asyncio.get_running_loop()
    set_main_event_loop(main_loop)

    # Track if coroutine was executed
    executed = []

    async def track_coro():
        executed.append(True)

    # Schedule from thread pool
    def thread_task():
        schedule_async_task(
            coro=track_coro(),
            logger=mock_logger,
            error_message="Test error",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(thread_task)
        future.result(timeout=2)

    # Wait for async task to complete
    await asyncio.sleep(0.01)

    # Verify task was executed
    assert len(executed) == 1
    assert executed[0] is True

    # Verify no errors were logged
    mock_logger.error.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_async_task_from_thread_pool_without_main_loop(mock_logger):
    """Test scheduling from thread pool without main loop logs error."""
    # Clear main event loop
    set_main_event_loop(None)

    # Track if coroutine was executed
    executed = []

    async def track_coro():
        executed.append(True)

    # Schedule from thread pool
    def thread_task():
        schedule_async_task(
            coro=track_coro(),
            logger=mock_logger,
            error_message="Test error",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(thread_task)
        future.result(timeout=2)

    # Wait a bit
    await asyncio.sleep(0.01)

    # Verify task was NOT executed (no main loop available)
    assert len(executed) == 0

    # Verify error was logged
    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args
    assert "Main event loop not available" in error_call[0][0]


@pytest.mark.asyncio
async def test_schedule_async_task_with_exception_in_coroutine(mock_logger):
    """Test that exceptions in coroutines are handled gracefully."""
    # Track if coroutine was executed
    executed = []

    async def failing_coro():
        executed.append(True)
        raise ValueError("Test exception")

    # Schedule the task
    schedule_async_task(
        coro=failing_coro(),
        logger=mock_logger,
        error_message="Test error",
    )

    # Wait for task to complete (and fail)
    await asyncio.sleep(0.01)

    # Verify coroutine started execution
    assert len(executed) == 1

    # Note: The exception in the coroutine itself won't be caught by schedule_async_task
    # because it's running in a separate task. The exception would be logged by asyncio's
    # default exception handler or caught within the coroutine itself.


def test_schedule_async_task_scheduling_exception(mock_logger):
    """Test that exceptions during scheduling are logged."""
    async def sample_coro():
        await asyncio.sleep(0.001)

    # Mock get_running_loop to raise an unexpected exception
    with patch("asyncio.get_running_loop", side_effect=Exception("Unexpected error")):
        # This should catch and log the exception
        schedule_async_task(
            coro=sample_coro(),
            logger=mock_logger,
            error_message="Test scheduling error",
        )

    # Verify error was logged
    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args
    # Check the first positional argument (format string) and keyword arguments
    assert error_call[0][0] == "%s: %s"
    assert error_call[0][1] == "Test scheduling error"


def test_set_and_get_main_event_loop():
    """Test setting and getting the main event loop."""
    # Create a mock event loop
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    # Set the loop
    set_main_event_loop(mock_loop)

    # Get the loop
    retrieved_loop = get_main_event_loop()

    # Verify it's the same loop
    assert retrieved_loop is mock_loop

    # Clean up
    set_main_event_loop(None)
