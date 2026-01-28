"""
Integration tests for trigger polling scheduler.

Tests:
- Scheduler start/stop lifecycle
- Periodic tick execution
- Error handling in the polling loop
- Graceful shutdown
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from seer.core.triggers.polling.scheduler import TriggerPollScheduler


# =============================================================================
# TriggerPollScheduler Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_start_stop(db_engine):
    """Test scheduler start and stop lifecycle."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Start scheduler
    await scheduler.start()

    # Verify task is created
    assert scheduler._task is not None
    assert not scheduler._task.done()

    # Stop scheduler
    await scheduler.stop()

    # Verify task is done
    assert scheduler._task.done()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_periodic_ticks(db_engine):
    """Test that scheduler periodically calls engine.tick()."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=0.1,  # Fast interval for testing
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    tick_count = 0

    async def mock_tick():
        nonlocal tick_count
        tick_count += 1

    with patch.object(scheduler.engine, "tick", side_effect=mock_tick):
        await scheduler.start()

        # Wait for a few ticks
        await asyncio.sleep(0.35)

        await scheduler.stop()

    # Should have at least 2-3 ticks in 0.35 seconds with 0.1s interval
    assert tick_count >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_handles_tick_errors(db_engine):
    """Test that scheduler continues running even if tick() fails."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=0.1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    tick_count = 0

    async def mock_tick_with_error():
        nonlocal tick_count
        tick_count += 1
        if tick_count == 2:
            raise ValueError("Tick failed")

    with patch.object(scheduler.engine, "tick", side_effect=mock_tick_with_error):
        await scheduler.start()

        # Wait for multiple ticks
        await asyncio.sleep(0.35)

        await scheduler.stop()

    # Should continue ticking after error
    assert tick_count >= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_double_start_is_idempotent(db_engine):
    """Test that calling start() multiple times doesn't create multiple tasks."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    await scheduler.start()
    first_task = scheduler._task

    # Call start again
    await scheduler.start()
    second_task = scheduler._task

    # Should be the same task
    assert first_task is second_task

    await scheduler.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_stop_without_start(db_engine):
    """Test that stop() is safe to call without start()."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    # Should not raise error
    await scheduler.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_stop_is_graceful(db_engine):
    """Test that stop() waits for current tick to complete."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    tick_started = asyncio.Event()
    tick_completed = asyncio.Event()

    async def slow_tick():
        tick_started.set()
        await asyncio.sleep(0.2)
        tick_completed.set()

    with patch.object(scheduler.engine, "tick", side_effect=slow_tick):
        await scheduler.start()

        # Wait for tick to start
        await tick_started.wait()

        # Stop scheduler
        await scheduler.stop()

        # Tick should have completed
        assert tick_completed.is_set()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_respects_interval(db_engine):
    """Test that scheduler respects the configured interval."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=0.2,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    tick_times = []

    async def record_tick():
        tick_times.append(asyncio.get_event_loop().time())

    with patch.object(scheduler.engine, "tick", side_effect=record_tick):
        await scheduler.start()

        # Wait for multiple ticks
        await asyncio.sleep(0.5)

        await scheduler.stop()

    # Calculate intervals between ticks
    if len(tick_times) >= 2:
        intervals = [tick_times[i + 1] - tick_times[i] for i in range(len(tick_times) - 1)]

        # Intervals should be approximately 0.2 seconds
        for interval in intervals:
            assert 0.15 <= interval <= 0.3  # Allow some tolerance


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_worker_id_set(db_engine):
    """Test that scheduler engine has a worker ID."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=1,
        max_batch_size=10,
        lock_timeout_seconds=60,
        trigger_event_dispatcher=mock_dispatcher,
    )

    worker_id = scheduler.engine.worker_id

    assert worker_id is not None
    assert worker_id.startswith("poller-")
    assert len(worker_id) > len("poller-")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduler_configuration_parameters(db_engine):
    """Test that scheduler correctly passes configuration to engine."""
    mock_dispatcher = AsyncMock()
    scheduler = TriggerPollScheduler(
        interval_seconds=10,
        max_batch_size=25,
        lock_timeout_seconds=120,
        trigger_event_dispatcher=mock_dispatcher,
    )

    assert scheduler.interval_seconds == 10
    assert scheduler.engine.max_batch_size == 25
    assert scheduler.engine.lock_timeout.total_seconds() == 120
    assert scheduler.engine.trigger_event_dispatcher is mock_dispatcher
