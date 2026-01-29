"""
Integration tests for trigger worker tasks.

Tests:
- Trigger event processing task
- Task error handling
- Integration with database models
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from seer.database.workflow_models import TriggerEvent, TriggerEventStatus
from seer.worker.tasks.triggers import trigger_event_task


# =============================================================================
# Trigger Event Task Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_success(db_engine, test_trigger_subscription):
    """Test successful trigger event processing."""
    # Create trigger event
    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process:
        mock_process.return_value = None

        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event.id,
        )

        # Verify service was called
        mock_process.assert_called_once_with(
            subscription_id=test_trigger_subscription.id,
            event_id=event.id,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_handles_errors(db_engine, test_trigger_subscription):
    """Test task handles processing errors."""
    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process:
        mock_process.side_effect = ValueError("Processing failed")

        # Task should raise the exception
        with pytest.raises(ValueError, match="Processing failed"):
            await trigger_event_task(
                subscription_id=test_trigger_subscription.id,
                event_id=event.id,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_logs_info(db_engine, test_trigger_subscription):
    """Test that task logs processing information."""
    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process, \
         patch("seer.worker.tasks.triggers.logger") as mock_logger:

        mock_process.return_value = None

        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event.id,
        )

        # Verify logging
        assert mock_logger.info.call_count >= 2  # Start and completion logs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_logs_errors(db_engine, test_trigger_subscription):
    """Test that task logs errors on failure."""
    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process, \
         patch("seer.worker.tasks.triggers.logger") as mock_logger:

        mock_process.side_effect = RuntimeError("Task failed")

        with pytest.raises(RuntimeError):
            await trigger_event_task(
                subscription_id=test_trigger_subscription.id,
                event_id=event.id,
            )

        # Verify error logging
        mock_logger.exception.assert_called_once()


# =============================================================================
# Task Integration with Database Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_updates_event_status(db_engine, test_trigger_subscription):
    """Test that trigger event processing updates event status."""
    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    async def mock_process_and_update(subscription_id, event_id):
        # Simulate updating event status
        event_obj = await TriggerEvent.get(id=event_id)
        event_obj.status = TriggerEventStatus.PROCESSED
        await event_obj.save()

    with patch("seer.worker.tasks.triggers.process_trigger_event",
               side_effect=mock_process_and_update):

        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event.id,
        )

        # Verify event status was updated
        await event.refresh_from_db()
        assert event.status == TriggerEventStatus.PROCESSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_with_multiple_events(db_engine, test_trigger_subscription):
    """Test processing multiple trigger events."""
    # Create multiple events
    event1 = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"id": "1"},
        status=TriggerEventStatus.RECEIVED,
    )

    event2 = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"id": "2"},
        status=TriggerEventStatus.RECEIVED,
    )

    processed_events = []

    async def track_processing(subscription_id, event_id):
        processed_events.append(event_id)

    with patch("seer.worker.tasks.triggers.process_trigger_event",
               side_effect=track_processing):

        # Process both events
        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event1.id,
        )
        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event2.id,
        )

        # Verify both events were processed
        assert len(processed_events) == 2
        assert event1.id in processed_events
        assert event2.id in processed_events


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_with_invalid_subscription_id(db_engine):
    """Test task handling of invalid subscription ID."""
    event = await TriggerEvent.create(
        trigger_key="test.trigger",
        event={"data": "test"},
    )

    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process:
        mock_process.side_effect = ValueError("Subscription not found")

        with pytest.raises(ValueError, match="Subscription not found"):
            await trigger_event_task(
                subscription_id=99999,  # Non-existent
                event_id=event.id,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_with_invalid_event_id(db_engine, test_trigger_subscription):
    """Test task handling of invalid event ID."""
    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process:
        mock_process.side_effect = ValueError("Event not found")

        with pytest.raises(ValueError, match="Event not found"):
            await trigger_event_task(
                subscription_id=test_trigger_subscription.id,
                event_id=99999,  # Non-existent
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_creates_workflow_run(db_engine, test_trigger_subscription, test_workflow):
    """Test that processing trigger event can create workflow run."""
    from seer.database.workflow_models import WorkflowRun

    event = await TriggerEvent.create(
        trigger_key=test_trigger_subscription.trigger_key,
        event={"data": "test"},
        status=TriggerEventStatus.RECEIVED,
    )

    async def mock_process_creates_run(subscription_id, event_id):
        # Simulate creating a workflow run
        await WorkflowRun.create(
            user=test_workflow.user,
            workflow=test_workflow,
            spec={"version": "2"},
            trigger_event_id=event_id,
        )

    with patch("seer.worker.tasks.triggers.process_trigger_event",
               side_effect=mock_process_creates_run):

        await trigger_event_task(
            subscription_id=test_trigger_subscription.id,
            event_id=event.id,
        )

        # Verify run was created
        runs = await WorkflowRun.filter(trigger_event_id=event.id).all()
        assert len(runs) == 1


# =============================================================================
# Task Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_trigger_event_task_is_broker_task():
    """Test that trigger_event_task is registered as broker task."""
    # Verify task has broker task attributes
    assert hasattr(trigger_event_task, "kiq")
    assert callable(trigger_event_task.kiq)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_callable():
    """Test that task is directly callable (not just via broker)."""
    with patch("seer.worker.tasks.triggers.process_trigger_event") as mock_process:
        mock_process.return_value = None

        # Should be callable directly
        await trigger_event_task(
            subscription_id=1,
            event_id=1,
        )

        assert mock_process.called


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_event_task_concurrent_processing(db_engine, test_trigger_subscription):
    """Test that multiple events can be processed concurrently."""
    import asyncio

    # Create multiple events
    events = []
    for i in range(3):
        event = await TriggerEvent.create(
            trigger_key=test_trigger_subscription.trigger_key,
            event={"id": str(i)},
            status=TriggerEventStatus.RECEIVED,
        )
        events.append(event)

    processed = []

    async def mock_process(subscription_id, event_id):
        await asyncio.sleep(0.01)  # Simulate processing time
        processed.append(event_id)

    with patch("seer.worker.tasks.triggers.process_trigger_event",
               side_effect=mock_process):

        # Process all events concurrently
        tasks = [
            trigger_event_task(
                subscription_id=test_trigger_subscription.id,
                event_id=event.id,
            )
            for event in events
        ]

        await asyncio.gather(*tasks)

        # All events should be processed
        assert len(processed) == 3
        assert all(event.id in processed for event in events)
