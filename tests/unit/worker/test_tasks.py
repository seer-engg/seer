"""
Unit tests for worker.tasks module.

Tests Taskiq worker task execution and error handling.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, patch

import pytest


# =============================================================================
# Trigger Event Task Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEventTask:
    """Tests for trigger_event_task function."""

    @pytest.mark.asyncio
    async def test_trigger_event_task_success(self):
        """Test trigger_event_task processes event successfully."""
        from seer.worker.tasks.triggers import trigger_event_task

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock) as mock_process:
            await trigger_event_task(subscription_id=123, event_id=456)

        mock_process.assert_called_once_with(subscription_id=123, event_id=456)

    @pytest.mark.asyncio
    async def test_trigger_event_task_logs_start(self):
        """Test trigger_event_task logs processing start."""
        from seer.worker.tasks.triggers import trigger_event_task

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock):
            with patch("seer.worker.tasks.triggers.logger") as mock_logger:
                await trigger_event_task(subscription_id=100, event_id=200)

        # Should have logged info about processing
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_trigger_event_task_exception_reraises(self):
        """Test trigger_event_task re-raises exceptions after logging."""
        from seer.worker.tasks.triggers import trigger_event_task

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock, side_effect=RuntimeError("Processing failed")):
            with patch("seer.worker.tasks.triggers.logger") as mock_logger:
                with pytest.raises(RuntimeError, match="Processing failed"):
                    await trigger_event_task(subscription_id=111, event_id=222)

        # Should have logged exception
        mock_logger.exception.assert_called()

    @pytest.mark.asyncio
    async def test_trigger_event_task_logs_completion(self):
        """Test trigger_event_task logs successful completion."""
        from seer.worker.tasks.triggers import trigger_event_task

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock):
            with patch("seer.worker.tasks.triggers.logger") as mock_logger:
                await trigger_event_task(subscription_id=333, event_id=444)

        # Check that info was called multiple times (start and completion)
        assert mock_logger.info.call_count >= 2


# =============================================================================
# Additional Worker Task Patterns
# =============================================================================


@pytest.mark.unit
class TestWorkerTaskPatterns:
    """Tests for common worker task patterns."""

    @pytest.mark.asyncio
    async def test_task_with_logging_context(self):
        """Test that tasks include proper logging context."""
        from seer.worker.tasks.triggers import trigger_event_task

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock):
            with patch("seer.worker.tasks.triggers.logger") as mock_logger:
                await trigger_event_task(subscription_id=555, event_id=666)

        # Verify logging includes extra context
        call_args = mock_logger.info.call_args_list[0]
        assert "extra" in call_args.kwargs or len(call_args.args) >= 2

    @pytest.mark.asyncio
    async def test_task_handles_db_errors(self):
        """Test that tasks handle database errors gracefully."""
        from seer.worker.tasks.triggers import trigger_event_task

        class MockDBError(Exception):
            pass

        with patch("seer.worker.tasks.triggers.process_trigger_event", new_callable=AsyncMock, side_effect=MockDBError("Connection lost")):
            with patch("seer.worker.tasks.triggers.logger"):
                with pytest.raises(MockDBError):
                    await trigger_event_task(subscription_id=777, event_id=888)
