"""
Unit tests for worker.tasks.polling module.

Tests:
- Single tick execution
- Error propagation
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
class TestPollTriggersOnce:
    """Tests for poll_triggers_once task."""

    @pytest.mark.asyncio
    async def test_creates_engine_and_ticks(self):
        from seer.worker.tasks.polling import poll_triggers_once

        mock_engine = AsyncMock()

        with (
            patch("seer.core.triggers.polling.engine.TriggerPollEngine", return_value=mock_engine),
            patch("seer.worker.trigger_dispatcher.dispatch_trigger_event"),
        ):
            await poll_triggers_once()

        mock_engine.tick.assert_called_once()

    @pytest.mark.asyncio
    async def test_propagates_engine_errors(self):
        from seer.worker.tasks.polling import poll_triggers_once

        mock_engine = AsyncMock()
        mock_engine.tick.side_effect = RuntimeError("Poll failed")

        with (
            patch("seer.core.triggers.polling.engine.TriggerPollEngine", return_value=mock_engine),
            patch("seer.worker.trigger_dispatcher.dispatch_trigger_event"),
        ):
            with pytest.raises(RuntimeError, match="Poll failed"):
                await poll_triggers_once()
