"""
Unit tests for Nexus agent tool tracking decorator.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.agents.nexus.tracking import track_nexus_tool

pytestmark = pytest.mark.unit


# =============================================================================
# track_nexus_tool decorator tests
# =============================================================================

@pytest.mark.asyncio
async def test_track_nexus_tool_fires_on_success():
    """nexus_tool_called event fires on successful tool invocation."""
    @track_nexus_tool("create_workflow")
    async def my_tool() -> str:
        return "ok"

    mock_user = MagicMock()
    mock_user.email = "user@example.com"

    with (
        patch("seer.agents.nexus.context._current_thread_id") as mock_thread_var,
        patch("seer.agents.nexus.context.get_user_for_thread", new_callable=AsyncMock) as mock_get_user,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
        patch("seer.analytics.workflow_tracking.config") as mock_cfg,
        patch("seer.agents.nexus.tracking.config") as mock_tracking_cfg,
    ):
        mock_thread_var.get.return_value = "thread_abc"
        mock_get_user.return_value = mock_user
        mock_cfg.is_posthog_configured = True
        mock_cfg.seer_mode = "cloud"
        mock_tracking_cfg.is_posthog_configured = True

        result = await my_tool()
        # Yield control to the event loop so the fire-and-forget tracking task runs
        await asyncio.sleep(0)

        assert result == "ok"
        mock_capture.assert_called_once()
        call_kwargs = mock_capture.call_args.kwargs
        assert call_kwargs["event"] == "nexus_tool_called"
        assert call_kwargs["distinct_id"] == "user@example.com"
        props = call_kwargs["properties"]
        assert props["tool_name"] == "create_workflow"
        assert props["success"] is True
        assert "error" not in props


@pytest.mark.asyncio
async def test_track_nexus_tool_fires_on_failure():
    """nexus_tool_called event fires even when tool raises, and re-raises the exception."""
    @track_nexus_tool("bad_tool")
    async def failing_tool() -> str:
        raise ValueError("something went wrong")

    mock_user = MagicMock()
    mock_user.email = "user@example.com"

    with (
        patch("seer.agents.nexus.context._current_thread_id") as mock_thread_var,
        patch("seer.agents.nexus.context.get_user_for_thread", new_callable=AsyncMock) as mock_get_user,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
        patch("seer.analytics.workflow_tracking.config") as mock_cfg,
        patch("seer.agents.nexus.tracking.config") as mock_tracking_cfg,
    ):
        mock_thread_var.get.return_value = "thread_abc"
        mock_get_user.return_value = mock_user
        mock_cfg.is_posthog_configured = True
        mock_cfg.seer_mode = "cloud"
        mock_tracking_cfg.is_posthog_configured = True

        with pytest.raises(ValueError, match="something went wrong"):
            await failing_tool()

        # Yield control so the fire-and-forget task runs
        await asyncio.sleep(0)

        mock_capture.assert_called_once()
        props = mock_capture.call_args.kwargs["properties"]
        assert props["success"] is False
        assert "something went wrong" in props["error"]


@pytest.mark.asyncio
async def test_track_nexus_tool_skips_when_posthog_not_configured():
    """No tracking occurs when PostHog is not configured."""
    @track_nexus_tool("my_tool")
    async def my_tool() -> str:
        return "ok"

    with (
        patch("seer.agents.nexus.tracking.config") as mock_cfg,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
    ):
        mock_cfg.is_posthog_configured = False
        result = await my_tool()

    assert result == "ok"
    mock_capture.assert_not_called()


@pytest.mark.asyncio
async def test_track_nexus_tool_skips_when_no_thread_id():
    """No tracking occurs when there is no active thread_id in context."""
    @track_nexus_tool("orphan_tool")
    async def my_tool() -> str:
        return "ok"

    with (
        patch("seer.agents.nexus.context._current_thread_id") as mock_thread_var,
        patch("seer.agents.nexus.tracking.config") as mock_cfg,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
    ):
        mock_thread_var.get.return_value = None
        mock_cfg.is_posthog_configured = True

        result = await my_tool()

    assert result == "ok"
    mock_capture.assert_not_called()
