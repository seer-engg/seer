"""
Unit tests for SSE helper (stream_events_sse).
"""
import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from seer.api.agents.workflow.chat_schema import StreamEventType
from seer.api.agents.workflow.sse import (
    _format_sse,
    _read_stream_loop,
    stream_events_sse,
)


def _make_event(event_type: StreamEventType, session_id: int = 1, data: dict = None) -> dict:
    """Build a StreamEvent dict for testing."""
    return {
        "type": event_type.value,
        "data": data or {},
        "session_id": session_id,
    }


def test_format_sse():
    """_format_sse should produce correct SSE wire format."""
    result = _format_sse("12345-0", '{"type":"done"}')
    assert result == 'id: 12345-0\ndata: {"type":"done"}\n\n'


@pytest.mark.asyncio
async def test_read_stream_loop_terminates_on_done():
    """_read_stream_loop should terminate when DONE event is received."""
    done_event = _make_event(StreamEventType.DONE, session_id=1)
    tool_event = _make_event(StreamEventType.TOOL_START, session_id=1, data={"tool_name": "foo"})

    # Simulate two reads: first returns tool_event, second returns done_event
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)

    call_count = 0

    async def fake_xread(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [("stream_key", [("100-0", {"data": json.dumps(tool_event)})])]
        if call_count == 2:
            return [("stream_key", [("200-0", {"data": json.dumps(done_event)})])]
        return []

    mock_redis.xread = fake_xread

    events = []
    async for sse_str in _read_stream_loop(mock_redis, "test_key", "0"):
        events.append(sse_str)

    assert len(events) == 2
    # First event is tool_start
    assert "100-0" in events[0]
    assert "tool_start" in events[0]
    # Second event is done
    assert "200-0" in events[1]
    assert "done" in events[1]


@pytest.mark.asyncio
async def test_read_stream_loop_resumes_from_last_event_id():
    """_read_stream_loop cursor should start from provided last_event_id."""
    done_event = _make_event(StreamEventType.DONE, session_id=42)

    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)

    captured_cursor = None

    async def fake_xread(streams, **kwargs):
        nonlocal captured_cursor
        # streams is {stream_key: cursor}
        captured_cursor = list(streams.values())[0]
        return [("stream_key", [("999-0", {"data": json.dumps(done_event)})])]

    mock_redis.xread = fake_xread

    events = []
    async for sse_str in _read_stream_loop(mock_redis, "test_key", "500-0"):
        events.append(sse_str)

    # Cursor passed to first XREAD should be "500-0"
    assert captured_cursor == "500-0"


@pytest.mark.asyncio
async def test_read_stream_loop_continues_on_empty_block_result():
    """_read_stream_loop should keep polling after empty XREAD result."""
    done_event = _make_event(StreamEventType.DONE, session_id=1)

    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)

    call_count = 0

    async def fake_xread(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return []  # Empty (timeout, no new messages)
        if call_count == 2:
            return [("stream_key", [("300-0", {"data": json.dumps(done_event)})])]
        return []

    mock_redis.xread = fake_xread

    events = []
    async for sse_str in _read_stream_loop(mock_redis, "test_key", "0"):
        events.append(sse_str)

    assert len(events) == 1
    assert "done" in events[0]
    assert call_count == 2


@pytest.mark.asyncio
async def test_stream_events_sse_full_flow():
    """stream_events_sse should connect to Redis, replay stream, and stop at DONE."""
    session_info = _make_event(StreamEventType.SESSION_INFO, session_id=10, data={"session_id": 10})
    agent_end = _make_event(StreamEventType.AGENT_END, session_id=10, data={"content": "Hello!"})
    done = _make_event(StreamEventType.DONE, session_id=10)

    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    call_count = 0

    async def fake_xread(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [("k", [
                ("10-0", {"data": json.dumps(session_info)}),
                ("10-1", {"data": json.dumps(agent_end)}),
                ("10-2", {"data": json.dumps(done)}),
            ])]
        return []

    mock_redis.xread = fake_xread

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"

            events = []
            async for sse_str in stream_events_sse(10, last_event_id=None):
                events.append(sse_str)

    assert len(events) == 3
    assert "session_info" in events[0]
    assert "agent_end" in events[1]
    assert "done" in events[2]
    mock_redis.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_stream_events_sse_missing_stream_completed_fallback():
    """When stream is missing and session is COMPLETED, fall back to DB history."""
    from unittest.mock import AsyncMock

    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.aclose = AsyncMock()

    # Mock DB models
    mock_session = MagicMock()
    mock_session.current_execution_status = MagicMock()
    mock_session.current_execution_status.__eq__ = MagicMock(return_value=True)

    from seer.database.workflow_models import ChatExecutionStatus

    mock_session.current_execution_status = ChatExecutionStatus.COMPLETED

    mock_msg1 = MagicMock()
    mock_msg1.role = "assistant"
    mock_msg1.content = "Here is your answer"

    mock_filter_chain = AsyncMock()
    mock_filter_chain.order_by = MagicMock(return_value=mock_filter_chain)
    mock_filter_chain.limit = MagicMock(return_value=mock_filter_chain)
    mock_filter_chain.all = AsyncMock(return_value=[mock_msg1])

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            with patch("seer.database.workflow_models.WorkflowChatSession.get_or_none", new_callable=AsyncMock, return_value=mock_session):
                with patch("seer.database.WorkflowChatMessage.filter", return_value=mock_filter_chain):
                    events = []
                    async for sse_str in stream_events_sse(99, last_event_id=None):
                        events.append(sse_str)

    # Should have at least agent_end (from DB message) + done
    assert len(events) >= 2
    assert any("done" in e for e in events)
