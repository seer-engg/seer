"""
Unit tests for StreamPublisher.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.agents.workflow.chat_schema import StreamEvent, StreamEventType
from seer.agents.nexus.stream_publisher import StreamPublisher, STREAM_KEY_PREFIX, STREAM_TTL_SECONDS


@pytest.fixture
def publisher():
    return StreamPublisher(session_id=42)


@pytest.mark.asyncio
async def test_stream_key_format(publisher):
    assert publisher.stream_key == f"{STREAM_KEY_PREFIX}:42"


@pytest.mark.asyncio
async def test_publish_xadd_and_expire():
    """publish() should XADD to stream and set TTL."""
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value="1234567890-0")
    mock_redis.expire = AsyncMock()

    publisher = StreamPublisher(session_id=99)

    with patch.object(publisher, "_get_redis", return_value=mock_redis):
        msg_id = await publisher.publish(StreamEventType.TOOL_START, {"tool_name": "my_tool"})

    assert msg_id == "1234567890-0"
    mock_redis.xadd.assert_called_once()
    call_args = mock_redis.xadd.call_args

    # First positional arg is the stream key
    assert call_args[0][0] == f"{STREAM_KEY_PREFIX}:99"
    # Second arg is the fields dict — contains "data" with JSON
    fields = call_args[0][1]
    assert "data" in fields

    # The JSON should be deserializable as a StreamEvent
    import json
    event_dict = json.loads(fields["data"])
    assert event_dict["type"] == StreamEventType.TOOL_START.value
    assert event_dict["data"]["tool_name"] == "my_tool"
    assert event_dict["session_id"] == 99

    # expire() should be called with the stream key and TTL
    mock_redis.expire.assert_called_once_with(f"{STREAM_KEY_PREFIX}:99", STREAM_TTL_SECONDS)


@pytest.mark.asyncio
async def test_publish_returns_none_on_error():
    """publish() should return None (not raise) when Redis fails."""
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(side_effect=ConnectionError("Redis down"))

    publisher = StreamPublisher(session_id=1)
    with patch.object(publisher, "_get_redis", return_value=mock_redis):
        result = await publisher.publish(StreamEventType.AGENT_START, {})

    assert result is None


@pytest.mark.asyncio
async def test_close_publishes_done_and_closes():
    """close() should publish DONE and close the Redis connection."""
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value="1-0")
    mock_redis.expire = AsyncMock()
    mock_redis.aclose = AsyncMock()

    publisher = StreamPublisher(session_id=5)
    publisher._redis = mock_redis  # Inject mock directly to bypass lazy init

    with patch.object(publisher, "_get_redis", return_value=mock_redis):
        await publisher.close()

    # The last XADD call should be for DONE event
    last_call = mock_redis.xadd.call_args_list[-1]
    import json
    fields = last_call[0][1]
    event_dict = json.loads(fields["data"])
    assert event_dict["type"] == StreamEventType.DONE.value

    mock_redis.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_publish_done_does_not_close():
    """publish_done() publishes DONE but keeps connection open."""
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value="1-0")
    mock_redis.expire = AsyncMock()
    mock_redis.aclose = AsyncMock()

    publisher = StreamPublisher(session_id=7)
    publisher._redis = mock_redis

    with patch.object(publisher, "_get_redis", return_value=mock_redis):
        await publisher.publish_done()

    # No close call
    mock_redis.aclose.assert_not_called()

    # Should have published DONE
    import json
    fields = mock_redis.xadd.call_args[0][1]
    event_dict = json.loads(fields["data"])
    assert event_dict["type"] == StreamEventType.DONE.value


@pytest.mark.asyncio
async def test_lazy_redis_init():
    """_get_redis() should create connection lazily on first call."""
    mock_redis_instance = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis_instance) as mock_from_url:
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://test:6379"

            publisher = StreamPublisher(session_id=1)
            assert publisher._redis is None

            # First call creates connection
            r1 = await publisher._get_redis()
            assert r1 is mock_redis_instance
            mock_from_url.assert_called_once_with("redis://test:6379", decode_responses=True)

            # Second call reuses connection
            r2 = await publisher._get_redis()
            assert r2 is mock_redis_instance
            assert mock_from_url.call_count == 1  # Still only once
