"""Unit tests for collaboration SSE helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from seer.api.collaboration.sse import _format_sse, stream_org_events_sse


def test_format_sse():
    assert _format_sse("123-0", '{"event_type":"workflow.updated"}') == (
        'id: 123-0\nevent: collaboration\ndata: {"event_type":"workflow.updated"}\n\n'
    )


@pytest.mark.asyncio
async def test_stream_org_events_sse_emits_sync_required_when_stream_missing():
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.aclose = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            generator = stream_org_events_sse(organization_id=12, last_event_id="100-0")
            first_chunk = await anext(generator)
            await generator.aclose()

    assert "sync.required" in first_chunk
    assert '"organization_id":12' in first_chunk


@pytest.mark.asyncio
async def test_stream_org_events_sse_emits_heartbeat_on_empty_read():
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.xread = AsyncMock(return_value=[])
    mock_redis.aclose = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            generator = stream_org_events_sse(organization_id=12)
            first_chunk = await anext(generator)
            await generator.aclose()

    assert first_chunk == ": heartbeat\n\n"


@pytest.mark.asyncio
async def test_stream_org_events_sse_reads_after_last_event_id():
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    async def fake_xread(streams, **kwargs):
        assert list(streams.values())[0] == "200-0"
        return [("org:events:12", [("201-0", {"data": '{"event_type":"workflow.updated"}'})])]

    mock_redis.xread = fake_xread

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            generator = stream_org_events_sse(organization_id=12, last_event_id="200-0")
            first_chunk = await anext(generator)
            await generator.aclose()

    assert "201-0" in first_chunk
    assert "workflow.updated" in first_chunk
