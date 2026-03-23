"""Unit tests for collaboration SSE helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from seer.api.collaboration.sse import _format_sse, _release_locks_on_disconnect, stream_org_events_sse


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


@pytest.fixture
def mock_user():
    return SimpleNamespace(
        id=1,
        user_id="user_123",
        first_name="Test",
        last_name="User",
        email="test@example.com",
    )


@pytest.mark.asyncio
async def test_release_locks_on_disconnect_calls_lock_service(mock_user):
    """Test that _release_locks_on_disconnect releases locks and publishes events."""
    mock_lock = SimpleNamespace(
        tab_id="tab-1",
        holder_name="Test User",
    )

    mock_lock_service = AsyncMock()
    mock_lock_service.release_locks_for_connection = AsyncMock(
        return_value=[("wf_123", mock_lock), ("wf_456", mock_lock)]
    )
    mock_lock_service.close = AsyncMock()

    with patch(
        "seer.services.collaboration.WorkflowLockService", return_value=mock_lock_service
    ):
        with patch("seer.services.collaboration.publish_collaboration_event") as mock_publish:
            mock_publish.return_value = None
            await _release_locks_on_disconnect(
                organization_id=12,
                sse_connection_id="12:user_123:tab-1",
                user=mock_user,
            )

    mock_lock_service.release_locks_for_connection.assert_called_once_with(
        12, "12:user_123:tab-1"
    )
    assert mock_publish.call_count == 2
    mock_lock_service.close.assert_called_once()


@pytest.mark.asyncio
async def test_stream_org_events_sse_releases_locks_on_close(mock_user):
    """Test that SSE stream releases locks when closed with user and tab_id."""
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.xread = AsyncMock(return_value=[])
    mock_redis.aclose = AsyncMock()

    mock_lock_service = AsyncMock()
    mock_lock_service.release_locks_for_connection = AsyncMock(return_value=[])
    mock_lock_service.close = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            with patch(
                "seer.services.collaboration.WorkflowLockService",
                return_value=mock_lock_service,
            ):
                generator = stream_org_events_sse(
                    organization_id=12,
                    user=mock_user,
                    tab_id="tab-1",
                )
                await anext(generator)  # Get first chunk (heartbeat)
                await generator.aclose()

    mock_lock_service.release_locks_for_connection.assert_called_once_with(
        12, "12:user_123:tab-1"
    )
    mock_lock_service.close.assert_called_once()


@pytest.mark.asyncio
async def test_stream_org_events_sse_skips_lock_release_without_tab_id(mock_user):
    """Test that SSE stream does not release locks when tab_id is not provided."""
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.xread = AsyncMock(return_value=[])
    mock_redis.aclose = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch("seer.config.config") as mock_config:
            mock_config.redis_url = "redis://localhost"
            with patch(
                "seer.services.collaboration.WorkflowLockService"
            ) as mock_lock_cls:
                generator = stream_org_events_sse(
                    organization_id=12,
                    user=mock_user,
                    tab_id=None,  # No tab_id
                )
                await anext(generator)
                await generator.aclose()

    # Lock service should not be instantiated when tab_id is not provided
    mock_lock_cls.assert_not_called()
