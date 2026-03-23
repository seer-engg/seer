"""Unit tests for workflow collaboration locks."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from seer.services.collaboration.lock_service import LOCK_TTL_SECONDS, WorkflowLockService


@pytest.fixture
def lock_service():
    return WorkflowLockService()


@pytest.fixture
def lock_user():
    return SimpleNamespace(
        id=1,
        user_id="user_1",
        first_name="Lock",
        last_name="Owner",
        email="lock@example.com",
    )


@pytest.fixture
def other_user():
    return SimpleNamespace(
        id=2,
        user_id="user_2",
        first_name="Other",
        last_name="User",
        email="other@example.com",
    )


@pytest.mark.asyncio
async def test_acquire_lock_success(lock_service, lock_user):
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=True)

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        acquired, lock = await lock_service.acquire_lock(
            organization_id=12,
            workflow_id="wf_123",
            user=lock_user,
            tab_id="tab-1",
        )

    assert acquired is True
    assert lock.workflow_id == "wf_123"
    assert lock.holder_db_user_id == 1
    assert lock.tab_id == "tab-1"
    stored_json = mock_redis.set.call_args.args[1]
    stored_payload = json.loads(stored_json)
    assert stored_payload["holder_name"] == "Lock Owner"
    assert mock_redis.set.call_args.kwargs["ex"] == LOCK_TTL_SECONDS
    assert mock_redis.set.call_args.kwargs["nx"] is True


@pytest.mark.asyncio
async def test_acquire_lock_conflict_returns_existing(lock_service, lock_user):
    existing_lock = lock_service._build_lock(organization_id=12, workflow_id="wf_123", user=lock_user, tab_id="tab-1")
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=False)
    mock_redis.get = AsyncMock(return_value=existing_lock.model_dump_json())

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        acquired, lock = await lock_service.acquire_lock(
            organization_id=12,
            workflow_id="wf_123",
            user=lock_user,
            tab_id="tab-2",
        )

    assert acquired is False
    assert lock.tab_id == "tab-1"


@pytest.mark.asyncio
async def test_heartbeat_lock_requires_holder(lock_service, lock_user, other_user):
    existing_lock = lock_service._build_lock(organization_id=12, workflow_id="wf_123", user=lock_user, tab_id="tab-1")
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=existing_lock.model_dump_json())
    mock_redis.set = AsyncMock()

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        refreshed = await lock_service.heartbeat_lock(
            organization_id=12,
            workflow_id="wf_123",
            user=other_user,
            tab_id="tab-1",
        )

    assert refreshed is None
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_release_lock_requires_holder(lock_service, lock_user, other_user):
    existing_lock = lock_service._build_lock(organization_id=12, workflow_id="wf_123", user=lock_user, tab_id="tab-1")
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=existing_lock.model_dump_json())
    mock_redis.delete = AsyncMock()

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        released = await lock_service.release_lock(
            organization_id=12,
            workflow_id="wf_123",
            user=other_user,
            tab_id="tab-1",
        )

    assert released is None
    mock_redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_lock_with_sse_connection_id(lock_service, lock_user):
    """Test that acquire_lock stores sse_connection_id in lock metadata."""
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=True)

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        acquired, lock = await lock_service.acquire_lock(
            organization_id=12,
            workflow_id="wf_123",
            user=lock_user,
            tab_id="tab-1",
            sse_connection_id="12:user_1:tab-1",
        )

    assert acquired is True
    assert lock.sse_connection_id == "12:user_1:tab-1"
    stored_json = mock_redis.set.call_args.args[1]
    stored_payload = json.loads(stored_json)
    assert stored_payload["sse_connection_id"] == "12:user_1:tab-1"


@pytest.mark.asyncio
async def test_release_locks_for_connection(lock_service, lock_user):
    """Test that release_locks_for_connection releases all locks held by an SSE connection."""
    connection_id = "12:user_1:tab-1"
    lock1 = lock_service._build_lock(
        organization_id=12,
        workflow_id="wf_123",
        user=lock_user,
        tab_id="tab-1",
        sse_connection_id=connection_id,
    )
    lock2 = lock_service._build_lock(
        organization_id=12,
        workflow_id="wf_456",
        user=lock_user,
        tab_id="tab-1",
        sse_connection_id=connection_id,
    )
    lock3_other = lock_service._build_lock(
        organization_id=12,
        workflow_id="wf_789",
        user=lock_user,
        tab_id="tab-2",
        sse_connection_id="12:user_1:tab-2",  # Different connection
    )

    async def mock_scan_iter(match=None):
        for key in [
            "org:12:workflow:wf_123:lock",
            "org:12:workflow:wf_456:lock",
            "org:12:workflow:wf_789:lock",
        ]:
            yield key

    async def mock_get(key):
        if "wf_123" in key:
            return lock1.model_dump_json()
        if "wf_456" in key:
            return lock2.model_dump_json()
        if "wf_789" in key:
            return lock3_other.model_dump_json()
        return None

    mock_redis = AsyncMock()
    mock_redis.scan_iter = mock_scan_iter
    mock_redis.get = mock_get
    mock_redis.delete = AsyncMock()

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        released = await lock_service.release_locks_for_connection(12, connection_id)

    assert len(released) == 2
    workflow_ids = [wf_id for wf_id, _ in released]
    assert "wf_123" in workflow_ids
    assert "wf_456" in workflow_ids
    assert "wf_789" not in workflow_ids  # Different connection, not released

    # Verify delete was called for the two matching locks
    assert mock_redis.delete.call_count == 2


@pytest.mark.asyncio
async def test_release_locks_for_connection_empty_when_no_matches(lock_service, lock_user):
    """Test that release_locks_for_connection returns empty list when no locks match."""
    connection_id = "12:user_1:tab-1"
    other_lock = lock_service._build_lock(
        organization_id=12,
        workflow_id="wf_123",
        user=lock_user,
        tab_id="tab-2",
        sse_connection_id="12:user_1:tab-2",  # Different connection
    )

    async def mock_scan_iter(match=None):
        yield "org:12:workflow:wf_123:lock"

    mock_redis = AsyncMock()
    mock_redis.scan_iter = mock_scan_iter
    mock_redis.get = AsyncMock(return_value=other_lock.model_dump_json())
    mock_redis.delete = AsyncMock()

    with patch.object(lock_service, "_get_redis", return_value=mock_redis):
        released = await lock_service.release_locks_for_connection(12, connection_id)

    assert len(released) == 0
    mock_redis.delete.assert_not_called()
