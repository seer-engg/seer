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
