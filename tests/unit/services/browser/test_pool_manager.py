"""Tests for BrowserPoolManager - pool lifecycle, concurrency, reaper, shutdown."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.browser.pool_manager import BrowserPoolManager, ManagedSession


@pytest.fixture(autouse=True)
async def reset_singleton():
    """Ensure singleton is reset between tests."""
    BrowserPoolManager._instance = None
    yield
    if BrowserPoolManager._instance is not None:
        await BrowserPoolManager.shutdown_instance()


@pytest.fixture
def mock_browser_session():
    """Create a mock BrowserSession."""
    session = MagicMock()
    session.start = AsyncMock()
    session.stop = AsyncMock()
    session.kill = AsyncMock()
    session.export_storage_state = AsyncMock(return_value={
        "cookies": [{"name": "test", "value": "123", "domain": ".example.com"}],
        "origins": [],
    })
    return session


class TestManagedSession:
    """Test ManagedSession dataclass."""

    def test_not_expired(self, mock_browser_session):
        ms = ManagedSession(
            id="test-1",
            session=mock_browser_session,
            user_id="user-1",
            profile_id=None,
            session_type="workflow",
            timeout=300,
        )
        assert ms.is_expired is False

    def test_expired(self, mock_browser_session):
        ms = ManagedSession(
            id="test-1",
            session=mock_browser_session,
            user_id="user-1",
            profile_id=None,
            session_type="workflow",
            created_at=0,  # Way in the past
            timeout=1,
        )
        assert ms.is_expired is True


class TestPoolCreateRelease:
    """Test session creation and release."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_create_session(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=3)
        managed = await pool.create_session(
            user_id="user-1",
            profile_id="profile-1",
            session_type="workflow",
            storage_state={"cookies": []},
            timeout=120,
        )

        assert managed.user_id == "user-1"
        assert managed.profile_id == "profile-1"
        assert managed.session_type == "workflow"
        assert managed.timeout == 120
        assert managed.session == mock_browser_session
        mock_browser_session.start.assert_called_once()

        status = pool.health_status()
        assert status["active_sessions"] == 1
        assert status["available_slots"] == 2

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_release_returns_storage_state(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=3)
        managed = await pool.create_session(user_id="user-1")

        state = await pool.release_session(managed.id)
        assert state is not None
        assert "cookies" in state
        mock_browser_session.export_storage_state.assert_called_once()
        mock_browser_session.stop.assert_called_once()

        status = pool.health_status()
        assert status["active_sessions"] == 0

        await pool.shutdown()

    async def test_release_nonexistent(self):
        pool = BrowserPoolManager(max_concurrent=3)
        result = await pool.release_session("nonexistent-id")
        assert result is None
        await pool.shutdown()


class TestConcurrencyControl:
    """Test semaphore-based concurrency limiting."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_semaphore_blocks_at_max(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=2)

        s1 = await pool.create_session(user_id="u1")
        s2 = await pool.create_session(user_id="u2")

        # Third creation should block (we test with a short timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                pool.create_session(user_id="u3"),
                timeout=0.1,
            )

        # Release one, then third should succeed
        await pool.release_session(s1.id)
        s3 = await pool.create_session(user_id="u3")
        assert s3 is not None

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_semaphore_released_on_create_failure(self, mock_profile_cls, mock_session_cls):
        mock_session = MagicMock()
        mock_session.start = AsyncMock(side_effect=RuntimeError("Launch failed"))
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=1)

        with pytest.raises(RuntimeError, match="Launch failed"):
            await pool.create_session(user_id="u1")

        # Semaphore should be released, so another create should not block
        mock_session2 = MagicMock()
        mock_session2.start = AsyncMock()
        mock_session_cls.return_value = mock_session2

        managed = await asyncio.wait_for(
            pool.create_session(user_id="u2"),
            timeout=0.5,
        )
        assert managed is not None

        await pool.shutdown()


class TestShutdown:
    """Test pool shutdown."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_shutdown_closes_all(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=5)
        await pool.create_session(user_id="u1")
        await pool.create_session(user_id="u2")

        assert pool.health_status()["active_sessions"] == 2

        await pool.shutdown()
        assert pool.health_status()["active_sessions"] == 0


class TestHealthStatus:
    """Test health status reporting."""

    async def test_empty_pool(self):
        pool = BrowserPoolManager(max_concurrent=5)
        status = pool.health_status()
        assert status["max_concurrent"] == 5
        assert status["active_sessions"] == 0
        assert status["available_slots"] == 5
        assert status["sessions"] == []
        await pool.shutdown()


class TestSingleton:
    """Test singleton behavior."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_get_instance_returns_same(self, mock_profile_cls, mock_session_cls):
        pool1 = await BrowserPoolManager.get_instance()
        pool2 = await BrowserPoolManager.get_instance()
        assert pool1 is pool2

    async def test_shutdown_instance_clears(self):
        pool1 = await BrowserPoolManager.get_instance()
        await BrowserPoolManager.shutdown_instance()
        assert BrowserPoolManager._instance is None

        pool2 = await BrowserPoolManager.get_instance()
        assert pool2 is not pool1
