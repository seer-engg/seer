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


class TestSessionReaper:
    """Test session reaper background task."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_reaper_removes_expired_sessions(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        """Test that reaper removes expired sessions."""
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=5)
        pool._reaper_interval = 0.05  # Speed up for test

        # Create session with very short timeout
        managed = await pool.create_session(
            user_id="user-1",
            timeout=0,  # Expires immediately
        )

        # Manually mark as expired by setting created_at in past
        managed.created_at = 0
        managed.timeout = 0

        # Start reaper
        pool._start_reaper()

        # Wait for reaper to run
        await asyncio.sleep(0.1)

        # Session should be removed
        assert managed.id not in pool._sessions

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_reaper_handles_release_exception(self, mock_profile_cls, mock_session_cls):
        """Test that reaper handles exceptions during session release."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock(side_effect=RuntimeError("Stop failed"))
        mock_session.export_storage_state = AsyncMock(side_effect=RuntimeError("Export failed"))
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)
        pool._reaper_interval = 0.05

        # Create session that will fail on release
        managed = await pool.create_session(user_id="user-1", timeout=0)
        managed.created_at = 0
        managed.timeout = 0

        pool._start_reaper()

        # Wait for reaper to run
        await asyncio.sleep(0.1)

        # Session should still be removed despite errors
        assert managed.id not in pool._sessions

        await pool.shutdown()


class TestGetSession:
    """Test get_session lookup method."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_get_session_returns_managed_session(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        """Test get_session returns correct ManagedSession."""
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=5)
        created = await pool.create_session(user_id="user-1", profile_id="profile-1")

        found = pool.get_session(created.id)

        assert found is created
        assert found.user_id == "user-1"
        assert found.profile_id == "profile-1"

        await pool.shutdown()

    async def test_get_session_returns_none_for_missing(self):
        """Test get_session returns None for non-existent session."""
        pool = BrowserPoolManager(max_concurrent=5)

        result = pool.get_session("nonexistent-id")

        assert result is None

        await pool.shutdown()


class TestStealthMode:
    """Test stealth mode profile creation."""

    @patch("seer.services.browser.pool_manager.get_stealth_profile_kwargs")
    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_stealth_mode_uses_stealth_profile(
        self, mock_profile_cls, mock_session_cls, mock_stealth_kwargs, mock_browser_session
    ):
        """Test that stealth mode uses stealth profile kwargs."""
        mock_session_cls.return_value = mock_browser_session
        mock_stealth_kwargs.return_value = {
            "headless_mode": "new",
            "extra_chromium_args": ["--disable-blink-features=AutomationControlled"],
        }

        pool = BrowserPoolManager(max_concurrent=5)
        await pool.create_session(
            user_id="user-1",
            stealth_mode=True,
            storage_state={"cookies": []},
        )

        # Verify stealth profile kwargs were used
        mock_stealth_kwargs.assert_called_once()
        mock_profile_cls.assert_called_once()
        call_kwargs = mock_profile_cls.call_args.kwargs
        assert call_kwargs.get("headless_mode") == "new"
        assert call_kwargs.get("keep_alive") is True

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_standard_mode_uses_headless_true(
        self, mock_profile_cls, mock_session_cls, mock_browser_session
    ):
        """Test that standard mode uses headless=True."""
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=5)
        await pool.create_session(
            user_id="user-1",
            stealth_mode=False,
            storage_state={"cookies": []},
        )

        mock_profile_cls.assert_called_once()
        call_kwargs = mock_profile_cls.call_args.kwargs
        assert call_kwargs.get("headless") is True
        assert call_kwargs.get("keep_alive") is True

        await pool.shutdown()


class TestCDPCookieApplication:
    """Test CDP cookie application after session start."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_cookies_applied_via_cdp(self, mock_profile_cls, mock_session_cls):
        """Test that cookies are applied via CDP after session start."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.export_storage_state = AsyncMock(return_value={})
        mock_session._cdp_set_cookies = AsyncMock()
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)
        cookies = [
            {"name": "session", "value": "abc123", "domain": ".example.com"},
            {"name": "auth", "value": "xyz789", "domain": ".example.com"},
        ]
        await pool.create_session(
            user_id="user-1",
            storage_state={"cookies": cookies},
        )

        mock_session._cdp_set_cookies.assert_called_once_with(cookies)

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_cdp_cookie_application_failure_logged(self, mock_profile_cls, mock_session_cls):
        """Test that CDP cookie application failure is logged but doesn't crash."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.export_storage_state = AsyncMock(return_value={})
        mock_session._cdp_set_cookies = AsyncMock(side_effect=RuntimeError("CDP error"))
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)

        # Should not raise
        managed = await pool.create_session(
            user_id="user-1",
            storage_state={"cookies": [{"name": "test", "value": "123"}]},
        )

        assert managed is not None

        await pool.shutdown()


class TestShutdownErrors:
    """Test shutdown error handling."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_shutdown_handles_session_errors(self, mock_profile_cls, mock_session_cls):
        """Test that shutdown handles errors from individual sessions."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock(side_effect=RuntimeError("Session cleanup failed"))
        mock_session.export_storage_state = AsyncMock(side_effect=RuntimeError("Export failed"))
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)
        await pool.create_session(user_id="u1")
        await pool.create_session(user_id="u2")

        # Should not raise despite session errors
        await pool.shutdown()

        assert pool.health_status()["active_sessions"] == 0

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_shutdown_cancels_reaper(self, mock_profile_cls, mock_session_cls, mock_browser_session):
        """Test that shutdown cancels the reaper task."""
        mock_session_cls.return_value = mock_browser_session

        pool = BrowserPoolManager(max_concurrent=5)
        pool._start_reaper()

        assert pool._reaper_task is not None
        assert not pool._reaper_task.done()

        await pool.shutdown()

        assert pool._reaper_task.done()


class TestReleaseSessionEdgeCases:
    """Test release_session edge cases."""

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_release_handles_export_failure(self, mock_profile_cls, mock_session_cls):
        """Test that release handles export_storage_state failure."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.export_storage_state = AsyncMock(side_effect=RuntimeError("Export failed"))
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)
        managed = await pool.create_session(user_id="u1")

        # Should not raise, returns None for storage_state
        result = await pool.release_session(managed.id)

        assert result is None
        assert pool.health_status()["active_sessions"] == 0

        await pool.shutdown()

    @patch("seer.services.browser.pool_manager.BrowserSession")
    @patch("seer.services.browser.pool_manager.BrowserUseProfile")
    async def test_release_handles_stop_failure(self, mock_profile_cls, mock_session_cls):
        """Test that release handles session.stop failure."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock(side_effect=RuntimeError("Stop failed"))
        mock_session.export_storage_state = AsyncMock(return_value={"cookies": []})
        mock_session_cls.return_value = mock_session

        pool = BrowserPoolManager(max_concurrent=5)
        managed = await pool.create_session(user_id="u1")

        # Should not raise, still returns storage_state
        result = await pool.release_session(managed.id)

        assert result == {"cookies": []}
        assert pool.health_status()["active_sessions"] == 0

        await pool.shutdown()
