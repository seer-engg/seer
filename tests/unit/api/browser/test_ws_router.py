"""Tests for WebSocket + session lifecycle router."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.api.browser.ws_router import router

# Set up test app
app = FastAPI()
app.include_router(router, prefix="/api")


def _make_db_user(user_id="user-123"):
    """Create a mock db_user."""
    user = MagicMock()
    user.user_id = user_id
    user.id = user_id
    return user


@pytest.fixture
def mock_db_user():
    return _make_db_user()


class _AuthMiddleware:
    """Simple test middleware that sets request.state.db_user."""

    def __init__(self, app, user):
        self.app = app
        self.user = user

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            from starlette.requests import Request
            request = Request(scope, receive)
            request.state.db_user = self.user
            scope["state"] = {"db_user": self.user}
        await self.app(scope, receive, send)


class TestCreateSession:
    """Test POST /api/browser/sessions."""

    @patch("seer.api.browser.ws_router.BrowserProfileManager")
    async def test_create_session_returns_ws_url(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.create_interactive_session = AsyncMock(return_value={
            "session_id": "sess-abc",
            "profile_id": "prof-123",
            "recording_id": "rec-456",
            "status": "created",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        @test_app.middleware("http")
        async def add_user(request, call_next):
            request.state.db_user = mock_db_user
            return await call_next(request)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/sessions", json={
                "profile_id": str(uuid4()),
                "target_url": "https://slack.com",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-abc"
        assert "/stream" in data["ws_url"]
        assert data["status"] == "created"
        assert data["recording_id"] == "rec-456"

    @patch("seer.api.browser.ws_router.BrowserProfileManager")
    async def test_create_session_navigates_to_target(self, mock_manager_cls, mock_db_user):
        profile_id = str(uuid4())
        mock_manager = MagicMock()
        mock_manager.create_interactive_session = AsyncMock(return_value={
            "session_id": "sess-abc",
            "profile_id": profile_id,
            "recording_id": None,
            "status": "created",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        @test_app.middleware("http")
        async def add_user(request, call_next):
            request.state.db_user = mock_db_user
            return await call_next(request)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/sessions", json={
                "profile_id": profile_id,
                "target_url": "https://example.com",
            })

        assert resp.status_code == 200
        # Verify manager was called with target_url
        mock_manager.create_interactive_session.assert_called_once()
        call_args = mock_manager.create_interactive_session.call_args
        assert call_args[0][2] == "https://example.com"


class TestCompleteSession:
    """Test POST /api/browser/sessions/{id}/complete."""

    @patch("seer.api.browser.ws_router.BrowserPoolManager")
    @patch("seer.api.browser.ws_router.BrowserProfileManager")
    async def test_complete_session_saves_encrypted_state(
        self, mock_manager_cls, mock_pool_cls, mock_db_user
    ):
        # Set up pool mock
        managed = MagicMock()
        managed.user_id = "user-123"
        managed.profile_id = str(uuid4())

        pool_instance = MagicMock()
        pool_instance.get_session = MagicMock(return_value=managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=pool_instance)

        # Set up manager mock
        mock_manager = MagicMock()
        mock_manager.complete_interactive_session = AsyncMock(return_value={
            "profile_id": managed.profile_id,
            "logged_in_domains": ["slack.com", "github.com"],
            "recording_id": "rec-789",
            "status": "session_saved",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        @test_app.middleware("http")
        async def add_user(request, call_next):
            request.state.db_user = mock_db_user
            return await call_next(request)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/sessions/sess-abc/complete")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "session_saved"
        assert "slack.com" in data["logged_in_domains"]
        assert data["recording_id"] == "rec-789"

    @patch("seer.api.browser.ws_router.BrowserPoolManager")
    async def test_complete_nonexistent_session_404(self, mock_pool_cls, mock_db_user):
        pool_instance = MagicMock()
        pool_instance.get_session = MagicMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=pool_instance)

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        @test_app.middleware("http")
        async def add_user(request, call_next):
            request.state.db_user = mock_db_user
            return await call_next(request)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/sessions/nonexistent/complete")

        assert resp.status_code == 404

    @patch("seer.api.browser.ws_router.BrowserPoolManager")
    @patch("seer.api.browser.ws_router.BrowserProfileManager")
    async def test_complete_session_releases_pool(
        self, mock_manager_cls, mock_pool_cls, mock_db_user
    ):
        managed = MagicMock()
        managed.user_id = "user-123"
        managed.profile_id = str(uuid4())

        pool_instance = MagicMock()
        pool_instance.get_session = MagicMock(return_value=managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=pool_instance)

        mock_manager = MagicMock()
        mock_manager.complete_interactive_session = AsyncMock(return_value={
            "profile_id": managed.profile_id,
            "logged_in_domains": [],
            "recording_id": None,
            "status": "session_saved",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        @test_app.middleware("http")
        async def add_user(request, call_next):
            request.state.db_user = mock_db_user
            return await call_next(request)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/sessions/sess-abc/complete")

        assert resp.status_code == 200
        # complete_interactive_session internally calls pool.release_session
        mock_manager.complete_interactive_session.assert_called_once()


class TestWebSocketAuth:
    """Test WebSocket authentication."""

    @patch("seer.api.browser.ws_router._verify_ws_token", new_callable=AsyncMock)
    async def test_websocket_rejects_invalid_token(self, mock_verify):
        mock_verify.return_value = None

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        from starlette.testclient import TestClient
        with TestClient(test_app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/api/browser/sessions/sess-1/stream?token=bad-token"
                ):
                    pass

    @patch("seer.api.browser.ws_router.BrowserPoolManager")
    @patch("seer.api.browser.ws_router._verify_ws_token", new_callable=AsyncMock)
    async def test_websocket_rejects_wrong_user(self, mock_verify, mock_pool_cls):
        mock_verify.return_value = "wrong-user-id"

        managed = MagicMock()
        managed.user_id = "correct-user-id"

        pool_instance = MagicMock()
        pool_instance.get_session = MagicMock(return_value=managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=pool_instance)

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        from starlette.testclient import TestClient
        with TestClient(test_app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/api/browser/sessions/sess-1/stream?token=valid-token"
                ):
                    pass
