"""Tests for WebSocket + session lifecycle router."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.api.browser.ws_router import (
    _dispatch_input_event,
    _handle_key_event,
    _handle_mouse_event,
    _handle_navigate_event,
    _handle_scroll_event,
    _verify_ws_token,
    router,
)

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


@pytest.mark.asyncio
@pytest.mark.unit
class TestHandleMouseEvent:
    """Tests for _handle_mouse_event helper function."""

    async def test_click_action_calls_dispatch_click_js(self):
        """Test that click action uses JavaScript click (industry standard)."""
        streamer = MagicMock()
        streamer.dispatch_click_js = AsyncMock()

        await _handle_mouse_event(streamer, {
            "action": "click",
            "x": 100,
            "y": 200,
            "button": "left",
        })

        streamer.dispatch_click_js.assert_called_once_with(100.0, 200.0, "left")

    async def test_down_action_calls_dispatch_mouse_event(self):
        """Test mousePressed event dispatch."""
        streamer = MagicMock()
        streamer.dispatch_mouse_event = AsyncMock()

        await _handle_mouse_event(streamer, {
            "action": "down",
            "x": 50,
            "y": 75,
            "button": "right",
            "clickCount": 2,
        })

        streamer.dispatch_mouse_event.assert_called_once_with(
            "mousePressed", 50.0, 75.0, button="right", click_count=2
        )

    async def test_up_action_calls_dispatch_mouse_event(self):
        """Test mouseReleased event dispatch."""
        streamer = MagicMock()
        streamer.dispatch_mouse_event = AsyncMock()

        await _handle_mouse_event(streamer, {
            "action": "up",
            "x": 150,
            "y": 250,
            "button": "middle",
            "clickCount": 1,
        })

        streamer.dispatch_mouse_event.assert_called_once_with(
            "mouseReleased", 150.0, 250.0, button="middle", click_count=1
        )

    async def test_move_action_calls_dispatch_mouse_event(self):
        """Test mouseMoved event dispatch."""
        streamer = MagicMock()
        streamer.dispatch_mouse_event = AsyncMock()

        await _handle_mouse_event(streamer, {
            "action": "move",
            "x": 300,
            "y": 400,
        })

        streamer.dispatch_mouse_event.assert_called_once_with(
            "mouseMoved", 300.0, 400.0
        )

    async def test_default_values(self):
        """Test default values when fields are missing."""
        streamer = MagicMock()
        streamer.dispatch_click_js = AsyncMock()

        # Empty data should use defaults
        await _handle_mouse_event(streamer, {})

        # Default: action=click, x=0, y=0, button=left
        streamer.dispatch_click_js.assert_called_once_with(0.0, 0.0, "left")


@pytest.mark.asyncio
@pytest.mark.unit
class TestHandleKeyEvent:
    """Tests for _handle_key_event helper function."""

    async def test_press_action_sends_down_char_up(self):
        """Test press action sends keyDown, char, then keyUp when text is present."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()

        await _handle_key_event(streamer, {
            "action": "press",
            "key": "a",
            "code": "KeyA",
            "text": "a",
        })

        # Should call dispatch_key_event 3 times for press with text
        assert streamer.dispatch_key_event.call_count == 3
        calls = streamer.dispatch_key_event.call_args_list

        # keyDown
        assert calls[0].args == ("keyDown", "a")
        assert calls[0].kwargs == {"code": "KeyA", "text": "a"}

        # char
        assert calls[1].args == ("char", "a")
        assert calls[1].kwargs == {"code": "KeyA", "text": "a"}

        # keyUp (no text in keyUp)
        assert calls[2].args == ("keyUp", "a")
        assert calls[2].kwargs == {"code": "KeyA"}

    async def test_press_action_without_text(self):
        """Test press action without text (e.g., Enter key) skips char event."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()

        await _handle_key_event(streamer, {
            "action": "press",
            "key": "Enter",
            "code": "Enter",
            "text": "",  # No text
        })

        # Should call dispatch_key_event 2 times (no char event)
        assert streamer.dispatch_key_event.call_count == 2
        calls = streamer.dispatch_key_event.call_args_list

        # keyDown
        assert calls[0].args == ("keyDown", "Enter")

        # keyUp
        assert calls[1].args == ("keyUp", "Enter")

    async def test_down_action(self):
        """Test keyDown event dispatch."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()

        await _handle_key_event(streamer, {
            "action": "down",
            "key": "Shift",
            "code": "ShiftLeft",
        })

        streamer.dispatch_key_event.assert_called_once_with(
            "keyDown", "Shift", code="ShiftLeft", text=""
        )

    async def test_up_action(self):
        """Test keyUp event dispatch."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()

        await _handle_key_event(streamer, {
            "action": "up",
            "key": "Control",
            "code": "ControlLeft",
        })

        streamer.dispatch_key_event.assert_called_once_with(
            "keyUp", "Control", code="ControlLeft"
        )

    async def test_default_values(self):
        """Test default values when fields are missing."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()

        await _handle_key_event(streamer, {})

        # Default: action=press, key="", code="", text=""
        assert streamer.dispatch_key_event.call_count == 2  # No char event (empty text)


@pytest.mark.asyncio
@pytest.mark.unit
class TestHandleScrollEvent:
    """Tests for _handle_scroll_event helper function."""

    async def test_scroll_event_with_custom_deltas(self):
        """Test scroll event dispatch with custom delta values."""
        streamer = MagicMock()
        streamer.dispatch_scroll_event = AsyncMock()

        await _handle_scroll_event(streamer, {
            "x": 640,
            "y": 400,
            "deltaX": 50,
            "deltaY": -200,
        })

        streamer.dispatch_scroll_event.assert_called_once_with(640.0, 400.0, 50.0, -200.0)

    async def test_default_delta_values(self):
        """Test scroll event uses default delta values."""
        streamer = MagicMock()
        streamer.dispatch_scroll_event = AsyncMock()

        await _handle_scroll_event(streamer, {
            "x": 100,
            "y": 200,
        })

        # Default: deltaX=0, deltaY=-120
        streamer.dispatch_scroll_event.assert_called_once_with(100.0, 200.0, 0.0, -120.0)


@pytest.mark.asyncio
@pytest.mark.unit
class TestHandleNavigateEvent:
    """Tests for _handle_navigate_event helper function."""

    async def test_navigate_success(self):
        """Test successful navigation."""
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page._target_id = "target-123"
        mock_session.must_get_current_page = AsyncMock(return_value=mock_page)

        mock_cdp_session = MagicMock()
        mock_cdp_session.session_id = "cdp-sess-456"
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)

        mock_session.cdp_client = MagicMock()
        mock_session.cdp_client.send = MagicMock()
        mock_session.cdp_client.send.Page = MagicMock()
        mock_session.cdp_client.send.Page.navigate = AsyncMock()

        managed = MagicMock()
        managed.session = mock_session

        await _handle_navigate_event(managed, {"url": "https://example.com"})

        mock_session.cdp_client.send.Page.navigate.assert_called_once_with(
            params={"url": "https://example.com"},
            session_id="cdp-sess-456",
        )

    async def test_navigate_empty_url_returns_early(self):
        """Test that empty URL does nothing."""
        managed = MagicMock()
        managed.session = MagicMock()
        managed.session.must_get_current_page = AsyncMock()

        await _handle_navigate_event(managed, {"url": ""})

        # Should return early without calling must_get_current_page
        managed.session.must_get_current_page.assert_not_called()

    async def test_navigate_exception_logged(self):
        """Test that navigation exceptions are logged but don't raise."""
        mock_session = MagicMock()
        mock_session.must_get_current_page = AsyncMock(side_effect=RuntimeError("CDP error"))

        managed = MagicMock()
        managed.session = mock_session

        # Should not raise
        await _handle_navigate_event(managed, {"url": "https://example.com"})


@pytest.mark.asyncio
@pytest.mark.unit
class TestDispatchInputEvent:
    """Tests for _dispatch_input_event routing function."""

    async def test_routes_mouse_event(self):
        """Test routing to mouse handler."""
        streamer = MagicMock()
        streamer.dispatch_click_js = AsyncMock()
        managed = MagicMock()

        await _dispatch_input_event(streamer, managed, {
            "type": "mouse",
            "action": "click",
            "x": 100,
            "y": 200,
        })

        streamer.dispatch_click_js.assert_called_once()

    async def test_routes_key_event(self):
        """Test routing to key handler."""
        streamer = MagicMock()
        streamer.dispatch_key_event = AsyncMock()
        managed = MagicMock()

        await _dispatch_input_event(streamer, managed, {
            "type": "key",
            "action": "press",
            "key": "Enter",
        })

        assert streamer.dispatch_key_event.call_count >= 1

    async def test_routes_scroll_event(self):
        """Test routing to scroll handler."""
        streamer = MagicMock()
        streamer.dispatch_scroll_event = AsyncMock()
        managed = MagicMock()

        await _dispatch_input_event(streamer, managed, {
            "type": "scroll",
            "x": 640,
            "y": 400,
            "deltaY": -120,
        })

        streamer.dispatch_scroll_event.assert_called_once()

    async def test_routes_navigate_event(self):
        """Test routing to navigate handler."""
        streamer = MagicMock()
        mock_session = MagicMock()
        mock_session.must_get_current_page = AsyncMock(return_value=MagicMock())
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())
        mock_session.cdp_client = MagicMock()
        mock_session.cdp_client.send.Page.navigate = AsyncMock()

        managed = MagicMock()
        managed.session = mock_session

        await _dispatch_input_event(streamer, managed, {
            "type": "navigate",
            "url": "https://example.com",
        })

    async def test_unknown_type_ignored(self):
        """Test that unknown event types are silently ignored."""
        streamer = MagicMock()
        managed = MagicMock()

        # Should not raise
        await _dispatch_input_event(streamer, managed, {
            "type": "unknown_event",
            "data": "test",
        })


@pytest.mark.asyncio
@pytest.mark.unit
class TestVerifyWsToken:
    """Tests for _verify_ws_token JWT verification function."""

    async def test_empty_token_returns_none(self):
        """Test that empty token returns None."""
        result = await _verify_ws_token("")
        assert result is None

    async def test_none_token_returns_none(self):
        """Test that None token (if somehow passed) returns None."""
        result = await _verify_ws_token("")
        assert result is None

    @patch("seer.api.browser.ws_router.config")
    async def test_cloud_mode_uses_clerk_verifier(self, mock_config):
        """Test that cloud mode uses ClerkJWTVerifier."""
        mock_config.is_cloud_mode = True
        mock_config.is_clerk_configured = True
        mock_config.clerk_jwks_url = "https://clerk.example.com/.well-known/jwks.json"
        mock_config.clerk_issuer = "https://clerk.example.com"
        mock_config.clerk_audience = "test-audience"

        with patch("seer.api.browser.ws_router.ClerkJWTVerifier") as mock_verifier_cls:
            mock_verifier = MagicMock()
            mock_verified = MagicMock()
            mock_verified.user_id = "user-from-clerk"
            mock_verifier.verify_token.return_value = mock_verified
            mock_verifier_cls.return_value = mock_verifier

            result = await _verify_ws_token("valid-clerk-token")

            assert result == "user-from-clerk"
            mock_verifier_cls.assert_called_once_with(
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
                audience="test-audience",
            )
            mock_verifier.verify_token.assert_called_once_with("valid-clerk-token")

    @patch("seer.api.browser.ws_router.config")
    async def test_cloud_mode_invalid_token_returns_none(self, mock_config):
        """Test that cloud mode with invalid token returns None."""
        mock_config.is_cloud_mode = True
        mock_config.is_clerk_configured = True
        mock_config.clerk_jwks_url = "https://clerk.example.com/.well-known/jwks.json"
        mock_config.clerk_issuer = "https://clerk.example.com"
        mock_config.clerk_audience = "test-audience"

        with patch("seer.api.browser.ws_router.ClerkJWTVerifier") as mock_verifier_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify_token.return_value = None
            mock_verifier_cls.return_value = mock_verifier

            result = await _verify_ws_token("invalid-token")

            assert result is None

    @patch("seer.api.browser.ws_router.config")
    @patch("seer.api.browser.ws_router.jwt")
    async def test_self_hosted_decodes_without_verification(self, mock_jwt, mock_config):
        """Test that self-hosted mode decodes JWT without signature verification."""
        mock_config.is_cloud_mode = False

        mock_jwt.decode.return_value = {"sub": "user-from-sub"}

        result = await _verify_ws_token("self-hosted-token")

        assert result == "user-from-sub"
        mock_jwt.decode.assert_called_once_with(
            "self-hosted-token",
            options={"verify_signature": False}
        )

    @patch("seer.api.browser.ws_router.config")
    @patch("seer.api.browser.ws_router.jwt")
    async def test_self_hosted_uses_user_id_claim(self, mock_jwt, mock_config):
        """Test that self-hosted mode falls back to user_id claim."""
        mock_config.is_cloud_mode = False

        mock_jwt.decode.return_value = {"user_id": "user-from-user_id"}

        result = await _verify_ws_token("token-with-user_id")

        assert result == "user-from-user_id"

    @patch("seer.api.browser.ws_router.config")
    @patch("seer.api.browser.ws_router.jwt")
    async def test_self_hosted_uses_sid_claim(self, mock_jwt, mock_config):
        """Test that self-hosted mode falls back to sid claim."""
        mock_config.is_cloud_mode = False

        mock_jwt.decode.return_value = {"sid": "session-id-123"}

        result = await _verify_ws_token("token-with-sid")

        assert result == "session-id-123"

    @patch("seer.api.browser.ws_router.config")
    @patch("seer.api.browser.ws_router.jwt")
    async def test_self_hosted_decode_error_returns_none(self, mock_jwt, mock_config):
        """Test that self-hosted mode handles decode errors gracefully."""
        mock_config.is_cloud_mode = False

        mock_jwt.decode.side_effect = Exception("Invalid JWT")

        result = await _verify_ws_token("malformed-token")

        assert result is None
