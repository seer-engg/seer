"""
Unit tests for MCP authentication middleware.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route

from seer.auth.clerk_verifier import VerifiedClerkToken
from seer.mcp.auth import (
    MCPAuthMiddleware,
    extract_bearer_token,
    www_authenticate_response,
    get_mcp_authenticated_user,
    mcp_authenticated_user,
)


class TestExtractBearerToken:
    """Tests for extract_bearer_token function."""

    def test_extracts_valid_bearer_token(self):
        """Test extracting a valid Bearer token from Authorization header."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer test-token-123"}

        result = extract_bearer_token(request)
        assert result == "test-token-123"

    def test_returns_none_for_missing_header(self):
        """Test returns None when Authorization header is missing."""
        request = MagicMock(spec=Request)
        request.headers = {}

        result = extract_bearer_token(request)
        assert result is None

    def test_returns_none_for_non_bearer_auth(self):
        """Test returns None for non-Bearer authorization."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}

        result = extract_bearer_token(request)
        assert result is None

    def test_returns_none_for_empty_bearer(self):
        """Test returns None when Bearer token is empty."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer "}

        result = extract_bearer_token(request)
        assert result is None

    def test_strips_whitespace(self):
        """Test that whitespace is stripped from token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer   token-with-spaces   "}

        result = extract_bearer_token(request)
        assert result == "token-with-spaces"


class TestWwwAuthenticateResponse:
    """Tests for www_authenticate_response function."""

    def test_returns_401_status(self):
        """Test that response has 401 status code."""
        response = www_authenticate_response("Test error")
        assert response.status_code == 401

    def test_includes_www_authenticate_header(self):
        """Test that WWW-Authenticate header is included."""
        response = www_authenticate_response("Token expired")
        assert "WWW-Authenticate" in response.headers
        assert 'Bearer realm="seer"' in response.headers["WWW-Authenticate"]
        assert "Token expired" in response.headers["WWW-Authenticate"]

    def test_includes_mcp_meta(self):
        """Test that _meta with mcp/www_authenticate is included in body."""
        response = www_authenticate_response("Test error")
        # The body contains JSON with _meta
        body = response.body.decode()
        assert "mcp/www_authenticate" in body
        assert "authentication_required" in body

    def test_custom_error_code(self):
        """Test custom error code in response."""
        response = www_authenticate_response("Missing scope", error="insufficient_scope")
        header = response.headers["WWW-Authenticate"]
        assert 'error="insufficient_scope"' in header


class TestMCPAuthMiddleware:
    """Tests for MCPAuthMiddleware."""

    @pytest.fixture
    def mock_verifier(self):
        """Create a mock ClerkJWTVerifier."""
        verifier = MagicMock()
        verifier.verify_token_with_error.return_value = (
            VerifiedClerkToken(
                user_id="user_123",
                email="test@example.com",
                first_name="Test",
                last_name="User",
                claims={"sub": "user_123"},
            ),
            None,
        )
        return verifier

    @pytest.fixture
    def app_with_middleware(self, mock_verifier):
        """Create a test app with MCPAuthMiddleware."""
        async def endpoint(request):
            # Return info about the authenticated user
            mcp_user = getattr(request.state, "mcp_user", None)
            if mcp_user:
                return Response(f"user_id={mcp_user.user_id}")
            return Response("no user")

        async def oauth_discovery(request):
            return Response("oauth metadata")

        app = Starlette(
            routes=[
                Route("/test", endpoint),
                Route("/.well-known/oauth-protected-resource", oauth_discovery),
            ],
        )

        # Add middleware
        app.add_middleware(MCPAuthMiddleware, verifier=mock_verifier)
        return app

    def test_allows_options_requests(self, app_with_middleware):
        """Test that OPTIONS requests bypass auth."""
        client = TestClient(app_with_middleware)
        response = client.options("/test")
        # Should not return 401 (middleware allows through)
        assert response.status_code != 401

    def test_allows_oauth_discovery_endpoint(self, app_with_middleware):
        """Test that OAuth discovery endpoint is public."""
        client = TestClient(app_with_middleware)
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        assert response.text == "oauth metadata"

    def test_rejects_missing_auth_header(self, app_with_middleware):
        """Test that requests without auth header get 401."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_rejects_invalid_token(self, app_with_middleware, mock_verifier):
        """Test that invalid tokens get 401."""
        mock_verifier.verify_token_with_error.return_value = (None, "Token expired")

        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer invalid-token"})
        assert response.status_code == 401
        assert "Token expired" in response.headers["WWW-Authenticate"]

    def test_allows_valid_token(self, app_with_middleware, mock_verifier):
        """Test that valid tokens are accepted."""
        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer valid-token"})
        assert response.status_code == 200
        assert "user_id=user_123" in response.text

    def test_sets_request_state(self, app_with_middleware, mock_verifier):
        """Test that authenticated user is set on request.state."""
        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer valid-token"})
        assert response.status_code == 200
        assert "user_123" in response.text


class TestGetMCPAuthenticatedUser:
    """Tests for get_mcp_authenticated_user context variable."""

    def test_returns_none_by_default(self):
        """Test that None is returned when no user is set."""
        result = get_mcp_authenticated_user()
        assert result is None

    def test_returns_set_user(self):
        """Test that set user is returned."""
        token = VerifiedClerkToken(
            user_id="ctx_user",
            email="ctx@example.com",
            first_name=None,
            last_name=None,
            claims={},
        )

        # Set the context variable
        ctx_token = mcp_authenticated_user.set(token)
        try:
            result = get_mcp_authenticated_user()
            assert result is not None
            assert result.user_id == "ctx_user"
        finally:
            # Reset the context variable
            mcp_authenticated_user.reset(ctx_token)

    def test_context_isolation(self):
        """Test that context is properly isolated after reset."""
        token = VerifiedClerkToken(
            user_id="temp_user",
            email=None,
            first_name=None,
            last_name=None,
            claims={},
        )

        ctx_token = mcp_authenticated_user.set(token)
        mcp_authenticated_user.reset(ctx_token)

        # Should be None after reset
        result = get_mcp_authenticated_user()
        assert result is None
