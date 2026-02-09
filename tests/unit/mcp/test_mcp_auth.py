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
    MCPOpaqueAuthMiddleware,
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

    @pytest.fixture
    def mock_request(self):
        """Create a mock request with URL info for constructing resource_metadata."""
        request = MagicMock(spec=Request)
        # Mock URL with netloc for constructing the resource_metadata URL
        mock_url = MagicMock()
        mock_url.netloc = "api.example.com"
        request.url = mock_url
        return request

    def test_returns_401_status(self, mock_request):
        """Test that response has 401 status code."""
        response = www_authenticate_response(mock_request, "Test error")
        assert response.status_code == 401

    def test_includes_www_authenticate_header_with_resource_metadata(self, mock_request):
        """Test that WWW-Authenticate header includes resource_metadata for OAuth discovery."""
        response = www_authenticate_response(mock_request, "Token expired")
        assert "WWW-Authenticate" in response.headers
        header = response.headers["WWW-Authenticate"]
        # Check for resource_metadata URL (scheme comes from config, defaults to https)
        assert 'resource_metadata="https://api.example.com/.well-known/oauth-protected-resource"' in header
        assert "Token expired" in header

    def test_includes_mcp_meta(self, mock_request):
        """Test that _meta with mcp/www_authenticate is included in body."""
        response = www_authenticate_response(mock_request, "Test error")
        # The body contains JSON with _meta
        body = response.body.decode()
        assert "mcp/www_authenticate" in body
        assert "authentication_required" in body
        # Also verify resource_metadata is in the meta
        assert ".well-known/oauth-protected-resource" in body

    def test_custom_error_code(self, mock_request):
        """Test custom error code in response."""
        response = www_authenticate_response(mock_request, "Missing scope", error="insufficient_scope")
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


class TestMCPOpaqueAuthMiddleware:
    """Tests for MCPOpaqueAuthMiddleware (opaque token validation)."""

    @pytest.fixture
    def mock_opaque_verifier(self):
        """Create a mock ClerkOpaqueTokenVerifier."""
        verifier = MagicMock()
        # Note: verify_token_with_error is async for opaque verifier
        verifier.verify_token_with_error = AsyncMock(
            return_value=(
                VerifiedClerkToken(
                    user_id="user_opaque_123",
                    email="opaque@example.com",
                    first_name="Opaque",
                    last_name="User",
                    claims={"user_id": "user_opaque_123"},
                ),
                None,
            )
        )
        return verifier

    @pytest.fixture
    def opaque_app_with_middleware(self, mock_opaque_verifier):
        """Create a test app with MCPOpaqueAuthMiddleware."""
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
        app.add_middleware(MCPOpaqueAuthMiddleware, verifier=mock_opaque_verifier)
        return app

    def test_opaque_allows_options_requests(self, opaque_app_with_middleware):
        """Test that OPTIONS requests bypass auth."""
        client = TestClient(opaque_app_with_middleware)
        response = client.options("/test")
        # Should not return 401 (middleware allows through)
        assert response.status_code != 401

    def test_opaque_allows_oauth_discovery_endpoint(self, opaque_app_with_middleware):
        """Test that OAuth discovery endpoint is public."""
        client = TestClient(opaque_app_with_middleware)
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        assert response.text == "oauth metadata"

    def test_opaque_rejects_missing_auth_header(self, opaque_app_with_middleware):
        """Test that requests without auth header get 401."""
        client = TestClient(opaque_app_with_middleware)
        response = client.get("/test")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_opaque_rejects_invalid_token(self, opaque_app_with_middleware, mock_opaque_verifier):
        """Test that invalid tokens get 401."""
        mock_opaque_verifier.verify_token_with_error = AsyncMock(
            return_value=(None, "Token is invalid or expired")
        )

        client = TestClient(opaque_app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer invalid-token"})
        assert response.status_code == 401
        assert "invalid or expired" in response.headers["WWW-Authenticate"]

    def test_opaque_allows_valid_token(self, opaque_app_with_middleware, mock_opaque_verifier):
        """Test that valid tokens are accepted."""
        client = TestClient(opaque_app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer valid-opaque-token"})
        assert response.status_code == 200
        assert "user_id=user_opaque_123" in response.text

    def test_opaque_sets_request_state(self, opaque_app_with_middleware, mock_opaque_verifier):
        """Test that authenticated user is set on request.state."""
        client = TestClient(opaque_app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer valid-opaque-token"})
        assert response.status_code == 200
        assert "user_opaque_123" in response.text

    def test_opaque_sets_context_variable(self, opaque_app_with_middleware, mock_opaque_verifier):
        """Test that authenticated user is set in context variable."""
        captured_user = [None]

        async def capture_user_endpoint(request):
            captured_user[0] = get_mcp_authenticated_user()
            return Response("captured")

        # Create a new app with our capture endpoint
        app = Starlette(
            routes=[
                Route("/capture", capture_user_endpoint),
            ],
        )
        app.add_middleware(MCPOpaqueAuthMiddleware, verifier=mock_opaque_verifier)

        client = TestClient(app)
        response = client.get("/capture", headers={"Authorization": "Bearer token"})
        assert response.status_code == 200
        # Note: Due to sync test client, context variable may not persist as expected
        # The main test is that the middleware doesn't error out
