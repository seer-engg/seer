"""
Unit tests for MCP OAuth discovery endpoint.
"""

import pytest
from unittest.mock import patch
from starlette.testclient import TestClient


@pytest.mark.unit
class TestOAuthProtectedResourceMetadata:
    """Tests for /.well-known/oauth-protected-resource endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the MCP HTTP app."""
        from seer.mcp.server import create_http_app
        app = create_http_app()
        return TestClient(app)

    def test_returns_oauth_metadata(self, client):
        """Test that endpoint returns OAuth protected resource metadata."""
        response = client.get("/.well-known/oauth-protected-resource")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "resource" in data
        assert "authorization_servers" in data
        assert "scopes_supported" in data

        # Verify authorization server is a list
        assert isinstance(data["authorization_servers"], list)
        assert len(data["authorization_servers"]) > 0

        # Verify scopes is a list
        assert isinstance(data["scopes_supported"], list)

    @patch("seer.mcp.server.config")
    def test_uses_config_values(self, mock_config, client):
        """Test that endpoint uses configuration values."""
        mock_config.mcp_server_url = "https://mcp.example.com"
        mock_config.mcp_oauth_authorization_server = "https://auth.example.com"
        mock_config.mcp_oauth_scopes = "profile,email,openid"
        mock_config.mcp_resource_documentation = "https://docs.example.com"

        # Need to recreate app with mocked config
        from seer.mcp.server import create_http_app
        app = create_http_app()
        test_client = TestClient(app)

        response = test_client.get("/.well-known/oauth-protected-resource")

        assert response.status_code == 200
        data = response.json()

        assert data["resource"] == "https://mcp.example.com"
        assert "https://auth.example.com" in data["authorization_servers"]
        assert "profile" in data["scopes_supported"]
        assert "email" in data["scopes_supported"]
        assert "openid" in data["scopes_supported"]
        assert data.get("resource_documentation") == "https://docs.example.com"

    def test_fallback_to_request_url(self, client):
        """Test that resource URL falls back to request URL when not configured."""
        with patch("seer.mcp.server.config") as mock_config:
            mock_config.mcp_server_url = None
            mock_config.mcp_oauth_authorization_server = "https://auth.example.com"
            mock_config.mcp_oauth_scopes = "profile"
            mock_config.mcp_resource_documentation = None

            from seer.mcp.server import create_http_app
            app = create_http_app()
            test_client = TestClient(app)

            response = test_client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 200
            data = response.json()

            # Should fall back to test client's base URL
            assert "resource" in data
            assert data["resource"] is not None

    def test_cors_headers(self, client):
        """Test that CORS headers are present on GET requests."""
        response = client.get(
            "/.well-known/oauth-protected-resource",
            headers={"Origin": "https://chat.openai.com"}
        )

        assert response.status_code == 200
        # CORS headers should be present
        assert response.headers.get("access-control-allow-origin") == "*"

    def test_documentation_url_optional(self, client):
        """Test that resource_documentation is optional."""
        with patch("seer.mcp.server.config") as mock_config:
            mock_config.mcp_server_url = "https://mcp.example.com"
            mock_config.mcp_oauth_authorization_server = "https://auth.example.com"
            mock_config.mcp_oauth_scopes = "profile"
            mock_config.mcp_resource_documentation = None

            from seer.mcp.server import create_http_app
            app = create_http_app()
            test_client = TestClient(app)

            response = test_client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 200
            data = response.json()

            # resource_documentation should not be in response when not configured
            assert "resource_documentation" not in data
