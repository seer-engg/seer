"""Unit tests for GitHubProvider OAuth and token introspection."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from seer.services.integrations.providers.github import GitHubProvider
from seer.services.integrations.providers.base import OAuthAuthorizeContext


@pytest.mark.unit
class TestGitHubProviderScopeHandling:
    """Test GitHubProvider.get_oauth_scope()."""

    def test_get_oauth_scope_joins_with_space(self):
        """Test that get_oauth_scope joins scopes with space separator."""
        provider = GitHubProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["repo", "user:email", "read:org"]

        scope = provider.get_oauth_scope(context)
        assert scope == "repo user:email read:org"

    def test_get_oauth_scope_empty_scopes(self):
        """Test get_oauth_scope with empty scopes."""
        provider = GitHubProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = []

        scope = provider.get_oauth_scope(context)
        assert scope == ""


@pytest.mark.unit
class TestGitHubProviderProperties:
    """Test GitHubProvider basic properties."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        provider = GitHubProvider()
        assert provider.provider == "github"

    def test_supports_provider(self):
        """Test supports_provider returns True for 'github'."""
        provider = GitHubProvider()
        assert provider.supports_provider("github") is True
        assert provider.supports_provider("google") is False
        assert provider.supports_provider("linkedin") is False


@pytest.mark.unit
class TestGitHubTokenIntrospection:
    """Test GitHubProvider token introspection functionality."""

    @pytest.mark.asyncio
    async def test_introspect_token_success(self):
        """Test successful token check."""
        provider = GitHubProvider()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 1,
            "token": "gho_test123",
            "scopes": ["repo", "user:email", "read:org"],
            "user": {"login": "octocat", "id": 1},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="test_client_id",
                client_secret="test_client_secret",
            )

            assert result is not None
            assert result["scopes"] == ["repo", "user:email", "read:org"]

            # Verify correct API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.github.com/applications/test_client_id/token"
            assert "Authorization" in call_args[1]["headers"]
            assert call_args[1]["headers"]["Authorization"].startswith("Basic ")
            assert call_args[1]["json"]["access_token"] == "test_token"

    @pytest.mark.asyncio
    async def test_introspect_token_invalid_token(self):
        """Test introspection returns None for invalid/revoked token."""
        provider = GitHubProvider()

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="invalid_token",
                client_id="test_client_id",
                client_secret="test_client_secret",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_introspect_token_missing_credentials(self):
        """Test introspection returns None when credentials missing."""
        provider = GitHubProvider()

        result = await provider.introspect_token(
            access_token="test_token",
            client_id="",  # Missing
            client_secret="test_client_secret",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_introspect_token_network_error(self):
        """Test introspection returns None on network error."""
        provider = GitHubProvider()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="test_client_id",
                client_secret="test_client_secret",
            )

            # Should return None on network error, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_uses_introspection(self):
        """Test resolve_granted_scopes prefers token check result."""
        provider = GitHubProvider()
        token = {"access_token": "test_token", "scope": "repo"}
        state_data = {"requested_scope": "repo user:email"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.github_client_id = "test_client_id"
        mock_config.github_client_secret = "test_client_secret"

        # Mock introspection response with scopes as array
        token_check_result = {
            "scopes": ["repo", "user:email", "read:org"],
        }

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = token_check_result
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should use token check result, joined with space
        assert result == "repo user:email read:org"
        mock_introspect.assert_called_once_with(
            access_token="test_token",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_on_introspection_failure(self):
        """Test fallback to token scope when introspection fails."""
        provider = GitHubProvider()
        token = {"access_token": "test_token", "scope": "repo user:email"}
        state_data = {"requested_scope": "repo"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.github_client_id = "test_client_id"
        mock_config.github_client_secret = "test_client_secret"

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should fall back to token scope
        assert result == "repo user:email"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_to_requested_scope(self):
        """Test fallback to requested scope when token has no scope."""
        provider = GitHubProvider()
        token = {"access_token": "test_token"}  # No scope in token
        state_data = {"requested_scope": "repo user:email"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.github_client_id = "test_client_id"
        mock_config.github_client_secret = "test_client_secret"

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should fall back to requested scope
        assert result == "repo user:email"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_without_credentials(self):
        """Test fallback when client credentials not configured."""
        provider = GitHubProvider()
        token = {"access_token": "test_token", "scope": "repo"}
        state_data = {"requested_scope": "user:email"}

        # Mock config without credentials
        mock_config = Mock()
        mock_config.github_client_id = None
        mock_config.github_client_secret = None

        with patch("seer.config.config", mock_config):
            result = await provider.resolve_granted_scopes(
                token=token,
                state_data=state_data,
            )

        # Should fall back to token scope (no introspection without credentials)
        assert result == "repo"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_no_access_token(self):
        """Test fallback when no access token in response."""
        provider = GitHubProvider()
        token = {}  # No access_token
        state_data = {"requested_scope": "repo user:email"}

        result = await provider.resolve_granted_scopes(
            token=token,
            state_data=state_data,
        )

        # Should fall back to requested scope
        assert result == "repo user:email"
