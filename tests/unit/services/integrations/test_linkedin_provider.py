"""Unit tests for LinkedInProvider OAuth scope handling."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from seer.services.integrations.providers.linkedin import LinkedInProvider
from seer.services.integrations.providers.base import OAuthAuthorizeContext


@pytest.mark.unit
class TestLinkedInProviderScopeHandling:
    """Test LinkedInProvider.get_oauth_scope()."""

    def test_get_oauth_scope_joins_with_space(self):
        """Test that get_oauth_scope joins scopes with space separator."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["openid", "profile", "email"]

        scope = provider.get_oauth_scope(context)
        assert scope == "openid profile email"

    def test_get_oauth_scope_single_scope(self):
        """Test get_oauth_scope with single scope adds required OpenID scopes."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["openid"]

        scope = provider.get_oauth_scope(context)
        # Profile is added as it's required for userinfo endpoint
        assert scope == "openid profile"

    def test_get_oauth_scope_empty_scopes_still_includes_required_openid_scopes(self):
        """Test get_oauth_scope with empty scopes still includes required OpenID scopes."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = []

        scope = provider.get_oauth_scope(context)
        # Even with no requested scopes, openid and profile are always included
        assert scope == "openid profile"

    def test_get_oauth_scope_posting_scope(self):
        """Test get_oauth_scope with posting scopes includes openid."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["openid", "profile", "email", "w_member_social"]

        scope = provider.get_oauth_scope(context)
        assert scope == "openid profile email w_member_social"

    def test_get_oauth_scope_always_includes_required_openid_scopes(self):
        """Test that openid and profile scopes are automatically added when missing."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["w_member_social"]  # Only posting scope

        scope = provider.get_oauth_scope(context)
        scope_list = scope.split()

        assert "openid" in scope_list
        assert "profile" in scope_list
        assert "w_member_social" in scope_list

    def test_get_oauth_scope_no_duplicate_openid_scopes(self):
        """Test that openid and profile are not duplicated if already present."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["openid", "profile", "w_member_social"]

        scope = provider.get_oauth_scope(context)
        scope_list = scope.split()

        # Each should appear exactly once
        assert scope_list.count("openid") == 1
        assert scope_list.count("profile") == 1

    def test_get_oauth_scope_preserves_order(self):
        """Test that scope order is preserved with required scopes appended."""
        provider = LinkedInProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["w_member_social", "email"]

        scope = provider.get_oauth_scope(context)
        scope_list = scope.split()

        # w_member_social and email should be first (in order), openid/profile appended
        assert scope_list.index("w_member_social") < scope_list.index("openid")
        assert scope_list.index("email") < scope_list.index("openid")
        assert "profile" in scope_list


@pytest.mark.unit
class TestLinkedInProviderUserProfile:
    """Test LinkedInProvider.fetch_user_profile()."""

    @pytest.mark.asyncio
    async def test_fetch_user_profile_success(self):
        """Test successful user profile fetch."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token_123"}
        state_data = {}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sub": "linkedin_user_id",
            "name": "John Doe",
            "email": "john@example.com",
            "picture": "https://example.com/pic.jpg",
            "email_verified": True
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            profile = await provider.fetch_user_profile(
                client=None,
                token=token,
                state_data=state_data
            )

            assert profile["sub"] == "linkedin_user_id"
            assert profile["name"] == "John Doe"
            assert profile["email"] == "john@example.com"

            # Verify correct API call
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[0][0] == "https://api.linkedin.com/v2/userinfo"
            assert call_args[1]["headers"]["Authorization"] == "Bearer test_token_123"

    @pytest.mark.asyncio
    async def test_fetch_user_profile_missing_token(self):
        """Test fetch_user_profile raises error when access_token is missing."""
        provider = LinkedInProvider()
        token = {}  # No access_token
        state_data = {}

        with pytest.raises(Exception) as exc_info:
            await provider.fetch_user_profile(
                client=None,
                token=token,
                state_data=state_data
            )

        assert "access token" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fetch_user_profile_api_error(self):
        """Test fetch_user_profile handles API errors."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token_123"}
        state_data = {}

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await provider.fetch_user_profile(
                    client=None,
                    token=token,
                    state_data=state_data
                )

            assert "401" in str(exc_info.value) or "failed" in str(exc_info.value).lower()


@pytest.mark.unit
class TestLinkedInProviderProperties:
    """Test LinkedInProvider basic properties."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        provider = LinkedInProvider()
        assert provider.provider == "linkedin"

    def test_supports_provider(self):
        """Test supports_provider returns True for 'linkedin'."""
        provider = LinkedInProvider()
        assert provider.supports_provider("linkedin") is True
        assert provider.supports_provider("github") is False
        assert provider.supports_provider("google") is False


@pytest.mark.unit
class TestLinkedInTokenIntrospection:
    """Test LinkedInProvider token introspection functionality."""

    @pytest.mark.asyncio
    async def test_introspect_token_success(self):
        """Test successful token introspection."""
        provider = LinkedInProvider()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "active": True,
            "scope": "openid profile email w_member_social",
            "client_id": "test_client",
            "exp": 1234567890,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="client_id",
                client_secret="client_secret",
            )

            assert result is not None
            assert result["active"] is True
            assert result["scope"] == "openid profile email w_member_social"

            # Verify correct API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://www.linkedin.com/oauth/v2/introspectToken"
            assert call_args[1]["data"]["client_id"] == "client_id"
            assert call_args[1]["data"]["client_secret"] == "client_secret"
            assert call_args[1]["data"]["token"] == "test_token"

    @pytest.mark.asyncio
    async def test_introspect_token_inactive(self):
        """Test introspection returns None for inactive token."""
        provider = LinkedInProvider()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "active": False,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="expired_token",
                client_id="client_id",
                client_secret="client_secret",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_introspect_token_api_error(self):
        """Test introspection returns None on API error."""
        provider = LinkedInProvider()

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="client_id",
                client_secret="client_secret",
            )

            # Should return None on failure, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_introspect_token_network_error(self):
        """Test introspection returns None on network error."""
        provider = LinkedInProvider()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="client_id",
                client_secret="client_secret",
            )

            # Should return None on network error, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_uses_introspection(self):
        """Test resolve_granted_scopes prefers introspection result."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token", "scope": "openid profile"}
        state_data = {"requested_scope": "openid profile email"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.linkedin_client_id = "test_client_id"
        mock_config.linkedin_client_secret = "test_client_secret"

        # Mock introspection response with more scopes than token response
        introspection_result = {
            "active": True,
            "scope": "openid profile email w_member_social",
        }

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = introspection_result
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should use introspection result
        assert result == "openid profile email w_member_social"
        mock_introspect.assert_called_once_with(
            access_token="test_token",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_on_introspection_failure(self):
        """Test fallback to token scope when introspection fails."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token", "scope": "openid profile email"}
        state_data = {"requested_scope": "openid profile"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.linkedin_client_id = "test_client_id"
        mock_config.linkedin_client_secret = "test_client_secret"

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should fall back to token scope
        assert result == "openid profile email"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_to_requested_scope(self):
        """Test fallback to requested scope when token has no scope."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token"}  # No scope in token
        state_data = {"requested_scope": "openid profile"}

        # Mock config with credentials
        mock_config = Mock()
        mock_config.linkedin_client_id = "test_client_id"
        mock_config.linkedin_client_secret = "test_client_secret"

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed
            with patch("seer.config.config", mock_config):
                result = await provider.resolve_granted_scopes(
                    token=token,
                    state_data=state_data,
                )

        # Should fall back to requested scope
        assert result == "openid profile"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_without_credentials(self):
        """Test fallback when client credentials not configured."""
        provider = LinkedInProvider()
        token = {"access_token": "test_token", "scope": "openid profile"}
        state_data = {"requested_scope": "openid"}

        # Mock config without credentials
        mock_config = Mock()
        mock_config.linkedin_client_id = None
        mock_config.linkedin_client_secret = None

        with patch("seer.config.config", mock_config):
            result = await provider.resolve_granted_scopes(
                token=token,
                state_data=state_data,
            )

        # Should fall back to token scope (no introspection without credentials)
        assert result == "openid profile"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_no_access_token(self):
        """Test fallback when no access token in response."""
        provider = LinkedInProvider()
        token = {}  # No access_token
        state_data = {"requested_scope": "openid profile"}

        result = await provider.resolve_granted_scopes(
            token=token,
            state_data=state_data,
        )

        # Should fall back to requested scope
        assert result == "openid profile"
