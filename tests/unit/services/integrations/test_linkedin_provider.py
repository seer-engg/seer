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
