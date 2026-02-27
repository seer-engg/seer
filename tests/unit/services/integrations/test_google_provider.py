"""Unit tests for GoogleProvider OAuth and token introspection."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from seer.services.integrations.providers.google import GoogleProvider
from seer.services.integrations.providers.base import OAuthAuthorizeContext


@pytest.mark.unit
class TestGoogleProviderScopeHandling:
    """Test GoogleProvider.get_oauth_scope()."""

    def test_get_oauth_scope_includes_required_openid_scopes(self):
        """Test that required OpenID scopes are always included."""
        provider = GoogleProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]

        scope = provider.get_oauth_scope(context)
        scope_list = scope.split()

        assert "openid" in scope_list
        assert "email" in scope_list
        assert "profile" in scope_list
        assert "https://www.googleapis.com/auth/gmail.readonly" in scope_list

    def test_get_oauth_scope_no_duplicates(self):
        """Test that scopes are not duplicated."""
        provider = GoogleProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["openid", "email", "profile"]

        scope = provider.get_oauth_scope(context)
        scope_list = scope.split()

        assert scope_list.count("openid") == 1
        assert scope_list.count("email") == 1
        assert scope_list.count("profile") == 1


@pytest.mark.unit
class TestGoogleProviderProperties:
    """Test GoogleProvider basic properties."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        provider = GoogleProvider()
        assert provider.provider == "google"

    def test_supports_provider_aliases(self):
        """Test supports_provider returns True for aliases."""
        provider = GoogleProvider()
        assert provider.supports_provider("google") is True
        assert provider.supports_provider("gmail") is True
        assert provider.supports_provider("googlesheets") is True
        assert provider.supports_provider("googledrive") is True
        assert provider.supports_provider("googlecalendar") is True
        assert provider.supports_provider("github") is False


@pytest.mark.unit
class TestGoogleTokenIntrospection:
    """Test GoogleProvider token introspection functionality."""

    @pytest.mark.asyncio
    async def test_introspect_token_success(self):
        """Test successful tokeninfo call."""
        provider = GoogleProvider()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "azp": "client_id",
            "aud": "client_id",
            "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",
            "exp": "1234567890",
            "access_type": "offline",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="",  # Not needed for Google
                client_secret="",  # Not needed for Google
            )

            assert result is not None
            assert result["scope"] == "openid email profile https://www.googleapis.com/auth/gmail.readonly"

            # Verify correct API call
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[0][0] == "https://oauth2.googleapis.com/tokeninfo"
            assert call_args[1]["params"]["access_token"] == "test_token"

    @pytest.mark.asyncio
    async def test_introspect_token_invalid_token(self):
        """Test introspection returns None for invalid token."""
        provider = GoogleProvider()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid token"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="invalid_token",
                client_id="",
                client_secret="",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_introspect_token_network_error(self):
        """Test introspection returns None on network error."""
        provider = GoogleProvider()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await provider.introspect_token(
                access_token="test_token",
                client_id="",
                client_secret="",
            )

            # Should return None on network error, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_uses_introspection(self):
        """Test resolve_granted_scopes prefers tokeninfo result."""
        provider = GoogleProvider()
        token = {"access_token": "test_token", "scope": "openid profile"}
        state_data = {"requested_scope": "openid"}

        # Mock introspection response with more scopes than token response
        tokeninfo_result = {
            "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",
        }

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = tokeninfo_result

            result = await provider.resolve_granted_scopes(
                token=token,
                state_data=state_data,
            )

        # Should use tokeninfo result
        assert result == "openid email profile https://www.googleapis.com/auth/gmail.readonly"
        mock_introspect.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_on_introspection_failure(self):
        """Test fallback to token scope when introspection fails."""
        provider = GoogleProvider()
        token = {"access_token": "test_token", "scope": "openid profile email"}
        state_data = {"requested_scope": "openid profile"}

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed

            result = await provider.resolve_granted_scopes(
                token=token,
                state_data=state_data,
            )

        # Should fall back to token scope
        assert result == "openid profile email"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_to_requested_scope(self):
        """Test fallback to requested scope when token has no scope."""
        provider = GoogleProvider()
        token = {"access_token": "test_token"}  # No scope in token
        state_data = {"requested_scope": "openid profile"}

        with patch.object(provider, "introspect_token", new_callable=AsyncMock) as mock_introspect:
            mock_introspect.return_value = None  # Introspection failed

            result = await provider.resolve_granted_scopes(
                token=token,
                state_data=state_data,
            )

        # Should fall back to requested scope
        assert result == "openid profile"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_no_access_token(self):
        """Test fallback when no access token in response."""
        provider = GoogleProvider()
        token = {}  # No access_token
        state_data = {"requested_scope": "openid profile"}

        result = await provider.resolve_granted_scopes(
            token=token,
            state_data=state_data,
        )

        # Should fall back to requested scope
        assert result == "openid profile"
