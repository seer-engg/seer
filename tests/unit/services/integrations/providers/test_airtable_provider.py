"""Unit tests for Airtable integration provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.integrations.providers.airtable import AirtableProvider
from seer.services.integrations.providers.base import OAuthAuthorizeContext


@pytest.mark.unit
class TestAirtableProviderOAuth:
    """Tests for Airtable provider OAuth methods."""

    def test_provider_attributes(self):
        """Test provider has correct attributes."""
        provider = AirtableProvider()

        assert provider.provider == "airtable"
        assert "base" in provider.resource_types
        assert "table" in provider.resource_types

    def test_get_oauth_scope(self):
        """Test scope formatting for Airtable."""
        provider = AirtableProvider()

        context = MagicMock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["data.records:read", "schema.bases:read", "user.email:read"]

        scope = provider.get_oauth_scope(context)

        assert scope == "data.records:read schema.bases:read user.email:read"

    def test_build_authorize_kwargs_simplified(self):
        """Test authorize kwargs returns minimal params (PKCE handled by Authlib)."""
        provider = AirtableProvider()

        context = MagicMock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["data.records:read"]

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",  # Ignored - Authlib handles state
            scope="data.records:read",
        )

        # Should only include scope - Authlib handles state and PKCE
        assert kwargs == {"scope": "data.records:read"}

        # Should NOT include PKCE params (Authlib handles them)
        assert "code_challenge" not in kwargs
        assert "code_challenge_method" not in kwargs
        assert "_code_verifier" not in kwargs
        assert "state" not in kwargs

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_from_token(self):
        """Test scope extraction from token response."""
        provider = AirtableProvider()

        token = {"scope": "data.records:read data.records:write"}
        state_data = {"requested_scope": "data.records:read"}

        scopes = await provider.resolve_granted_scopes(token=token, state_data=state_data)

        # Should prefer token scope over state
        assert scopes == "data.records:read data.records:write"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_fallback_to_state(self):
        """Test scope extraction falls back to state when token lacks scope."""
        provider = AirtableProvider()

        token = {"access_token": "test_token"}  # No scope field
        state_data = {"requested_scope": "data.records:read"}

        scopes = await provider.resolve_granted_scopes(token=token, state_data=state_data)

        assert scopes == "data.records:read"


@pytest.mark.unit
class TestAirtableProviderProfile:
    """Tests for Airtable provider profile fetching."""

    @pytest.mark.asyncio
    async def test_fetch_user_profile_success(self):
        """Test successful user profile fetch."""
        provider = AirtableProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "usr_test123",
            "email": "test@example.com",
            "scopes": ["data.records:read", "schema.bases:read"],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            profile = await provider.fetch_user_profile(
                client=None,
                token={"access_token": "test_token"},
                state_data={},
            )

            assert profile["id"] == "usr_test123"
            assert profile["email"] == "test@example.com"

            # Verify correct endpoint was called
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert "meta/whoami" in call_args[0][0]
            assert "Bearer test_token" in call_args[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_fetch_user_profile_missing_token(self):
        """Test profile fetch fails when token is missing."""
        provider = AirtableProvider()

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await provider.fetch_user_profile(
                client=None,
                token={},  # No access_token
                state_data={},
            )

        assert exc_info.value.status_code == 500
        assert "access token" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_fetch_user_profile_api_error(self):
        """Test profile fetch handles API errors."""
        provider = AirtableProvider()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await provider.fetch_user_profile(
                    client=None,
                    token={"access_token": "invalid_token"},
                    state_data={},
                )

            assert exc_info.value.status_code == 500
            assert "401" in exc_info.value.detail
