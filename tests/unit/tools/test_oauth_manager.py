"""Unit tests for OAuth token refresh — specifically the Oura provider branch."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.oauth_manager import get_oauth_token, refresh_oauth_token


def _make_connection(provider: str = "oura", refresh_token: str = "rt_test") -> MagicMock:
    conn = AsyncMock()
    conn.id = 1
    conn.provider = provider
    conn.refresh_token_enc = refresh_token
    conn.access_token_enc = None
    conn.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    conn.scopes = ""
    conn.updated_at = None
    conn.save = AsyncMock()
    return conn


@pytest.mark.unit
class TestRefreshOAuthTokenOura:

    @pytest.mark.asyncio
    async def test_oura_refresh_success(self):
        conn = _make_connection(provider="oura")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 86400,
            "scope": "heartrate",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("seer.tools.oauth_manager.config") as mock_config, \
             patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client_cls:
            mock_config.oura_client_id = "oura_id"
            mock_config.oura_client_secret = "oura_secret"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await refresh_oauth_token(conn)

        assert result.access_token_enc == "new_access_token"
        assert result.scopes == "heartrate"
        conn.save.assert_awaited_once()

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.args[0] == "https://api.ouraring.com/oauth/token"
        assert call_kwargs.kwargs["data"]["client_id"] == "oura_id"
        assert call_kwargs.kwargs["data"]["grant_type"] == "refresh_token"

    @pytest.mark.asyncio
    async def test_oura_refresh_missing_credentials(self):
        conn = _make_connection(provider="oura")

        with patch("seer.tools.oauth_manager.config") as mock_config:
            mock_config.oura_client_id = None
            mock_config.oura_client_secret = None

            with pytest.raises(HTTPException, match="Oura OAuth client credentials not configured"):
                await refresh_oauth_token(conn)

    @pytest.mark.asyncio
    async def test_unsupported_provider_still_raises(self):
        conn = _make_connection(provider="unknown_provider")

        with pytest.raises(HTTPException, match="Token refresh not supported for provider"):
            await refresh_oauth_token(conn)


@pytest.mark.unit
class TestGetOAuthTokenConnectionIdTypes:
    """Regression: connection_id from workflow runtime is int, not str."""

    @pytest.mark.asyncio
    async def test_int_connection_id_does_not_raise_type_error(self):
        """get_oauth_token must accept int connection_id without TypeError."""
        mock_user = MagicMock()
        mock_conn = _make_connection(provider="notion")
        mock_conn.access_token_enc = "token_abc"
        mock_conn.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch("seer.tools.oauth_manager.OAuthConnection") as MockOAuth:
            MockOAuth.filter.return_value.first = AsyncMock(return_value=mock_conn)
            connection, token = await get_oauth_token(mock_user, connection_id=22)

        assert connection == mock_conn
        assert token == "token_abc"

    @pytest.mark.asyncio
    async def test_str_connection_id_with_colon_prefix(self):
        """get_oauth_token parses 'provider:id' string connection_id."""
        mock_user = MagicMock()
        mock_conn = _make_connection(provider="notion")
        mock_conn.access_token_enc = "token_abc"
        mock_conn.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch("seer.tools.oauth_manager.OAuthConnection") as MockOAuth:
            MockOAuth.filter.return_value.first = AsyncMock(return_value=mock_conn)
            connection, token = await get_oauth_token(mock_user, connection_id="notion:22")

        assert connection == mock_conn
        # Verify the filter was called with id=22 (parsed from "notion:22")
        call_args = MockOAuth.filter.call_args
        assert call_args is not None
