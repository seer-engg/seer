"""Unit tests for OuraProvider OAuth profile fetching."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from seer.services.integrations.providers.oura import OuraProvider


@pytest.mark.unit
class TestOuraProviderFetchUserProfile:
    """Test OuraProvider.fetch_user_profile()."""

    @pytest.mark.asyncio
    async def test_fetch_user_profile_success(self):
        provider = OuraProvider()
        token = {"access_token": "test_token_123"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "oura_user_abc",
            "email": "user@example.com",
            "age": 30,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await provider.fetch_user_profile(
                client=None, token=token, state_data={}
            )

        assert result == {"id": "oura_user_abc", "email": "user@example.com", "age": 30}
        mock_client.get.assert_called_once_with(
            "https://api.ouraring.com/v2/usercollection/personal_info",
            headers={"Authorization": "Bearer test_token_123"},
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_fetch_user_profile_no_access_token(self):
        provider = OuraProvider()

        with pytest.raises(HTTPException) as exc_info:
            await provider.fetch_user_profile(
                client=None, token={}, state_data={}
            )

        assert exc_info.value.status_code == 500
        assert "No access token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_fetch_user_profile_api_error(self):
        provider = OuraProvider()
        token = {"access_token": "bad_token"}

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await provider.fetch_user_profile(
                    client=None, token=token, state_data={}
                )

        assert exc_info.value.status_code == 502


@pytest.mark.unit
class TestOuraProviderProperties:
    def test_provider_name(self):
        assert OuraProvider().provider == "oura"
