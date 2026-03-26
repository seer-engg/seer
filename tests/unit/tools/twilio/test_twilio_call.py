"""Unit tests for Twilio make call tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.twilio.call import TwilioMakeCallTool


def _make_credentials(
    account_sid: str = "AC_test_sid",
    auth_token: str = "test_auth_token",
    from_number: str = "+15551234567",
) -> ResolvedCredentials:
    """Create mock ResolvedCredentials with Twilio secrets."""
    creds = MagicMock(spec=ResolvedCredentials)
    creds.access_token = None
    creds.secrets = {
        "twilio_account_sid": account_sid,
        "twilio_auth_token": auth_token,
        "twilio_from_number": from_number,
    }
    return creds


@pytest.mark.unit
class TestTwilioMakeCallTool:
    """Tests for TwilioMakeCallTool."""

    def test_tool_attributes(self):
        tool = TwilioMakeCallTool()
        assert tool.name == "twilio_make_call"
        assert tool.provider is None
        assert tool.integration_type == "phone"
        assert tool.required_secrets == []
        assert "twilio_account_sid" in tool.optional_secrets
        assert "twilio_auth_token" in tool.optional_secrets
        assert "twilio_from_number" in tool.optional_secrets

    def test_get_parameters_schema(self):
        tool = TwilioMakeCallTool()
        schema = tool.get_parameters_schema()
        assert schema["type"] == "object"
        assert "to" in schema["properties"]
        assert "message" in schema["properties"]
        assert schema["required"] == ["to", "message"]

    def test_get_output_schema(self):
        tool = TwilioMakeCallTool()
        schema = tool.get_output_schema()
        assert schema["type"] == "object"
        assert "sid" in schema["properties"]
        assert "status" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = TwilioMakeCallTool()
        credentials = _make_credentials()

        mock_response = httpx.Response(
            status_code=201,
            json={
                "sid": "CA_test_call_sid",
                "status": "queued",
                "to": "+14155559999",
                "from": "+15551234567",
            },
            request=httpx.Request("POST", "https://api.twilio.com/test"),
        )

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                access_token=None,
                arguments={"to": "+14155559999", "message": "Hey, check in time!"},
                credentials=credentials,
            )

        assert result["sid"] == "CA_test_call_sid"
        assert result["status"] == "queued"
        assert result["to"] == "+14155559999"

        # Verify the request was made with correct params
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "Calls.json" in call_args[0][1]
        assert call_args[1]["data"]["To"] == "+14155559999"
        assert call_args[1]["data"]["From"] == "+15551234567"
        assert "<Say" in call_args[1]["data"]["Twiml"]
        assert "Hey, check in time!" in call_args[1]["data"]["Twiml"]

    @pytest.mark.asyncio
    async def test_execute_missing_to(self):
        tool = TwilioMakeCallTool()
        credentials = _make_credentials()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"message": "hello"},
                credentials=credentials,
            )
        assert exc_info.value.status_code == 400
        assert "to" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_missing_message(self):
        tool = TwilioMakeCallTool()
        credentials = _make_credentials()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"to": "+14155559999"},
                credentials=credentials,
            )
        assert exc_info.value.status_code == 400
        assert "message" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_no_credentials(self):
        tool = TwilioMakeCallTool()

        with patch("seer.config.config") as mock_config:
            mock_config.twilio_account_sid = None
            mock_config.twilio_auth_token = None
            mock_config.twilio_from_number = None

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"to": "+14155559999", "message": "hello"},
                    credentials=None,
                )
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_execute_missing_secret(self):
        tool = TwilioMakeCallTool()
        creds = MagicMock(spec=ResolvedCredentials)
        creds.secrets = {"twilio_account_sid": "AC_test", "twilio_auth_token": "tok"}

        with patch("seer.config.config") as mock_config:
            mock_config.twilio_account_sid = None
            mock_config.twilio_auth_token = None
            mock_config.twilio_from_number = None

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"to": "+14155559999", "message": "hello"},
                    credentials=creds,
                )
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_execute_api_error(self):
        tool = TwilioMakeCallTool()
        credentials = _make_credentials()

        mock_response = httpx.Response(
            status_code=400,
            json={"message": "The 'To' number is not a valid phone number."},
            request=httpx.Request("POST", "https://api.twilio.com/test"),
        )

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"to": "invalid", "message": "hello"},
                    credentials=credentials,
                )
            assert exc_info.value.status_code == 400
