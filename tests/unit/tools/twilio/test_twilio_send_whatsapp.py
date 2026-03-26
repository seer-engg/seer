"""Unit tests for Twilio send WhatsApp tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.twilio.send_whatsapp import TwilioSendWhatsAppTool


def _make_credentials(
    account_sid: str = "AC_test_sid",
    auth_token: str = "test_auth_token",
    from_number: str = "+15551234567",
) -> ResolvedCredentials:
    creds = MagicMock(spec=ResolvedCredentials)
    creds.access_token = None
    creds.secrets = {
        "twilio_account_sid": account_sid,
        "twilio_auth_token": auth_token,
        "twilio_from_number": from_number,
    }
    return creds


@pytest.mark.unit
class TestTwilioSendWhatsAppTool:

    def test_tool_attributes(self):
        tool = TwilioSendWhatsAppTool()
        assert tool.name == "twilio_send_whatsapp"
        assert tool.provider == "twilio"
        assert tool.integration_type == "whatsapp"

    def test_get_parameters_schema(self):
        tool = TwilioSendWhatsAppTool()
        schema = tool.get_parameters_schema()
        assert "to" in schema["properties"]
        assert "message" in schema["properties"]
        assert schema["required"] == ["to", "message"]

    def test_get_output_schema(self):
        tool = TwilioSendWhatsAppTool()
        schema = tool.get_output_schema()
        assert "sid" in schema["properties"]
        assert "status" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = TwilioSendWhatsAppTool()
        creds = _make_credentials()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "sid": "SM123",
            "status": "queued",
            "to": "whatsapp:+14155551234",
            "from": "whatsapp:+15551234567",
        }

        mock_cfg = MagicMock()
        mock_cfg.twilio_whatsapp_from_number = None

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls, \
             patch("seer.config.config", mock_cfg):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                None,
                {"to": "+14155551234", "message": "Hello from Seer!"},
                credentials=creds,
            )

        assert result["sid"] == "SM123"
        assert result["status"] == "queued"

        # Verify whatsapp: prefix was added
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["data"]["To"] == "whatsapp:+14155551234"
        assert call_kwargs.kwargs["data"]["From"] == "whatsapp:+15551234567"
        assert call_kwargs.kwargs["data"]["Body"] == "Hello from Seer!"

    @pytest.mark.asyncio
    async def test_execute_strips_existing_whatsapp_prefix(self):
        tool = TwilioSendWhatsAppTool()
        creds = _make_credentials()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"sid": "SM123", "status": "queued", "to": "whatsapp:+14155551234", "from": "whatsapp:+15551234567"}

        mock_cfg = MagicMock()
        mock_cfg.twilio_whatsapp_from_number = None

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls, \
             patch("seer.config.config", mock_cfg):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await tool.execute(
                None,
                {"to": "whatsapp:+14155551234", "message": "test"},
                credentials=creds,
            )

        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["data"]["To"] == "whatsapp:+14155551234"

    @pytest.mark.asyncio
    async def test_execute_config_fallback_when_no_user_secrets(self):
        """Platform-provided creds from config work when user has no secrets."""
        tool = TwilioSendWhatsAppTool()
        creds = MagicMock(spec=ResolvedCredentials)
        creds.secrets = {}  # No user-provided secrets

        mock_config = MagicMock()
        mock_config.twilio_account_sid = "AC_platform_sid"
        mock_config.twilio_auth_token = "platform_token"
        mock_config.twilio_from_number = "+15559999999"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"sid": "SM456", "status": "queued", "to": "whatsapp:+14155551234", "from": "whatsapp:+15559999999"}

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls, \
             patch("seer.config.config", mock_config):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                None,
                {"to": "+14155551234", "message": "Platform creds!"},
                credentials=creds,
            )

        assert result["sid"] == "SM456"
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["auth"] == ("AC_platform_sid", "platform_token")

    @pytest.mark.asyncio
    async def test_user_secrets_override_config(self):
        """User-provided secrets take priority over platform config."""
        tool = TwilioSendWhatsAppTool()
        creds = _make_credentials(account_sid="AC_user", auth_token="user_token", from_number="+15551111111")

        mock_config = MagicMock()
        mock_config.twilio_account_sid = "AC_platform"
        mock_config.twilio_auth_token = "platform_token"
        mock_config.twilio_from_number = "+15559999999"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"sid": "SM789", "status": "queued", "to": "whatsapp:+14155551234", "from": "whatsapp:+15551111111"}

        with patch("seer.tools.twilio.base.httpx.AsyncClient") as mock_client_cls, \
             patch("seer.config.config", mock_config):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                None,
                {"to": "+14155551234", "message": "User creds!"},
                credentials=creds,
            )

        assert result["sid"] == "SM789"
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["auth"] == ("AC_user", "user_token")

    @pytest.mark.asyncio
    async def test_execute_missing_to(self):
        tool = TwilioSendWhatsAppTool()
        with pytest.raises(HTTPException, match="'to' is required"):
            await tool.execute(None, {"message": "hello"}, credentials=_make_credentials())

    @pytest.mark.asyncio
    async def test_execute_missing_message(self):
        tool = TwilioSendWhatsAppTool()
        with pytest.raises(HTTPException, match="'message' is required"):
            await tool.execute(None, {"to": "+1234"}, credentials=_make_credentials())
