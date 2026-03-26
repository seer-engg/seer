"""Unit tests for Twilio WhatsApp webhook endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.webhooks.twilio_whatsapp import _parse_twilio_form, _get_whatsapp_subscription


@pytest.mark.unit
class TestParseTwilioForm:

    def test_basic_message(self):
        form = {
            "From": "whatsapp:+14155551234",
            "To": "whatsapp:+15551234567",
            "Body": "Hello!",
            "MessageSid": "SM123",
            "NumMedia": "0",
        }
        result = _parse_twilio_form(form)
        assert result["from"] == "+14155551234"
        assert result["to"] == "+15551234567"
        assert result["body"] == "Hello!"
        assert result["message_sid"] == "SM123"
        assert result["num_media"] == 0
        assert result["media_urls"] == []

    def test_message_with_media(self):
        form = {
            "From": "whatsapp:+14155551234",
            "To": "whatsapp:+15551234567",
            "Body": "Check this out",
            "MessageSid": "SM456",
            "NumMedia": "2",
            "MediaUrl0": "https://api.twilio.com/media/img1.jpg",
            "MediaContentType0": "image/jpeg",
            "MediaUrl1": "https://api.twilio.com/media/doc.pdf",
            "MediaContentType1": "application/pdf",
        }
        result = _parse_twilio_form(form)
        assert result["num_media"] == 2
        assert len(result["media_urls"]) == 2
        assert result["media_urls"][0]["url"] == "https://api.twilio.com/media/img1.jpg"
        assert result["media_urls"][0]["content_type"] == "image/jpeg"

    def test_empty_body(self):
        form = {"From": "whatsapp:+1234", "To": "whatsapp:+5678", "MessageSid": "SM789"}
        result = _parse_twilio_form(form)
        assert result["body"] == ""
        assert result["num_media"] == 0


@pytest.mark.unit
class TestGetWhatsAppSubscription:

    @pytest.mark.asyncio
    async def test_no_subscription_raises(self):
        with patch("seer.api.webhooks.twilio_whatsapp.TriggerSubscription") as mock_ts:
            mock_ts.filter.return_value.all = AsyncMock(return_value=[])
            with pytest.raises(LookupError, match="No active"):
                await _get_whatsapp_subscription()

    @pytest.mark.asyncio
    async def test_single_subscription_returned(self):
        sub = MagicMock()
        sub.trigger_key = "webhook.twilio.whatsapp"
        sub.enabled = True
        with patch("seer.api.webhooks.twilio_whatsapp.TriggerSubscription") as mock_ts:
            mock_ts.filter.return_value.all = AsyncMock(return_value=[sub])
            result = await _get_whatsapp_subscription()
        assert result is sub

    @pytest.mark.asyncio
    async def test_multiple_subscriptions_match_by_to_number(self):
        sub1 = MagicMock()
        sub1.provider_config = {"to_number": "+15551111111"}
        sub2 = MagicMock()
        sub2.provider_config = {"to_number": "+15552222222"}
        with patch("seer.api.webhooks.twilio_whatsapp.TriggerSubscription") as mock_ts:
            mock_ts.filter.return_value.all = AsyncMock(return_value=[sub1, sub2])
            result = await _get_whatsapp_subscription(to_number="whatsapp:+15552222222")
        assert result is sub2


@pytest.mark.unit
class TestTwilioWhatsAppEndpoint:

    @pytest.mark.asyncio
    async def test_webhook_returns_twiml(self):
        """Integration-style test using FastAPI TestClient."""
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        app = FastAPI()
        from seer.api.webhooks.twilio_whatsapp import router
        app.include_router(router)

        sub = MagicMock()
        sub.id = 1
        sub.enabled = True
        sub.trigger_key = "webhook.twilio.whatsapp"

        mock_event = MagicMock()
        mock_event.id = "evt_test123"

        with (
            patch("seer.api.webhooks.twilio_whatsapp._get_whatsapp_subscription", new_callable=AsyncMock, return_value=sub),
            patch("seer.api.webhooks.twilio_whatsapp.webhook_services.handle_webhook_for_subscription", new_callable=AsyncMock, return_value=mock_event),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/twilio/whatsapp",
                    data={
                        "From": "whatsapp:+14155551234",
                        "To": "whatsapp:+15551234567",
                        "Body": "Hello",
                        "MessageSid": "SM123",
                    },
                )

        assert resp.status_code == 200
        assert "<Response></Response>" in resp.text
        assert resp.headers["content-type"] == "application/xml"

    @pytest.mark.asyncio
    async def test_webhook_no_subscription_returns_twiml(self):
        """When no subscription exists, still return TwiML (don't error)."""
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        app = FastAPI()
        from seer.api.webhooks.twilio_whatsapp import router
        app.include_router(router)

        with patch("seer.api.webhooks.twilio_whatsapp._get_whatsapp_subscription", new_callable=AsyncMock, side_effect=LookupError("No active")):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/twilio/whatsapp",
                    data={"From": "whatsapp:+1234", "To": "whatsapp:+5678", "Body": "hi", "MessageSid": "SM000"},
                )

        assert resp.status_code == 200
        assert "<Response></Response>" in resp.text
