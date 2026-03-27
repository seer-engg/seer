"""
Twilio WhatsApp webhook — receives inbound WhatsApp messages via Twilio.

Twilio sends form-encoded POST with fields: From, To, Body, MessageSid, etc.
Routes to the active subscription for trigger_key=webhook.twilio.whatsapp.
Returns empty TwiML <Response/> as Twilio expects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response

from seer.api.webhooks import services as webhook_services
from seer.database import TriggerSubscription
from seer.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["webhooks"])

TWIML_EMPTY = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


async def _get_whatsapp_subscription(to_number: Optional[str] = None) -> TriggerSubscription:
    """Find the active subscription for webhook.twilio.whatsapp.

    Single-subscription lookup for now. If multiple exist in the future,
    match by `To` number in provider_config.
    """
    subscriptions = await TriggerSubscription.filter(
        trigger_key="webhook.twilio.whatsapp",
        enabled=True,
    ).all()

    if not subscriptions:
        raise LookupError("No active webhook.twilio.whatsapp subscription")

    if len(subscriptions) == 1:
        return subscriptions[0]

    # Multiple subscriptions: match by To number in provider_config
    if to_number:
        clean = to_number.removeprefix("whatsapp:")
        for sub in subscriptions:
            cfg_number = (sub.provider_config or {}).get("to_number", "")
            if cfg_number.removeprefix("whatsapp:") == clean:
                return sub

    # Fallback to first
    return subscriptions[0]


def _parse_twilio_form(form: dict) -> dict:
    """Normalize Twilio's form-encoded fields into our trigger event payload."""
    from_number = (form.get("From") or "").removeprefix("whatsapp:")
    to_number = (form.get("To") or "").removeprefix("whatsapp:")

    num_media = int(form.get("NumMedia") or 0)
    media_urls = []
    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if url:
            media_urls.append({
                "url": url,
                "content_type": form.get(f"MediaContentType{i}"),
            })

    return {
        "from": from_number,
        "to": to_number,
        "body": form.get("Body") or "",
        "message_sid": form.get("MessageSid") or "",
        "num_media": num_media,
        "media_urls": media_urls,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/twilio/whatsapp")
async def twilio_whatsapp_webhook(request: Request) -> Response:
    """Receive inbound WhatsApp messages from Twilio."""
    form = dict(await request.form())
    message_sid = form.get("MessageSid")

    logger.info(
        "Twilio WhatsApp webhook received",
        extra={"message_sid": message_sid, "from": form.get("From")},
    )

    payload = _parse_twilio_form(form)

    try:
        subscription = await _get_whatsapp_subscription(to_number=form.get("To"))
    except LookupError:
        logger.warning("No active WhatsApp subscription found, ignoring message %s", message_sid)
        return Response(content=TWIML_EMPTY, media_type="application/xml")

    await webhook_services.handle_webhook_for_subscription(
        subscription,
        payload=payload,
        headers=request.headers,
        provider_event_id=message_sid,
    )

    return Response(content=TWIML_EMPTY, media_type="application/xml")
