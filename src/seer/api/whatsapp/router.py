"""WhatsApp webhook endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response, status

from seer.api.whatsapp import services as wa_services
from seer.logger import get_logger
from seer.services.whatsapp.client import handle_verification_challenge, verify_webhook_signature

router = APIRouter(prefix="/v1/webhooks/whatsapp", tags=["whatsapp"])
logger = get_logger(__name__)


@router.get("")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
):
    """Handle Meta webhook verification challenge."""
    challenge = handle_verification_challenge(hub_mode, hub_verify_token, hub_challenge)
    if challenge is not None:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=status.HTTP_403_FORBIDDEN)


@router.post("", status_code=status.HTTP_200_OK)
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages. Returns 200 immediately, processes async."""
    body_bytes = await request.body()

    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_webhook_signature(body_bytes, signature):
        logger.warning("Invalid WhatsApp webhook signature")
        return {"status": "error", "detail": "invalid signature"}

    body = await request.json()
    messages = wa_services.extract_messages(body)

    for msg in messages:
        # Idempotency check
        if await wa_services.is_duplicate(msg["message_id"]):
            logger.debug("Duplicate WhatsApp message skipped", extra={"message_id": msg["message_id"]})
            continue

        phone = msg["phone"]
        text = msg["text"]

        user = await wa_services.get_user_for_phone(phone)
        if not user:
            await wa_services.handle_unlinked_phone(phone)
            continue

        await wa_services.route_message(phone, text, user)

    return {"status": "ok"}


__all__ = ["router"]
