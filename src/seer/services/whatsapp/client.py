"""WhatsApp Cloud API client for sending messages and verifying webhooks."""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Optional

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


async def send_text_message(phone: str, text: str) -> Dict[str, Any]:
    """Send a text message via WhatsApp Cloud API."""
    url = f"{GRAPH_API_BASE}/{config.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {config.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    # WhatsApp has a 4096 char limit per message; chunk if needed
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]
    last_response = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "text",
                "text": {"body": chunk},
            }
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            last_response = resp.json()
    return last_response


async def send_media_message(
    phone: str, media_url: str, caption: Optional[str] = None
) -> Dict[str, Any]:
    """Send an image message with optional caption via WhatsApp Cloud API."""
    url = f"{GRAPH_API_BASE}/{config.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {config.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    image_obj: Dict[str, str] = {"link": media_url}
    if caption:
        image_obj["caption"] = caption[:1024]
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "image",
        "image": image_obj,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify X-Hub-Signature-256 from Meta webhook."""
    if not config.whatsapp_app_secret:
        logger.warning("whatsapp_app_secret not configured, skipping signature verification")
        return True
    expected = hmac.HMAC(
        config.whatsapp_app_secret.encode(),  # pylint: disable=no-member  # Reason: config field is str at runtime, not FieldInfo
        payload,
        hashlib.sha256,
    ).hexdigest()
    received = signature.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


def handle_verification_challenge(mode: str, token: str, challenge: str) -> Optional[str]:
    """Handle Meta webhook verification GET request. Returns challenge if valid, else None."""
    if mode == "subscribe" and token == config.whatsapp_verify_token:
        logger.info("WhatsApp webhook verification successful")
        return challenge
    logger.warning("WhatsApp webhook verification failed", extra={"mode": mode})
    return None
