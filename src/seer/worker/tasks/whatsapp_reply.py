"""Background task to send WhatsApp reply messages."""
from __future__ import annotations

from seer.logger import get_logger
from seer.services.whatsapp.client import send_text_message
from seer.worker.broker_instance import broker

logger = get_logger(__name__)


@broker.task
async def whatsapp_reply_task(phone: str, text: str) -> None:
    """Send a reply message back to a WhatsApp user."""
    logger.info("Sending WhatsApp reply", extra={"phone": phone, "text_length": len(text)})
    try:
        await send_text_message(phone, text)
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Background task must not propagate failures
        logger.exception("Failed to send WhatsApp reply", extra={"phone": phone})


__all__ = ["whatsapp_reply_task"]
