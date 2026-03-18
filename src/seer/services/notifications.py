"""
Ntfy push notification service for workflow approvals.

Sends push notifications via ntfy.sh (or self-hosted Ntfy) for HITL
approval flows. Zero setup: install Ntfy app on phone, subscribe to topic.
"""

from __future__ import annotations

from typing import Dict

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


async def send_ntfy_notification(
    *,
    title: str,
    message: str,
    tags: list[str] | None = None,
    click_url: str | None = None,
    priority: int = 4,
    topic: str | None = None,
) -> bool:
    """
    Send a push notification via Ntfy.

    Args:
        title: Notification title
        message: Notification body
        tags: Ntfy emoji tags (e.g., ["shopping_cart", "white_check_mark"])
        click_url: URL to open when notification is tapped
        priority: 1-5 (5=urgent, 4=high, 3=default)
        topic: Override default topic from config

    Returns:
        True if sent successfully, False otherwise
    """
    effective_topic = topic or config.ntfy_topic
    if not effective_topic:
        logger.warning("Ntfy notification skipped: no topic configured")
        return False

    url = f"{config.ntfy_base_url}/{effective_topic}"

    headers: Dict[str, str] = {
        "Title": title,
        "Priority": str(priority),
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Ntfy notification sent to topic '%s'", effective_topic)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: notifications must not crash caller
        logger.error("Ntfy notification failed: %s", exc)
        return False


async def send_approval_notification(
    *,
    cart_summary: str,
    total: str,
    approval_url: str,
) -> bool:
    """
    Send a cart approval push notification.

    Args:
        run_id: Workflow run ID
        cart_summary: Human-readable cart contents
        total: Cart total (e.g., "$23.47")
        approval_url: URL to the approval page
    """
    return await send_ntfy_notification(
        title=f"Target Cart Ready — {total}",
        message=cart_summary,
        tags=["shopping_cart", "white_check_mark"],
        click_url=approval_url,
        priority=4,
    )
