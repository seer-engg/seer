"""
Email tracking service — records sends, opens, clicks.

Owns the tracking_id lifecycle:
1. create_tracking_record() — before send, returns tracking_id for pixel/link injection
2. finalize_send() — after send, updates with provider_email_id
3. record_open() / record_click() — called from tracking endpoints
"""
from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import urlencode
from uuid import UUID, uuid4

from seer.config import config
from seer.database.email_models import EmailEvent
from seer.logger import get_logger
from seer.observability.posthog_client import capture_event

logger = get_logger(__name__)


# =============================================================================
# Tracking record lifecycle
# =============================================================================


async def create_tracking_record(  # pylint: disable=too-many-arguments  # Reason: all fields needed for email event correlation
    *,
    provider: str,
    email_type: str,
    recipient: str,
    subject: str | None = None,
    organization_id: int | None = None,
    workflow_run_id: str | None = None,
    user_id: str | None = None,
    metadata_json: Dict[str, Any] | None = None,
) -> UUID:
    """
    Create a pending email event and return the tracking_id.

    Call this BEFORE sending the email so the tracking_id can be
    embedded in the HTML (pixel + rewritten links).
    """
    tracking_id = uuid4()
    await EmailEvent.create(
        tracking_id=tracking_id,
        provider=provider,
        email_type=email_type,
        event_type="pending",
        recipient=recipient,
        subject=subject[:500] if subject else None,
        organization_id=organization_id,
        workflow_run_id=workflow_run_id,
        user_id=user_id,
        metadata_json=metadata_json,
    )
    return tracking_id


async def finalize_send(
    tracking_id: UUID,
    provider_email_id: str | None = None,
) -> None:
    """
    Mark a pending tracking record as sent and fire PostHog event.

    Call this AFTER the email provider confirms send success.
    """
    event = await EmailEvent.filter(tracking_id=tracking_id, event_type="pending").first()
    if not event:
        logger.warning("finalize_send: no pending event for tracking_id=%s", tracking_id)
        return

    event.event_type = "sent"
    event.provider_email_id = provider_email_id
    await event.save()

    capture_event(
        distinct_id=event.user_id or "system",
        event="email_sent",
        properties={
            "tracking_id": str(tracking_id),
            "provider": event.provider,
            "email_type": event.email_type,
            "recipient": event.recipient,
            "subject": event.subject,
            "organization_id": event.organization_id,
            "workflow_run_id": event.workflow_run_id,
        },
    )


async def record_open(tracking_id: UUID) -> None:
    """Record an email open event from the tracking pixel."""
    original = await EmailEvent.filter(tracking_id=tracking_id, event_type="sent").first()
    if not original:
        logger.debug("record_open: no sent event for tracking_id=%s", tracking_id)
        return

    await EmailEvent.create(
        tracking_id=tracking_id,
        provider=original.provider,
        email_type=original.email_type,
        event_type="opened",
        recipient=original.recipient,
        subject=original.subject,
        provider_email_id=original.provider_email_id,
        organization_id=original.organization_id,
        workflow_run_id=original.workflow_run_id,
        user_id=original.user_id,
    )

    capture_event(
        distinct_id=original.user_id or "system",
        event="email_opened",
        properties={
            "tracking_id": str(tracking_id),
            "email_type": original.email_type,
            "recipient": original.recipient,
            "organization_id": original.organization_id,
            "workflow_run_id": original.workflow_run_id,
        },
    )


async def record_click(tracking_id: UUID, link_url: str) -> None:
    """Record an email click event from the link redirect."""
    original = await EmailEvent.filter(tracking_id=tracking_id, event_type="sent").first()
    if not original:
        logger.debug("record_click: no sent event for tracking_id=%s", tracking_id)
        return

    await EmailEvent.create(
        tracking_id=tracking_id,
        provider=original.provider,
        email_type=original.email_type,
        event_type="clicked",
        recipient=original.recipient,
        subject=original.subject,
        provider_email_id=original.provider_email_id,
        organization_id=original.organization_id,
        workflow_run_id=original.workflow_run_id,
        user_id=original.user_id,
        link_url=link_url,
    )

    capture_event(
        distinct_id=original.user_id or "system",
        event="email_clicked",
        properties={
            "tracking_id": str(tracking_id),
            "email_type": original.email_type,
            "recipient": original.recipient,
            "link_url": link_url,
            "organization_id": original.organization_id,
            "workflow_run_id": original.workflow_run_id,
        },
    )


# =============================================================================
# HTML injection
# =============================================================================

# Matches href="..." in <a> tags (double or single quotes)
_HREF_RE = re.compile(r'(<a\b[^>]*?\bhref\s*=\s*")([^"]+)(")', re.IGNORECASE)


def _get_tracking_base_url() -> str:
    """Return the base URL for tracking endpoints."""
    base = config.webhook_base_url or "http://localhost:8000"
    return base.rstrip("/")


def inject_tracking(html_body: str, tracking_id: UUID) -> str:
    """
    Inject tracking pixel and rewrite all <a href> links in email HTML.

    - Appends a 1x1 transparent tracking pixel before </body>
    - Rewrites every <a href="URL"> to route through our click tracker
    - Skips mailto: and tel: links
    """
    base_url = _get_tracking_base_url()
    tid = str(tracking_id)

    # Rewrite links
    def _rewrite_link(match: re.Match) -> str:
        prefix, url, suffix = match.group(1), match.group(2), match.group(3)
        # Skip non-http links
        if url.startswith(("mailto:", "tel:", "#")):
            return match.group(0)
        redirect_url = f"{base_url}/v1/track/{tid}/click?{urlencode({'url': url})}"
        return f"{prefix}{redirect_url}{suffix}"

    html_body = _HREF_RE.sub(_rewrite_link, html_body)

    # Inject tracking pixel
    pixel = (
        f'<img src="{base_url}/v1/track/{tid}/open" '
        f'width="1" height="1" alt="" style="display:none;width:1px;height:1px;border:0">'
    )

    if "</body>" in html_body.lower():
        # Insert before </body>
        idx = html_body.lower().rfind("</body>")
        html_body = html_body[:idx] + pixel + "\n" + html_body[idx:]
    else:
        html_body += "\n" + pixel

    return html_body
