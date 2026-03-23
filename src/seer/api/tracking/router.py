"""
Email tracking endpoints — open pixel and click redirect.

These endpoints are unauthenticated because recipients clicking links
in emails are not necessarily Seer users.
"""
from __future__ import annotations

import base64
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import RedirectResponse

from seer.logger import get_logger
from seer.services import email_tracking

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/track", tags=["email-tracking"])

# 1x1 transparent GIF (43 bytes)
TRANSPARENT_GIF = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)

_SAFE_SCHEMES = {"http", "https"}


@router.get("/{tracking_id}/open", status_code=status.HTTP_200_OK)
async def track_open(tracking_id: UUID) -> Response:
    """Return a 1x1 transparent GIF and record the open."""
    try:
        await email_tracking.record_open(tracking_id)
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: tracking must never fail the response
        logger.exception("Failed to record open for %s", tracking_id)

    return Response(
        content=TRANSPARENT_GIF,
        media_type="image/gif",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.get("/{tracking_id}/click", status_code=status.HTTP_302_FOUND)
async def track_click(
    tracking_id: UUID,
    url: str = Query(..., description="Original link URL"),
) -> Response:
    """Record the click and redirect to the original URL."""
    # Validate URL scheme to prevent javascript:/data: injection
    parsed = urlparse(url)
    if parsed.scheme not in _SAFE_SCHEMES:
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content="Invalid URL scheme")

    try:
        await email_tracking.record_click(tracking_id, url)
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: tracking must never block the redirect
        logger.exception("Failed to record click for %s", tracking_id)

    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


__all__ = ["router"]
