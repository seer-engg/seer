"""
Shared utilities for Google Calendar polling adapters.

Contains common parsing, normalization, and error-handling logic
used by both event_changed and event_start adapters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from seer.core.triggers.polling.adapters.base import (
    PollAdapter,
    PollAdapterError,
    PollContext,
)

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


def parse_rfc3339(value: Optional[str]) -> Optional[datetime]:
    """Parse RFC3339 datetime string to datetime object."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def normalize_event(event: Dict[str, Any], calendar_id: str) -> Dict[str, Any]:
    """Normalize Google Calendar event to trigger payload format."""
    start = event.get("start", {})
    end = event.get("end", {})
    organizer = event.get("organizer", {})

    is_all_day = "date" in start and "dateTime" not in start

    attendees: List[Dict[str, Any]] = []
    for att in event.get("attendees", []):
        attendees.append({
            "email": att.get("email"),
            "display_name": att.get("displayName"),
            "response_status": att.get("responseStatus"),
            "optional": att.get("optional", False),
            "self": att.get("self", False),
        })

    return {
        "event_id": event.get("id"),
        "calendar_id": calendar_id,
        "summary": event.get("summary"),
        "description": event.get("description"),
        "location": event.get("location"),
        "status": event.get("status"),
        "html_link": event.get("htmlLink"),
        "start": {
            "datetime": start.get("dateTime") or start.get("date"),
            "timezone": start.get("timeZone"),
        },
        "end": {
            "datetime": end.get("dateTime") or end.get("date"),
            "timezone": end.get("timeZone"),
        },
        "organizer": {
            "email": organizer.get("email"),
            "display_name": organizer.get("displayName"),
            "self": organizer.get("self", False),
        },
        "attendees": attendees,
        "is_all_day": is_all_day,
        "recurring_event_id": event.get("recurringEventId"),
        "created": event.get("created"),
        "updated": event.get("updated"),
    }


class GoogleCalendarBaseAdapter(PollAdapter):
    """Base class with shared HTTP and config helpers for Google Calendar adapters."""

    async def _raise_for_status(self, response: httpx.Response, operation: str) -> None:
        """Raise PollAdapterError for HTTP errors."""
        if response.status_code < 400:
            return

        detail = {"status": response.status_code, "body": response.text[:500], "operation": operation}

        if response.status_code in {401, 403}:
            raise PollAdapterError(
                "Google Calendar authentication error",
                permanent=True,
                detail=detail,
            )
        if response.status_code == 429:
            raise PollAdapterError(
                "Google Calendar rate limited",
                backoff_seconds=60,
                detail=detail,
            )
        raise PollAdapterError("Google Calendar API error", detail=detail)

    def _resolve_calendar_id(self, ctx: PollContext) -> str:
        """Get calendar ID from config, defaulting to 'primary'."""
        config = ctx.subscription.provider_config or {}
        calendar_id = config.get("calendar_id")
        if calendar_id and isinstance(calendar_id, str) and calendar_id.strip():
            return calendar_id.strip()
        return "primary"
