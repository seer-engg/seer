"""
Google Calendar polling adapter for event start time triggers.

Fires at a configurable offset before or after an event's start time.
Uses the Calendar API to fetch upcoming events and tracks which instances
have already been triggered to prevent duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from seer.core.triggers.polling.adapters.base import (
    PollAdapter,
    PollAdapterError,
    PollContext,
    PolledEvent,
    PollResult,
    register_adapter,
)
from seer.logger import get_logger

logger = get_logger(__name__)

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
LOOKAHEAD_HOURS = 24
MAX_EVENTS_PER_POLL = 100
CURSOR_CLEANUP_DAYS = 7


@dataclass
class _PollConfig:
    calendar_id: str
    offset_minutes: int
    include_all_day: bool


def _parse_rfc3339(value: Optional[str]) -> Optional[datetime]:
    """Parse RFC3339 datetime string to datetime object."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _normalize_event(event: Dict[str, Any], calendar_id: str) -> Dict[str, Any]:
    """Normalize Google Calendar event to trigger payload format."""
    start = event.get("start", {})
    end = event.get("end", {})
    organizer = event.get("organizer", {})

    is_all_day = "date" in start and "dateTime" not in start

    attendees = []
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


def _get_event_start_utc(event: Dict[str, Any]) -> Optional[datetime]:
    """Extract UTC start time from event, handling all-day events."""
    start = event.get("start", {})
    datetime_str = start.get("dateTime")

    if datetime_str:
        return _parse_rfc3339(datetime_str)

    # All-day event: use the date at midnight UTC
    date_str = start.get("date")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


def _make_instance_key(event_id: str, start_utc: datetime) -> str:
    """Create unique key for an event instance (handles recurring events)."""
    return f"{event_id}_{start_utc.isoformat()}"


class GoogleCalendarEventStartAdapter(PollAdapter):
    """
    Poll Google Calendar for events approaching their trigger time.

    Fires at a configurable offset before or after event start times.
    Each event instance (including recurring) only triggers once.
    """

    trigger_key = "poll.google_calendar.event_start"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """
        Initialize cursor without backfilling existing events.

        Sets bootstrapped_at to now so only future events trigger.
        """
        now = datetime.now(timezone.utc)
        return {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": now.isoformat(),
        }

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """
        Fetch upcoming events and emit those whose trigger time has passed.
        """
        now = datetime.now(timezone.utc)
        config = self._resolve_poll_config(ctx)
        bootstrapped_at, triggered_instances, last_cleanup = self._unpack_cursor(cursor, now)

        events = await self._fetch_calendar_events(ctx, config, now)

        polled_events, next_trigger_time, triggered_instances = self._process_events(
            events,
            config,
            bootstrapped_at=bootstrapped_at,
            triggered_instances=triggered_instances,
            now=now,
            ctx=ctx,
        )

        triggered_instances, last_cleanup = self._maybe_cleanup(triggered_instances, last_cleanup, now)

        new_cursor = {
            "triggered_instances": triggered_instances,
            "last_cleanup_utc": last_cleanup.isoformat(),
            "bootstrapped_at": bootstrapped_at.isoformat(),
        }

        return PollResult(
            events=polled_events,
            cursor=new_cursor,
            has_more=False,
            rate_limit_hint=self._compute_rate_limit_hint(next_trigger_time, now),
        )

    def _resolve_poll_config(self, ctx: PollContext) -> _PollConfig:
        return _PollConfig(
            calendar_id=self._resolve_calendar_id(ctx),
            offset_minutes=self._resolve_offset_minutes(ctx),
            include_all_day=self._resolve_include_all_day(ctx),
        )

    def _unpack_cursor(
        self, cursor: Dict[str, Any], now: datetime
    ) -> tuple[datetime, Dict[str, str], Optional[datetime]]:
        bootstrapped_at = _parse_rfc3339(cursor.get("bootstrapped_at")) or now
        triggered_instances: Dict[str, str] = cursor.get("triggered_instances", {})
        last_cleanup = _parse_rfc3339(cursor.get("last_cleanup_utc"))
        return bootstrapped_at, triggered_instances, last_cleanup

    async def _fetch_calendar_events(
        self, ctx: PollContext, config: _PollConfig, now: datetime
    ) -> List[Dict[str, Any]]:
        """Compute the time window, build headers, and fetch events with error handling."""
        time_min = now - timedelta(minutes=config.offset_minutes + 60) if config.offset_minutes > 0 else now
        time_max = now + timedelta(hours=LOOKAHEAD_HOURS)
        headers = {"Authorization": f"Bearer {ctx.access_token}", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                return await self._fetch_events(
                    client, headers, config.calendar_id, time_min=time_min, time_max=time_max
                )
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Failed to fetch Google Calendar events")
            raise PollAdapterError(
                "Failed to fetch calendar events",
                detail={"error": str(exc)},
            ) from exc

    def _process_events(
        self,
        events: List[Dict[str, Any]],
        config: _PollConfig,
        *,
        bootstrapped_at: datetime,
        triggered_instances: Dict[str, str],
        now: datetime,
        ctx: PollContext,
    ) -> tuple[List[PolledEvent], Optional[datetime], Dict[str, str]]:
        """Evaluate each event and return fired events, next upcoming trigger, and updated instances."""
        polled: List[PolledEvent] = []
        next_trigger_time: Optional[datetime] = None
        updated_instances = dict(triggered_instances)

        for event in events:
            if not self._is_event_eligible(event, config, ctx):
                continue

            start_utc = _get_event_start_utc(event)
            if not start_utc:
                continue

            trigger_time = start_utc + timedelta(minutes=config.offset_minutes)
            instance_key = _make_instance_key(event.get("id", ""), start_utc)

            if instance_key in updated_instances or trigger_time < bootstrapped_at:
                continue

            if trigger_time <= now:
                polled.append(self._build_polled_event(event, config, start_utc, trigger_time))
                updated_instances[instance_key] = now.isoformat()
            elif next_trigger_time is None or trigger_time < next_trigger_time:
                next_trigger_time = trigger_time

        return polled, next_trigger_time, updated_instances

    def _is_event_eligible(
        self, event: Dict[str, Any], config: _PollConfig, ctx: PollContext
    ) -> bool:
        """Return False for cancelled events, filtered-out all-day events, or non-matching filters."""
        if event.get("status") == "cancelled":
            return False
        is_all_day = "date" in event.get("start", {}) and "dateTime" not in event.get("start", {})
        if is_all_day and not config.include_all_day:
            return False
        return self._matches_filters(event, ctx)

    def _build_polled_event(
        self,
        event: Dict[str, Any],
        config: _PollConfig,
        start_utc: datetime,
        trigger_time: datetime,
    ) -> PolledEvent:
        offset_minutes = config.offset_minutes
        if offset_minutes < 0:
            trigger_type = "before_start"
        elif offset_minutes == 0:
            trigger_type = "at_start"
        else:
            trigger_type = "after_start"

        payload = _normalize_event(event, config.calendar_id)
        payload.update({
            "trigger_type": trigger_type,
            "offset_minutes": offset_minutes,
            "trigger_time": trigger_time.isoformat(),
            "event_start": start_utc.isoformat(),
        })

        return PolledEvent(
            payload=payload,
            raw=event,
            provider_event_id=f"{event.get('id')}_{start_utc.isoformat()}",
            occurred_at=trigger_time,
        )

    def _maybe_cleanup(
        self,
        triggered_instances: Dict[str, str],
        last_cleanup: Optional[datetime],
        now: datetime,
    ) -> tuple[Dict[str, str], datetime]:
        """Prune stale instance keys once per day."""
        if last_cleanup and (now - last_cleanup).days < 1:
            return triggered_instances, last_cleanup
        cutoff = now - timedelta(days=CURSOR_CLEANUP_DAYS)
        cleaned = {
            k: v for k, v in triggered_instances.items()
            if _parse_rfc3339(v) and _parse_rfc3339(v) > cutoff  # type: ignore[operator]
        }
        return cleaned, now

    def _compute_rate_limit_hint(self, next_trigger_time: Optional[datetime], now: datetime) -> Optional[int]:
        """Return seconds to next poll, scaled to approach the next trigger without over-polling."""
        if not next_trigger_time:
            return None
        seconds_until = int((next_trigger_time - now).total_seconds())
        return max(30, min(seconds_until // 2, 300))

    async def _fetch_events(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        calendar_id: str,
        *,
        time_min: datetime,
        time_max: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch calendar events within the time window."""
        all_events: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        while True:
            params: Dict[str, Any] = {
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "singleEvents": True,  # Expand recurring events
                "maxResults": MAX_EVENTS_PER_POLL,
                "orderBy": "startTime",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(
                f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                headers=headers,
                params=params,
            )
            await self._raise_for_status(resp, "list events")

            data = resp.json()
            all_events.extend(data.get("items", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return all_events

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

    def _resolve_offset_minutes(self, ctx: PollContext) -> int:
        """Get offset_minutes from config with bounds checking."""
        config = ctx.subscription.provider_config or {}
        value = config.get("offset_minutes", -15)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = -15
        return max(-1440, min(numeric, 1440))

    def _resolve_include_all_day(self, ctx: PollContext) -> bool:
        """Get include_all_day_events from config."""
        config = ctx.subscription.provider_config or {}
        return bool(config.get("include_all_day_events", False))

    def _matches_filters(self, event: Dict[str, Any], ctx: PollContext) -> bool:
        """Check if event matches configured filters."""
        config = ctx.subscription.provider_config or {}

        # Summary contains filter (case-insensitive)
        if summary_contains := config.get("summary_contains"):
            summary = (event.get("summary") or "").lower()
            if summary_contains.lower() not in summary:
                return False

        # Organizer email filter (case-insensitive)
        if organizer_email := config.get("organizer_email"):
            event_organizer = (event.get("organizer", {}).get("email") or "").lower()
            if organizer_email.lower() != event_organizer:
                return False

        return True


register_adapter(GoogleCalendarEventStartAdapter())
