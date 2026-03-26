"""
Google Calendar polling adapter for event changes.

Uses the Calendar API's syncToken mechanism for efficient incremental sync.
Emits events when calendar events are created or updated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from seer.core.triggers.polling.adapters.base import (
    PollAdapterError,
    PollContext,
    PolledEvent,
    PollResult,
    register_adapter,
)
from seer.core.triggers.polling.adapters.google_calendar_common import (
    CALENDAR_API_BASE,
    GoogleCalendarBaseAdapter,
    normalize_event,
    parse_rfc3339,
)
from seer.logger import get_logger

logger = get_logger(__name__)

MAX_EVENTS_PER_POLL = 50


def _is_newly_created(event: Dict[str, Any]) -> bool:
    """
    Determine if an event was newly created vs updated.

    Google Calendar doesn't distinguish between create and update in the sync response.
    We detect creation by checking if 'created' and 'updated' timestamps are within 1 second.
    """
    created_str = event.get("created")
    updated_str = event.get("updated")

    if not created_str or not updated_str:
        return False

    created_dt = parse_rfc3339(created_str)
    updated_dt = parse_rfc3339(updated_str)

    if not created_dt or not updated_dt:
        return False

    # If created and updated are within 1 second, consider it a new event
    diff = abs((updated_dt - created_dt).total_seconds())
    return diff < 1.0


def _normalize_event_with_type(event: Dict[str, Any], calendar_id: str) -> Dict[str, Any]:
    """Normalize event and add event_type field for change detection."""
    payload = normalize_event(event, calendar_id)
    payload["event_type"] = "created" if _is_newly_created(event) else "updated"
    return payload


class GoogleCalendarEventChangedAdapter(GoogleCalendarBaseAdapter):
    """
    Poll Google Calendar for created/updated events using syncToken strategy.

    The syncToken mechanism is more efficient than timestamp-based polling:
    - Initial sync fetches all events and returns a syncToken
    - Subsequent polls use the syncToken to get only changes
    - HTTP 410 Gone indicates token expiry, requiring a full resync
    """

    trigger_key = "poll.google_calendar.event_changed"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """
        Perform initial full sync to obtain a syncToken.

        This doesn't emit any events - it just establishes the baseline.
        """
        calendar_id = self._resolve_calendar_id(ctx)
        headers = {"Authorization": f"Bearer {ctx.access_token}", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                sync_token = await self._perform_full_sync(client, headers, calendar_id)
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Failed to bootstrap Google Calendar cursor")
            raise PollAdapterError(
                "Failed to initialize Google Calendar sync",
                detail={"error": str(exc)}
            ) from exc

        return {"sync_token": sync_token, "calendar_id": calendar_id}

    async def _perform_full_sync(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        calendar_id: str,
    ) -> str:
        """
        Perform full sync to get initial syncToken.

        Paginates through all events without emitting them.
        """
        page_token: Optional[str] = None

        while True:
            params: Dict[str, Any] = {
                "maxResults": 250,  # Max allowed for full sync
                "singleEvents": True,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(
                f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                headers=headers,
                params=params,
            )
            await self._raise_for_status(resp, "full sync")

            data = resp.json()
            page_token = data.get("nextPageToken")

            if not page_token:
                # Last page - return the syncToken
                sync_token = data.get("nextSyncToken")
                if not sync_token:
                    raise PollAdapterError(
                        "Google Calendar API did not return syncToken",
                        detail={"response_keys": list(data.keys())},
                    )
                return sync_token

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """
        Poll for calendar event changes using syncToken.

        Returns:
            PollResult with normalized events and updated cursor.
        """
        sync_token = cursor.get("sync_token")
        calendar_id = cursor.get("calendar_id") or self._resolve_calendar_id(ctx)

        if not sync_token:
            logger.warning("No syncToken in cursor, bootstrapping...")
            return PollResult(events=[], cursor=await self.bootstrap_cursor(ctx))

        headers = {"Authorization": f"Bearer {ctx.access_token}", "Accept": "application/json"}
        params: Dict[str, Any] = {
            "syncToken": sync_token,
            "maxResults": self._resolve_max_results(ctx),
            "singleEvents": True,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                    headers=headers,
                    params=params,
                )

                # Handle 410 Gone - syncToken expired, need full resync
                if resp.status_code == 410:
                    logger.info("Google Calendar syncToken expired, performing full resync")
                    resync_token = await self._perform_full_sync(client, headers, calendar_id)
                    return PollResult(events=[], cursor={"sync_token": resync_token, "calendar_id": calendar_id})

                await self._raise_for_status(resp, "incremental sync")
                data = resp.json()

                polled_events = self._process_events(data.get("items") or [], calendar_id)
                return self._build_poll_result(polled_events, data, sync_token, calendar_id)

        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Google Calendar polling failure")
            raise PollAdapterError("Unexpected Google Calendar polling failure", detail={"error": str(exc)}) from exc

    def _process_events(self, events: List[Dict[str, Any]], calendar_id: str) -> List[PolledEvent]:
        """Process raw calendar events into PolledEvent objects."""
        polled_events: List[PolledEvent] = []
        for event in events:
            if event.get("status") == "cancelled":
                continue
            polled_events.append(
                PolledEvent(
                    payload=_normalize_event_with_type(event, calendar_id),
                    raw=event,
                    provider_event_id=event.get("id"),
                    occurred_at=parse_rfc3339(event.get("updated")) or datetime.now(timezone.utc),
                )
            )
        return polled_events

    def _build_poll_result(
        self, events: List[PolledEvent], data: Dict[str, Any], sync_token: str, calendar_id: str
    ) -> PollResult:
        """Build PollResult with appropriate cursor based on response."""
        new_sync_token = data.get("nextSyncToken")
        if new_sync_token:
            return PollResult(events=events, cursor={"sync_token": new_sync_token, "calendar_id": calendar_id}, has_more=False)
        page_token = data.get("nextPageToken")
        return PollResult(
            events=events,
            cursor={"sync_token": sync_token, "page_token": page_token, "calendar_id": calendar_id},
            has_more=bool(page_token),
        )

    def _resolve_max_results(self, ctx: PollContext) -> int:
        """Get max_results from config with bounds checking."""
        config = ctx.subscription.provider_config or {}
        value = config.get("max_results", MAX_EVENTS_PER_POLL)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = MAX_EVENTS_PER_POLL
        return max(1, min(numeric, MAX_EVENTS_PER_POLL))


register_adapter(GoogleCalendarEventChangedAdapter())
