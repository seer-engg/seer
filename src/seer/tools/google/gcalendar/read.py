"""
Google Calendar read operations - listing and getting events.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gcalendar.helpers import (
    CALENDAR_API_BASE,
    CALENDAR_EVENT_SCHEMA,
    CALENDAR_EVENTS_LIST_SCHEMA,
    CALENDAR_ID_PARAM_SCHEMA,
    CALENDAR_ID_RESOURCE_PICKER,
    EVENT_ID_PARAM_SCHEMA,
    GCALENDAR_INTEGRATION_TYPE,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.gcalendar.read")


class GoogleCalendarListEventsTool(GoogleAPIClient):
    """List events from a Google Calendar with filtering."""

    name = "google_calendar_list_events"
    description = "List events from a Google Calendar. Supports filtering by time range and search query."
    required_scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
    integration_type = GCALENDAR_INTEGRATION_TYPE

    def get_resource_pickers(self) -> Dict[str, Any]:
        return CALENDAR_ID_RESOURCE_PICKER.copy()

    def get_output_schema(self) -> Dict[str, Any]:
        return CALENDAR_EVENTS_LIST_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID to list events from (default: 'primary')",
                    "default": "primary",
                },
                "time_min": {
                    "type": "string",
                    "description": "Start of time range (RFC3339, e.g., '2026-01-01T00:00:00Z')",
                },
                "time_max": {
                    "type": "string",
                    "description": "End of time range (RFC3339, e.g., '2026-12-31T23:59:59Z')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of events to return (default: 50, max: 250)",
                    "minimum": 1,
                    "maximum": 250,
                    "default": 50,
                },
                "q": {
                    "type": "string",
                    "description": "Free text search query to filter events",
                },
                "single_events": {
                    "type": "boolean",
                    "description": "Expand recurring events into instances (default: true)",
                    "default": True,
                },
                "order_by": {
                    "type": "string",
                    "description": "Order of events: 'startTime' (requires singleEvents=true) or 'updated'",
                    "enum": ["startTime", "updated"],
                    "default": "startTime",
                },
                "page_token": {
                    "type": "string",
                    "description": "Token for pagination",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context  # unused but required for interface consistency

        calendar_id = arguments.get("calendar_id", "primary") or "primary"
        max_results = min(max(1, int(arguments.get("max_results", 50) or 50)), 250)
        single_events = arguments.get("single_events", True)

        params: Dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": single_events,
        }

        if arguments.get("time_min"):
            params["timeMin"] = arguments["time_min"]
        if arguments.get("time_max"):
            params["timeMax"] = arguments["time_max"]
        if arguments.get("q"):
            params["q"] = arguments["q"]
        if arguments.get("order_by"):
            params["orderBy"] = arguments["order_by"]
        if arguments.get("page_token"):
            params["pageToken"] = arguments["page_token"]

        logger.info(
            "Listing Google Calendar events: calendar=%s, max_results=%s",
            calendar_id,
            max_results,
        )

        resp = await self._make_request(
            "GET",
            f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
            access_token,
            params=params,
        )
        return resp.json()


class GoogleCalendarGetEventTool(GoogleAPIClient):
    """Get a single Google Calendar event by ID."""

    name = "google_calendar_get_event"
    description = "Get a single event from a Google Calendar by event ID."
    required_scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
    integration_type = GCALENDAR_INTEGRATION_TYPE

    def get_resource_pickers(self) -> Dict[str, Any]:
        return CALENDAR_ID_RESOURCE_PICKER.copy()

    def get_output_schema(self) -> Dict[str, Any]:
        return CALENDAR_EVENT_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calendar_id": CALENDAR_ID_PARAM_SCHEMA,
                "event_id": EVENT_ID_PARAM_SCHEMA,
            },
            "required": ["event_id"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context  # unused but required for interface consistency

        calendar_id = arguments.get("calendar_id", "primary") or "primary"
        event_id = arguments.get("event_id")
        if not event_id:
            raise ValueError("event_id is required")

        logger.info(
            "Getting Google Calendar event: calendar=%s, event=%s",
            calendar_id,
            event_id,
        )

        resp = await self._make_request(
            "GET",
            f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
            access_token,
        )
        return resp.json()
