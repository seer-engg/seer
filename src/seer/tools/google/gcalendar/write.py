"""
Google Calendar write operations - creating, updating, and deleting events.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gcalendar.helpers import (
    CALENDAR_API_BASE,
    CALENDAR_EVENT_SCHEMA,
    CALENDAR_ID_PARAM_SCHEMA,
    CALENDAR_ID_RESOURCE_PICKER,
    EVENT_ID_PARAM_SCHEMA,
    GCALENDAR_INTEGRATION_TYPE,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.gcalendar.write")


def _build_event_time(
    datetime_str: Optional[str],
    date_str: Optional[str],
    timezone: Optional[str],
) -> Optional[Dict[str, str]]:
    """Build event time object for start/end."""
    if datetime_str:
        result: Dict[str, str] = {"dateTime": datetime_str}
        if timezone:
            result["timeZone"] = timezone
        return result
    if date_str:
        return {"date": date_str}
    return None


class GoogleCalendarCreateEventTool(GoogleAPIClient):
    """Create a new event in a Google Calendar."""

    name = "google_calendar_create_event"
    description = "Create a new event in a Google Calendar."
    required_scopes = ["https://www.googleapis.com/auth/calendar.events"]
    integration_type = GCALENDAR_INTEGRATION_TYPE

    def get_resource_pickers(self) -> Dict[str, Any]:
        return CALENDAR_ID_RESOURCE_PICKER.copy()

    def get_output_schema(self) -> Dict[str, Any]:
        return CALENDAR_EVENT_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID to create event in (default: 'primary')",
                    "default": "primary",
                },
                "summary": {
                    "type": "string",
                    "description": "Event title",
                },
                "description": {
                    "type": "string",
                    "description": "Event description",
                },
                "location": {
                    "type": "string",
                    "description": "Event location",
                },
                "start_datetime": {
                    "type": "string",
                    "description": "Start time (RFC3339, e.g., '2026-01-15T09:00:00-05:00')",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date for all-day events (YYYY-MM-DD)",
                },
                "end_datetime": {
                    "type": "string",
                    "description": "End time (RFC3339, e.g., '2026-01-15T10:00:00-05:00')",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date for all-day events (YYYY-MM-DD)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone (e.g., 'America/New_York')",
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "Attendee email"},
                            "optional": {"type": "boolean", "description": "Is attendance optional"},
                        },
                        "required": ["email"],
                    },
                    "description": "List of attendees",
                },
                "send_updates": {
                    "type": "string",
                    "description": "Notification mode: 'all', 'externalOnly', or 'none'",
                    "enum": ["all", "externalOnly", "none"],
                    "default": "none",
                },
            },
            "required": ["summary"],
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
        summary = arguments.get("summary")
        if not summary:
            raise ValueError("summary is required")

        # Build event body
        event_body: Dict[str, Any] = {"summary": summary}

        if arguments.get("description"):
            event_body["description"] = arguments["description"]
        if arguments.get("location"):
            event_body["location"] = arguments["location"]

        # Build start/end times
        start = _build_event_time(
            arguments.get("start_datetime"),
            arguments.get("start_date"),
            arguments.get("timezone"),
        )
        end = _build_event_time(
            arguments.get("end_datetime"),
            arguments.get("end_date"),
            arguments.get("timezone"),
        )

        if start:
            event_body["start"] = start
        if end:
            event_body["end"] = end

        # Add attendees
        if arguments.get("attendees"):
            event_body["attendees"] = arguments["attendees"]

        params: Dict[str, Any] = {}
        if arguments.get("send_updates"):
            params["sendUpdates"] = arguments["send_updates"]

        logger.info(
            "Creating Google Calendar event: calendar=%s, summary='%s'",
            calendar_id,
            summary[:50] if summary else "",
        )

        resp = await self._make_request(
            "POST",
            f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
            access_token,
            json_body=event_body,
            params=params if params else None,
        )
        return resp.json()


class GoogleCalendarUpdateEventTool(GoogleAPIClient):
    """Update an existing event in a Google Calendar."""

    name = "google_calendar_update_event"
    description = "Update an existing event in a Google Calendar."
    required_scopes = ["https://www.googleapis.com/auth/calendar.events"]
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
                "summary": {
                    "type": "string",
                    "description": "New event title",
                },
                "description": {
                    "type": "string",
                    "description": "New event description",
                },
                "location": {
                    "type": "string",
                    "description": "New event location",
                },
                "start_datetime": {
                    "type": "string",
                    "description": "New start time (RFC3339)",
                },
                "start_date": {
                    "type": "string",
                    "description": "New start date for all-day events (YYYY-MM-DD)",
                },
                "end_datetime": {
                    "type": "string",
                    "description": "New end time (RFC3339)",
                },
                "end_date": {
                    "type": "string",
                    "description": "New end date for all-day events (YYYY-MM-DD)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone",
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "optional": {"type": "boolean"},
                        },
                        "required": ["email"],
                    },
                    "description": "Updated list of attendees (replaces existing)",
                },
                "send_updates": {
                    "type": "string",
                    "description": "Notification mode: 'all', 'externalOnly', or 'none'",
                    "enum": ["all", "externalOnly", "none"],
                    "default": "none",
                },
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

        # Build patch body with only provided fields
        event_body: Dict[str, Any] = {}

        if "summary" in arguments and arguments["summary"] is not None:
            event_body["summary"] = arguments["summary"]
        if "description" in arguments and arguments["description"] is not None:
            event_body["description"] = arguments["description"]
        if "location" in arguments and arguments["location"] is not None:
            event_body["location"] = arguments["location"]

        # Build start/end times if provided
        start = _build_event_time(
            arguments.get("start_datetime"),
            arguments.get("start_date"),
            arguments.get("timezone"),
        )
        end = _build_event_time(
            arguments.get("end_datetime"),
            arguments.get("end_date"),
            arguments.get("timezone"),
        )

        if start:
            event_body["start"] = start
        if end:
            event_body["end"] = end

        if "attendees" in arguments and arguments["attendees"] is not None:
            event_body["attendees"] = arguments["attendees"]

        params: Dict[str, Any] = {}
        if arguments.get("send_updates"):
            params["sendUpdates"] = arguments["send_updates"]

        logger.info(
            "Updating Google Calendar event: calendar=%s, event=%s",
            calendar_id,
            event_id,
        )

        resp = await self._make_request(
            "PATCH",
            f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
            access_token,
            json_body=event_body,
            params=params if params else None,
        )
        return resp.json()


class GoogleCalendarDeleteEventTool(GoogleAPIClient):
    """Delete an event from a Google Calendar."""

    name = "google_calendar_delete_event"
    description = "Delete an event from a Google Calendar."
    required_scopes = ["https://www.googleapis.com/auth/calendar.events"]
    integration_type = GCALENDAR_INTEGRATION_TYPE

    def get_resource_pickers(self) -> Dict[str, Any]:
        return CALENDAR_ID_RESOURCE_PICKER.copy()

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "event_id": {"type": "string"},
            },
            "required": ["success", "event_id"],
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calendar_id": CALENDAR_ID_PARAM_SCHEMA,
                "event_id": EVENT_ID_PARAM_SCHEMA,
                "send_updates": {
                    "type": "string",
                    "description": "Notification mode: 'all', 'externalOnly', or 'none'",
                    "enum": ["all", "externalOnly", "none"],
                    "default": "none",
                },
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

        params: Dict[str, Any] = {}
        if arguments.get("send_updates"):
            params["sendUpdates"] = arguments["send_updates"]

        logger.info(
            "Deleting Google Calendar event: calendar=%s, event=%s",
            calendar_id,
            event_id,
        )

        await self._make_request(
            "DELETE",
            f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
            access_token,
            params=params if params else None,
        )

        return {"success": True, "event_id": event_id}
