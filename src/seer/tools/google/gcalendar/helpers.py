"""
Shared helpers and constants for Google Calendar tools.
"""

from typing import Any, Dict

# Google Calendar API base URL
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


# Calendar event schema for output
CALENDAR_EVENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Event ID"},
        "status": {"type": "string", "description": "Event status (confirmed, tentative, cancelled)"},
        "htmlLink": {"type": "string", "description": "URL to view event in Google Calendar"},
        "created": {"type": "string", "description": "Creation timestamp (RFC3339)"},
        "updated": {"type": "string", "description": "Last modification timestamp (RFC3339)"},
        "summary": {"type": "string", "description": "Event title"},
        "description": {"type": "string", "description": "Event description"},
        "location": {"type": "string", "description": "Event location"},
        "start": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date for all-day events (YYYY-MM-DD)"},
                "dateTime": {"type": "string", "description": "Start time (RFC3339)"},
                "timeZone": {"type": "string", "description": "IANA timezone"},
            },
        },
        "end": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date for all-day events (YYYY-MM-DD)"},
                "dateTime": {"type": "string", "description": "End time (RFC3339)"},
                "timeZone": {"type": "string", "description": "IANA timezone"},
            },
        },
        "organizer": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "displayName": {"type": "string"},
                "self": {"type": "boolean"},
            },
        },
        "attendees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "displayName": {"type": "string"},
                    "responseStatus": {"type": "string"},
                    "optional": {"type": "boolean"},
                },
            },
        },
        "recurringEventId": {"type": "string", "description": "Parent recurring event ID"},
        "iCalUID": {"type": "string", "description": "iCalendar UID"},
    },
    "additionalProperties": True,
}


CALENDAR_EVENTS_LIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": CALENDAR_EVENT_SCHEMA,
        },
        "nextPageToken": {"type": "string"},
        "nextSyncToken": {"type": "string"},
    },
    "additionalProperties": True,
}
