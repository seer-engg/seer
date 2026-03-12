"""
In-memory registry describing workflow trigger metadata and schemas.
"""
# pylint: disable=too-many-lines  # Registry contains schema definitions for all builtin triggers

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, MutableMapping, Optional

from seer.core.schema.models import JsonSchema
from seer.core.schema.models import TriggerDefinition, TriggerSchemas, TriggerMetadata



class TriggerRegistry:
    """Stores trigger definitions and their schemas."""

    def __init__(self, initial: MutableMapping[str, TriggerDefinition] | None = None) -> None:
        self._triggers: Dict[str, TriggerDefinition] = dict(initial or {})

    def register(self, trigger: TriggerDefinition) -> None:
        self._triggers[trigger.key] = trigger

    def get(self, key: str) -> TriggerDefinition:
        try:
            return self._triggers[key]
        except KeyError as exc:
            raise KeyError(f"Trigger '{key}' is not registered") from exc

    def maybe_get(self, key: str) -> Optional[TriggerDefinition]:
        return self._triggers.get(key)

    def all(self) -> List[TriggerDefinition]:
        return list(self._triggers.values())


def _default_event_envelope_schema() -> JsonSchema:
    """Canonical schema shared by webhook-based triggers."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "trigger_key": {"type": "string"},
            "provider": {"type": "string"},
            "account_id": {"type": ["string", "integer", "null"]},
            "occurred_at": {"type": "string", "format": "date-time"},
            "received_at": {"type": "string", "format": "date-time"},
            "data": {"type": "object"},
            "raw": {"type": ["object", "array", "null"]},
        },
        "required": ["id", "trigger_key", "provider", "occurred_at", "data"],
    }


def _enveloped_event_schema(payload_schema: JsonSchema) -> JsonSchema:
    """Wrap a payload schema in the standard event envelope."""

    envelope = _default_event_envelope_schema()
    envelope["properties"]["data"] = deepcopy(payload_schema)
    return envelope


POLLING_TRIGGERS = [
    "poll.airtable.new_record_in_view",
    "poll.gmail.email_received",
    "poll.discord.message_received",
    "poll.slack.message_received",
    "poll.google_calendar.event_changed",
    "schedule.cron",
]


def _webhook_generic_config_schema() -> JsonSchema:
    """
    Config schema for generic webhooks.

    Generic webhooks don't require any provider configuration.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "description": "Generic webhooks do not require provider configuration.",
    }


def _oauth_connection_property() -> Dict[str, Any]:
    """
    Common property for specifying which OAuth account to use.

    OAuth-based triggers (Gmail, Slack, Google Calendar, Discord) may have multiple
    accounts connected. This property allows specifying which account to use.
    """
    return {
        "provider_connection_id": {
            "type": "integer",
            "description": (
                "OAuth connection ID when user has multiple accounts connected. "
                "Required when get_trigger_accounts() returns requires_selection: true."
            ),
        }
    }


def _airtable_new_record_in_view_payload_schema() -> JsonSchema:
    """Payload schema for Airtable new record in view trigger."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "record_id": {"type": "string", "description": "Airtable record ID"},
            "created_time": {"type": ["string", "null"], "format": "date-time", "description": "Record creation timestamp"},
            "fields": {"type": "object", "additionalProperties": True, "description": "Record field values"},
            "base_id": {"type": "string", "description": "Airtable base ID"},
            "table_id": {"type": "string", "description": "Airtable table ID"},
            "view_id": {"type": "string", "description": "Airtable view ID being monitored"},
        },
        "required": ["record_id", "fields"],
    }


def _airtable_new_record_in_view_config_schema() -> JsonSchema:
    """Config schema for Airtable new record in view trigger."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "base_id": {
                "type": "string",
                "description": "Airtable base ID (e.g., 'appXXXXXXXXXXXXXX')",
                "x-resource-picker": {
                    "provider": "airtable",
                    "resource_type": "base",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                },
            },
            "table_id": {
                "type": "string",
                "description": "Table ID within the base (e.g., 'tblXXXXXXXXXXXXXX')",
                "x-resource-picker": {
                    "provider": "airtable",
                    "resource_type": "table",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "depends_on": "base_id",
                },
            },
            "view_id": {
                "type": "string",
                "description": "View ID to monitor for new records",
                "x-resource-picker": {
                    "provider": "airtable",
                    "resource_type": "view",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "depends_on": "table_id",
                },
            },
            "max_records": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 100,
                "description": "Maximum records to fetch per poll (for cursor state management).",
            },
            **_oauth_connection_property(),
        },
        "required": ["base_id", "table_id", "view_id"],
    }


def _airtable_new_record_in_view_sample_event() -> Dict[str, Any]:
    """Sample event for Airtable new record in view trigger."""
    payload = {
        "record_id": "recABC123XYZ456",
        "created_time": "2026-02-27T10:00:00.000Z",
        "fields": {
            "Name": "New Task",
            "Status": "To Do",
            "Priority": "High",
            "Assignee": "John Doe",
        },
        "base_id": "appXXXXXXXXXXXXXX",
        "table_id": "tblYYYYYYYYYYYYYY",
        "view_id": "viwZZZZZZZZZZZZZZ",
    }
    return {
        "id": "evt_sample_poll_airtable_new_record_in_view",
        "trigger_key": "poll.airtable.new_record_in_view",
        "provider": "airtable",
        "account_id": None,
        "occurred_at": "2026-02-27T10:00:00Z",
        "received_at": "2026-02-27T10:00:05Z",
        "data": payload,
        "raw": {"id": "recABC123XYZ456", "createdTime": "2026-02-27T10:00:00.000Z", "fields": payload["fields"]},
    }


def _register_builtin_triggers(registry: TriggerRegistry) -> None:
    registry.register(
        TriggerDefinition(
            key="webhook.generic",
            title="Webhook",
            provider="generic",
            mode="webhook",
            description="Accepts arbitrary JSON payloads via signed webhook requests.",
            schemas=TriggerSchemas(
                event=_default_event_envelope_schema(),
                config=_webhook_generic_config_schema(),
            ),
            meta=TriggerMetadata(
                requires_connection=False,
            ),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.gmail.email_received",
            title="Gmail",
            provider="gmail",
            mode="polling",
            description="Poll a Gmail inbox for newly received messages using OAuth credentials.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_gmail_email_received_payload_schema()),
                config=_gmail_email_received_config_schema(),
            ),
            meta=TriggerMetadata(sample_event=_gmail_email_received_sample_event()),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.airtable.new_record_in_view",
            title="Airtable View",
            provider="airtable",
            mode="polling",
            description="Poll an Airtable view for new records (newly created or moved into view).",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_airtable_new_record_in_view_payload_schema()),
                config=_airtable_new_record_in_view_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_airtable_new_record_in_view_sample_event(),
                requires_connection=True,
                required_scopes=["data.records:read", "schema.bases:read"],
            ),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.google_calendar.event_changed",
            title="Google Calendar",
            provider="google",
            mode="polling",
            description="Poll a Google Calendar for newly created or updated events.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_google_calendar_event_changed_payload_schema()),
                config=_google_calendar_event_changed_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_google_calendar_event_changed_sample_event(),
                requires_connection=True,
                required_scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            ),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.discord.message_received",
            title="Discord",
            provider="discord",
            mode="polling",
            description="Poll a Discord channel for newly received messages using bot credentials.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_discord_message_received_payload_schema()),
                config=_discord_message_received_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_discord_message_received_sample_event(),
                requires_connection=True,
            ),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.slack.message_received",
            title="Slack",
            provider="slack",
            mode="polling",
            description="Poll a Slack channel for newly received messages using OAuth credentials.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_slack_message_received_payload_schema()),
                config=_slack_message_received_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_slack_message_received_sample_event(),
                requires_connection=True,
                required_scopes=[
                    "channels:history",  # Public channel messages
                    "groups:history",    # Private channel messages
                ],
            ),
        )
    )
    registry.register(
        TriggerDefinition(
            key="schedule.cron",
            title="Scheduler",
            provider="schedule",
            mode="polling",
            description="Execute workflow on a cron schedule with timezone support.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_cron_schedule_payload_schema()),
                config=_cron_schedule_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_cron_schedule_sample_event(),
                requires_connection=False,
            ),
        )
    )

    registry.register(
        TriggerDefinition(
            key="webhook.supabase.db_changes",
            title="Supabase",
            provider="supabase",
            mode="webhook",
            description="Receive real-time webhooks when rows are inserted, updated, or deleted in Supabase tables.",
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_supabase_db_changes_payload_schema()),
                config=_supabase_db_changes_config_schema(),
            ),
            meta=TriggerMetadata(sample_event=_supabase_db_changes_sample_event()),
        )
    )

    registry.register(
        TriggerDefinition(
            key="form.hosted",
            title="Form",
            provider="form",
            mode="webhook",
            description=(
                "Create a public form with custom fields that non-technical users can "
                "fill out to trigger workflows."
            ),
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_form_hosted_payload_schema()),
                config=_form_trigger_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_form_hosted_sample_event(),
                requires_connection=False,
            ),
        )
    )

    registry.register(
        TriggerDefinition(
            key="form.hitl",
            title="HITL Form",
            provider="form",
            mode="webhook",
            description=(
                "Auto-generated form for Human-in-the-Loop workflow responses. "
                "Created when a workflow hits an HITL node with Gmail delivery channel."
            ),
            schemas=TriggerSchemas(
                event=_enveloped_event_schema(_form_hosted_payload_schema()),
                config=_form_trigger_config_schema(),
            ),
            meta=TriggerMetadata(
                sample_event=_form_hitl_sample_event(),
                requires_connection=False,
            ),
        )
    )


def _gmail_email_received_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message_id": {"type": "string"},
            "thread_id": {"type": "string"},
            "internal_date_ms": {"type": "integer"},
            "snippet": {"type": ["string", "null"]},
            "subject": {"type": ["string", "null"]},
            "from": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                },
            },
            "to": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                    },
                },
            },
            "labels": {"type": "array", "items": {"type": "string"}},
            "date_header": {"type": ["string", "null"]},
            "history_id": {"type": ["string", "null"]},
        },
        "required": ["message_id", "thread_id", "internal_date_ms"],
    }


def _gmail_email_received_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to specific Gmail label IDs (defaults to INBOX).",
            },
            "query": {
                "type": "string",
                "description": "Optional Gmail search query appended to the poll watermark (e.g., 'is:unread').",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "default": 25,
                "description": "Maximum messages to examine per poll cycle (capped at 25).",
            },
            "overlap_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 900000,
                "default": 300000,
                "description": (
                    "Overlap window in milliseconds to re-read recent messages for "
                    "dedupe safety."
                ),
            },
            **_oauth_connection_property(),
        },
    }


def _gmail_email_received_sample_event() -> Dict[str, Any]:
    payload = {
        "message_id": "18c123example",
        "thread_id": "18c123example",
        "internal_date_ms": 1735630123456,
        "snippet": "Reminder about tomorrow's demo",
        "subject": "Demo tomorrow?",
        "from": {"name": "Product Team", "email": "product@example.com"},
        "to": [{"name": "You", "email": "you@example.com"}],
        "labels": ["INBOX", "UNREAD"],
        "date_header": "Fri, 13 Dec 2025 10:00:00 -0000",
        "history_id": "123456",
    }
    return {
        "id": "evt_sample_poll_gmail_email_received",
        "trigger_key": "poll.gmail.email_received",
        "provider": "gmail",
        "account_id": None,
        "occurred_at": "2025-12-13T10:00:00Z",
        "received_at": "2025-12-13T10:00:05Z",
        "data": payload,
        "raw": {"payload": payload},
    }


def _google_calendar_event_changed_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string", "description": "Google Calendar event ID"},
            "event_type": {
                "type": "string",
                "enum": ["created", "updated"],
                "description": "Whether the event was newly created or updated",
            },
            "calendar_id": {"type": "string", "description": "Calendar being monitored"},
            "summary": {"type": ["string", "null"], "description": "Event title"},
            "description": {"type": ["string", "null"], "description": "Event description"},
            "location": {"type": ["string", "null"], "description": "Event location"},
            "status": {"type": ["string", "null"], "description": "Event status (confirmed/tentative/cancelled)"},
            "html_link": {"type": ["string", "null"], "description": "URL to view in Google Calendar"},
            "start": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "datetime": {"type": ["string", "null"], "description": "Start time (RFC3339 or date)"},
                    "timezone": {"type": ["string", "null"], "description": "IANA timezone"},
                },
            },
            "end": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "datetime": {"type": ["string", "null"], "description": "End time (RFC3339 or date)"},
                    "timezone": {"type": ["string", "null"], "description": "IANA timezone"},
                },
            },
            "organizer": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "email": {"type": ["string", "null"]},
                    "display_name": {"type": ["string", "null"]},
                    "self": {"type": "boolean"},
                },
            },
            "attendees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "email": {"type": ["string", "null"]},
                        "display_name": {"type": ["string", "null"]},
                        "response_status": {"type": ["string", "null"]},
                        "optional": {"type": "boolean"},
                        "self": {"type": "boolean"},
                    },
                },
            },
            "is_all_day": {"type": "boolean", "description": "Whether this is an all-day event"},
            "recurring_event_id": {"type": ["string", "null"], "description": "Parent event if recurring instance"},
            "created": {"type": ["string", "null"], "description": "Event creation timestamp (RFC3339)"},
            "updated": {"type": ["string", "null"], "description": "Event last modified timestamp (RFC3339)"},
        },
        "required": ["event_id", "event_type", "calendar_id"],
    }


def _google_calendar_event_changed_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID to monitor for events (defaults to primary calendar).",
                "default": "primary",
                "x-resource-picker": {
                    "provider": "google_calendar",
                    "resource_type": "google_calendar",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": False,
                },
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 50,
                "description": "Maximum events to examine per poll cycle (capped at 50).",
            },
            **_oauth_connection_property(),
        },
    }


def _google_calendar_event_changed_sample_event() -> Dict[str, Any]:
    payload = {
        "event_id": "abc123def456",
        "event_type": "created",
        "calendar_id": "primary",
        "summary": "Team Meeting",
        "description": "Weekly sync with the engineering team",
        "location": "Conference Room A",
        "status": "confirmed",
        "html_link": "https://www.google.com/calendar/event?eid=abc123",
        "start": {"datetime": "2026-01-15T10:00:00-05:00", "timezone": "America/New_York"},
        "end": {"datetime": "2026-01-15T11:00:00-05:00", "timezone": "America/New_York"},
        "organizer": {"email": "organizer@example.com", "display_name": "Jane Doe", "self": False},
        "attendees": [
            {"email": "attendee@example.com", "display_name": "John Smith", "response_status": "accepted", "optional": False, "self": True},
        ],
        "is_all_day": False,
        "recurring_event_id": None,
        "created": "2026-01-10T09:00:00Z",
        "updated": "2026-01-10T09:00:00Z",
    }
    return {
        "id": "evt_sample_poll_google_calendar_event_changed",
        "trigger_key": "poll.google_calendar.event_changed",
        "provider": "google",
        "account_id": None,
        "occurred_at": "2026-01-10T09:00:00Z",
        "received_at": "2026-01-10T09:00:05Z",
        "data": payload,
        "raw": {"payload": payload},
    }


def _discord_message_received_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message_id": {"type": "string"},
            "channel_id": {"type": "string"},
            "guild_id": {"type": ["string", "null"]},
            "author": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "username": {"type": "string"},
                    "discriminator": {"type": ["string", "null"]},
                    "bot": {"type": "boolean"},
                },
                "required": ["id", "username", "bot"],
            },
            "content": {"type": ["string", "null"]},
            "timestamp": {"type": "string"},
            "edited_timestamp": {"type": ["string", "null"]},
            "mentions_bot": {"type": "boolean"},
            "attachments": {"type": "array", "items": {"type": "object"}},
            "embeds": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["message_id", "channel_id", "author", "timestamp", "mentions_bot"],
    }


def _discord_message_received_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "guild_id": {
                "type": "string",
                "description": "Discord server (guild) ID to monitor for new messages.",
                "x-resource-picker": {
                    "provider": "discord",
                    "resource_type": "guild",
                    "display_field": "name",
                    "value_field": "resource_id",
                    "search_enabled": True,
                },
            },
            "channel_id": {
                "type": "string",
                "description": "Discord channel ID within the guild to monitor for new messages.",
                "x-resource-picker": {
                    "provider": "discord",
                    "resource_type": "channel",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "depends_on": "guild_id",
                },
            },
            "include_bot_messages": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include messages sent by bots (defaults to false).",
            },
            "only_mentions": {
                "type": "boolean",
                "default": False,
                "description": "Only trigger if the bot is explicitly mentioned in the message (defaults to false).",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
                "description": "Maximum number of messages to examine per poll cycle (capped at 100).",
            },
            **_oauth_connection_property(),
        },
        "required": ["guild_id", "channel_id"],
    }


def _discord_message_received_sample_event() -> Dict[str, Any]:
    payload = {
        "message_id": "1234567890123456789",
        "channel_id": "9876543210987654321",
        "guild_id": "1111111111111111111",
        "author": {
            "id": "2222222222222222222",
            "username": "johndoe",
            "discriminator": "1234",
            "bot": False,
        },
        "content": "Hello, world!",
        "timestamp": "2025-12-13T10:00:00.000000+00:00",
        "edited_timestamp": None,
        "mentions_bot": False,
        "attachments": [],
        "embeds": [],
    }
    return {
        "id": "evt_sample_poll_discord_message_received",
        "trigger_key": "poll.discord.message_received",
        "provider": "discord",
        "account_id": None,
        "occurred_at": "2025-12-13T10:00:00Z",
        "received_at": "2025-12-13T10:00:05Z",
        "data": payload,
        "raw": payload,
    }


def _slack_message_received_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message_id": {"type": "string"},
            "channel_id": {"type": "string"},
            "workspace_id": {"type": ["string", "null"]},
            "user": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": ["string", "null"]},
                    "username": {"type": ["string", "null"]},
                    "is_bot": {"type": "boolean"},
                },
                "required": ["id", "is_bot"],
            },
            "text": {"type": ["string", "null"]},
            "timestamp": {"type": ["string", "null"]},
            "thread_ts": {"type": ["string", "null"]},
            "mentions_bot": {"type": "boolean"},
            "attachments": {"type": "array", "items": {"type": "object"}},
            "blocks": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["message_id", "channel_id", "user", "mentions_bot"],
    }


def _slack_message_received_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Slack workspace (team) ID to monitor for new messages.",
                "x-resource-picker": {
                    "provider": "slack",
                    "resource_type": "workspace",
                    "display_field": "name",
                    "value_field": "resource_id",
                    "search_enabled": True,
                },
            },
            "channel_id": {
                "type": "string",
                "description": "Slack channel ID within the workspace to monitor for new messages.",
                "x-resource-picker": {
                    "provider": "slack",
                    "resource_type": "channel",
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "depends_on": "workspace_id",
                },
            },
            "include_bot_messages": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include messages sent by bots (defaults to false).",
            },
            "only_app_mentions": {
                "type": "boolean",
                "default": False,
                "description": "Only trigger if the app is explicitly mentioned in the message (defaults to false).",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
                "description": "Maximum number of messages to examine per poll cycle (capped at 100).",
            },
            **_oauth_connection_property(),
        },
        "required": ["workspace_id", "channel_id"],
    }


def _slack_message_received_sample_event() -> Dict[str, Any]:
    payload = {
        "message_id": "1735630123.456789",
        "channel_id": "C01234567890",
        "workspace_id": "T01234567890",
        "user": {
            "id": "U01234567890",
            "username": "johndoe",
            "is_bot": False,
        },
        "text": "Hello team!",
        "timestamp": "2025-12-13T10:00:00+00:00",
        "thread_ts": None,
        "mentions_bot": False,
        "attachments": [],
        "blocks": [],
    }
    return {
        "id": "evt_sample_poll_slack_message_received",
        "trigger_key": "poll.slack.message_received",
        "provider": "slack",
        "account_id": None,
        "occurred_at": "2025-12-13T10:00:00Z",
        "received_at": "2025-12-13T10:00:05Z",
        "data": payload,
        "raw": payload,
    }


def _cron_schedule_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scheduled_time": {"type": "string", "format": "date-time"},
            "actual_time": {"type": "string", "format": "date-time"},
            "cron_expression": {"type": "string"},
            "timezone": {"type": "string"},
        },
        "required": ["scheduled_time", "actual_time", "cron_expression", "timezone"],
    }


def _cron_schedule_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cron_expression": {
                "type": "string",
                "description": "5-field cron expression (minute hour day month weekday)",
                "pattern": (
                    r"^(\*|([0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])|\*/[0-9]+)"
                    r"(\s+(\*|([0-9]|1[0-9]|2[0-3])|\*/[0-9]+)){1}"
                    r"(\s+(\*|([1-9]|[12][0-9]|3[01])|\*/[0-9]+)){1}"
                    r"(\s+(\*|([1-9]|1[0-2])|\*/[0-9]+)){1}"
                    r"(\s+(\*|[0-6]|\*/[0-9]+)){1}$"
                ),
            },
            "timezone": {
                "type": "string",
                "description": "IANA timezone identifier",
                "default": "America/Chicago",
                "enum": [
                    "UTC",
                    "America/New_York",
                    "America/Chicago",
                    "America/Denver",
                    "America/Los_Angeles",
                    "America/Anchorage",
                    "Pacific/Honolulu",
                    "America/Phoenix",
                    "America/Toronto",
                    "America/Vancouver",
                    "America/Mexico_City",
                    "America/Sao_Paulo",
                    "America/Buenos_Aires",
                    "Europe/London",
                    "Europe/Paris",
                    "Europe/Berlin",
                    "Europe/Amsterdam",
                    "Europe/Rome",
                    "Europe/Madrid",
                    "Europe/Moscow",
                    "Asia/Tokyo",
                    "Asia/Shanghai",
                    "Asia/Hong_Kong",
                    "Asia/Singapore",
                    "Asia/Kolkata",
                    "Asia/Dubai",
                    "Asia/Seoul",
                    "Asia/Bangkok",
                    "Asia/Jakarta",
                    "Asia/Karachi",
                    "Asia/Dhaka",
                    "Australia/Sydney",
                    "Australia/Melbourne",
                    "Australia/Perth",
                    "Pacific/Auckland",
                    "Africa/Cairo",
                    "Africa/Lagos",
                    "Africa/Johannesburg",
                    "Africa/Nairobi",
                ],
                "enumLabels": {
                    "UTC": "UTC",
                    "America/New_York": "Eastern Time (America/New_York)",
                    "America/Chicago": "Central Time (America/Chicago)",
                    "America/Denver": "Mountain Time (America/Denver)",
                    "America/Los_Angeles": "Pacific Time (America/Los_Angeles)",
                    "America/Anchorage": "Alaska Time (America/Anchorage)",
                    "Pacific/Honolulu": "Hawaii Time (Pacific/Honolulu)",
                    "America/Phoenix": "Arizona Time (America/Phoenix)",
                    "America/Toronto": "Eastern Time (America/Toronto)",
                    "America/Vancouver": "Pacific Time (America/Vancouver)",
                    "America/Mexico_City": "Central Time (America/Mexico_City)",
                    "America/Sao_Paulo": "Brasilia Time (America/Sao_Paulo)",
                    "America/Buenos_Aires": "Argentina Time (America/Buenos_Aires)",
                    "Europe/London": "GMT (Europe/London)",
                    "Europe/Paris": "Central European (Europe/Paris)",
                    "Europe/Berlin": "Central European (Europe/Berlin)",
                    "Europe/Amsterdam": "Central European (Europe/Amsterdam)",
                    "Europe/Rome": "Central European (Europe/Rome)",
                    "Europe/Madrid": "Central European (Europe/Madrid)",
                    "Europe/Moscow": "Moscow Time (Europe/Moscow)",
                    "Asia/Tokyo": "Japan Time (Asia/Tokyo)",
                    "Asia/Shanghai": "China Time (Asia/Shanghai)",
                    "Asia/Hong_Kong": "Hong Kong Time (Asia/Hong_Kong)",
                    "Asia/Singapore": "Singapore Time (Asia/Singapore)",
                    "Asia/Kolkata": "India Time (Asia/Kolkata)",
                    "Asia/Dubai": "Gulf Time (Asia/Dubai)",
                    "Asia/Seoul": "Korea Time (Asia/Seoul)",
                    "Asia/Bangkok": "Indochina Time (Asia/Bangkok)",
                    "Asia/Jakarta": "Western Indonesia (Asia/Jakarta)",
                    "Asia/Karachi": "Pakistan Time (Asia/Karachi)",
                    "Asia/Dhaka": "Bangladesh Time (Asia/Dhaka)",
                    "Australia/Sydney": "Australian Eastern (Australia/Sydney)",
                    "Australia/Melbourne": "Australian Eastern (Australia/Melbourne)",
                    "Australia/Perth": "Australian Western (Australia/Perth)",
                    "Pacific/Auckland": "New Zealand Time (Pacific/Auckland)",
                    "Africa/Cairo": "Eastern European (Africa/Cairo)",
                    "Africa/Lagos": "West Africa Time (Africa/Lagos)",
                    "Africa/Johannesburg": "South Africa Time (Africa/Johannesburg)",
                    "Africa/Nairobi": "East Africa Time (Africa/Nairobi)",
                },
            },
            "description": {
                "type": "string",
                "description": "Optional human-readable description of the schedule",
            },
        },
        "required": ["cron_expression", "timezone"],
    }


def _cron_schedule_sample_event() -> Dict[str, Any]:
    return {
        "id": "evt_sample_schedule_cron",
        "trigger_key": "schedule.cron",
        "provider": "schedule",
        "account_id": None,
        "occurred_at": "2025-01-05T10:00:00Z",
        "received_at": "2025-01-05T10:00:01Z",
        "data": {
            "scheduled_time": "2025-01-05T10:00:00Z",
            "actual_time": "2025-01-05T10:00:01Z",
            "cron_expression": "0 10 * * *",
            "timezone": "America/New_York",
        },
        "raw": None,
    }


def _supabase_db_changes_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["INSERT", "UPDATE", "DELETE"]},
            "table": {"type": "string"},
            "schema": {"type": "string"},
            "record": {"type": ["object", "null"]},
            "old_record": {"type": ["object", "null"]},
        },
        "required": ["type", "table", "schema"],
    }


def _supabase_db_changes_config_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "integration_resource_id": {
                "type": "integer",
                "description": "The Supabase project resource ID (from IntegrationResource table).",
            },
            "table": {
                "type": "string",
                "description": "The table name to monitor (e.g., 'orders', 'users').",
            },
            "schema": {
                "type": "string",
                "default": "public",
                "description": "The database schema name (defaults to 'public').",
            },
            "events": {
                "type": "array",
                "items": {"type": "string", "enum": ["INSERT", "UPDATE", "DELETE"]},
                "minItems": 1,
                "description": "Database operations to trigger on.",
            },
        },
        "required": ["integration_resource_id", "table", "events"],
    }


def _supabase_db_changes_sample_event() -> Dict[str, Any]:
    payload = {
        "type": "INSERT",
        "table": "orders",
        "schema": "public",
        "record": {
            "id": 123,
            "user_id": 456,
            "total": 99.99,
            "status": "pending",
            "created_at": "2026-01-06T10:00:00Z",
        },
        "old_record": None,
    }
    return {
        "id": "evt_sample_webhook_supabase_db_changes",
        "trigger_key": "webhook.supabase.db_changes",
        "provider": "supabase",
        "account_id": None,
        "occurred_at": "2026-01-06T10:00:00Z",
        "received_at": "2026-01-06T10:00:01Z",
        "data": payload,
        "raw": {"payload": payload},
    }


def _form_trigger_config_schema() -> JsonSchema:
    """
    Config schema for form triggers (form.hosted, form.hitl).

    Form configuration (title, fields, description) should be in ui_meta.form_config,
    NOT in provider_config. This schema enforces an empty provider_config.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "description": (
            "Form triggers do not use provider_config. "
            "Place form configuration in ui_meta.form_config instead."
        ),
    }


def _form_hosted_payload_schema() -> JsonSchema:
    return {
        "type": "object",
        "description": "Form submission data with custom field values",
        "additionalProperties": True,
    }


def _form_hosted_sample_event() -> Dict[str, Any]:
    payload = {
        "name": "John Doe",
        "email": "john@example.com",
        "company": "Acme Corp",
        "message": "I'm interested in learning more about your product",
    }
    return {
        "id": "evt_sample_form_hosted",
        "trigger_key": "form.hosted",
        "provider": "form",
        "account_id": None,
        "occurred_at": "2026-01-08T14:30:00Z",
        "received_at": "2026-01-08T14:30:00Z",
        "data": payload,
        "raw": {"payload": payload},
    }


def _form_hitl_sample_event() -> Dict[str, Any]:
    """Sample event for HITL form submissions."""
    payload = {
        "approval_decision": "approved",
        "comments": "Looks good, proceed with the order.",
        "priority": "high",
    }
    return {
        "id": "evt_sample_form_hitl",
        "trigger_key": "form.hitl",
        "provider": "form",
        "account_id": None,
        "occurred_at": "2026-01-08T15:00:00Z",
        "received_at": "2026-01-08T15:00:00Z",
        "data": payload,
        "raw": {"payload": payload},
    }


trigger_registry = TriggerRegistry()
_register_builtin_triggers(trigger_registry)


__all__ = ["TriggerDefinition", "TriggerRegistry", "trigger_registry"]
