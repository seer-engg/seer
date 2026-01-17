"""
In-memory registry describing workflow trigger metadata and schemas.
"""

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
    "poll.gmail.email_received",
    "schedule.cron",
]

def _register_builtin_triggers(registry: TriggerRegistry) -> None:
    registry.register(
        TriggerDefinition(
            key="webhook.generic",
            title="Webhook",
            provider="generic",
            mode="webhook",
            description="Accepts arbitrary JSON payloads via signed webhook requests.",
            schemas=TriggerSchemas(event=_default_event_envelope_schema()),
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
            schemas=TriggerSchemas(event=_enveloped_event_schema(_form_hosted_payload_schema())),
            meta=TriggerMetadata(
                sample_event=_form_hosted_sample_event(),
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
                "description": "IANA timezone identifier (e.g., America/New_York, UTC)",
                "default": "UTC",
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


trigger_registry = TriggerRegistry()
_register_builtin_triggers(trigger_registry)


__all__ = ["TriggerDefinition", "TriggerRegistry", "trigger_registry"]
