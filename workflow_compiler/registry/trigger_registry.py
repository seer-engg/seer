"""
In-memory registry describing workflow trigger metadata and schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional

from workflow_compiler.schema.models import JsonSchema


@dataclass
class TriggerDefinition:
    key: str
    title: str
    provider: str
    mode: str
    description: Optional[str] = None
    event_schema: JsonSchema = field(default_factory=dict)
    filter_schema: Optional[JsonSchema] = None
    config_schema: Optional[JsonSchema] = None
    sample_event: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def _register_builtin_triggers(registry: TriggerRegistry) -> None:
    registry.register(
        TriggerDefinition(
            key="webhook.generic",
            title="Generic Webhook",
            provider="generic",
            mode="webhook",
            description="Accepts arbitrary JSON payloads via signed webhook requests.",
            event_schema=_default_event_envelope_schema(),
        )
    )
    registry.register(
        TriggerDefinition(
            key="poll.gmail.email_received",
            title="Gmail – New Email",
            provider="gmail",
            mode="polling",
            description="Poll a Gmail inbox for newly received messages using OAuth credentials.",
            event_schema=_gmail_email_received_event_schema(),
            config_schema=_gmail_email_received_config_schema(),
            sample_event=_gmail_email_received_sample_event(),
            metadata={"polling": True, "integration": "gmail"},
        )
    )


def _gmail_email_received_event_schema() -> JsonSchema:
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
                "description": "Overlap window in milliseconds to re-read recent messages for dedupe safety.",
            },
        },
    }


def _gmail_email_received_sample_event() -> Dict[str, Any]:
    return {
        "message_id": "18c123example",
        "thread_id": "18c123example",
        "internal_date_ms": 1735630123456,
        "snippet": "Reminder about tomorrow's demo",
        "subject": "Demo tomorrow?",
        "from": {"name": "Product Team", "email": "product@example.com"},
        "to": [
            {"name": "You", "email": "you@example.com"},
        ],
        "labels": ["INBOX", "UNREAD"],
        "date_header": "Fri, 13 Dec 2025 10:00:00 -0000",
        "history_id": "123456",
    }


trigger_registry = TriggerRegistry()
_register_builtin_triggers(trigger_registry)


__all__ = ["TriggerDefinition", "TriggerRegistry", "trigger_registry"]

