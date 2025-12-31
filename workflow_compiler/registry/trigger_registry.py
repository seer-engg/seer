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


trigger_registry = TriggerRegistry()
_register_builtin_triggers(trigger_registry)


__all__ = ["TriggerDefinition", "TriggerRegistry", "trigger_registry"]

