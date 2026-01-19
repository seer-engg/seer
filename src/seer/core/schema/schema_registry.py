"""
Utility helpers for resolving schema specifications into concrete JSON schema
dicts. The registry is intentionally lightweight – it simply stores known
schemas (by id) and can also work with inline schemas that clients supply.
"""

from __future__ import annotations

from typing import Dict, MutableMapping

from seer.core.schema.models import (
    InlineSchema,
    JsonSchema,
    SchemaRef,
    SchemaSpec,
)


class SchemaNotFoundError(KeyError):
    """Raised when a schema reference cannot be resolved."""


class SchemaRegistry:
    """
    In-memory registry that maps schema ids to JSON Schema documents.

    The compiler treats this registry as authoritative for any SchemaRef.
    """

    def __init__(self, initial: MutableMapping[str, JsonSchema] | None = None) -> None:
        self._schemas: Dict[str, JsonSchema] = dict(initial or {})

    def register(self, schema_id: str, schema: JsonSchema) -> None:
        """
        Register (or override) a schema.
        """
        self._schemas[schema_id] = schema

    def get(self, schema_id: str) -> JsonSchema | None:
        return self._schemas.get(schema_id)

    def all(self) -> Dict[str, JsonSchema]:
        return dict(self._schemas)

    def has_schema(self, schema_id: str) -> bool:
        return schema_id in self._schemas

    def resolve_ref(self, ref: SchemaRef) -> JsonSchema:
        try:
            return self._schemas[ref.id]
        except KeyError as exc:
            raise SchemaNotFoundError(f"Schema '{ref.id}' is not registered") from exc

    def resolve_spec(self, spec: SchemaSpec) -> JsonSchema:
        if isinstance(spec, InlineSchema):
            return spec.json_schema
        return self.resolve_ref(spec)


def ensure_json_schema(schema: JsonSchema, *, schema_id: str | None = None) -> JsonSchema:
    """
    Defensive helper to ensure the provided value looks like a JSON schema. The
    implementation is intentionally conservative – for now we only check that
    the schema is a mapping with a stringified `type` when present. This keeps
    the compiler fast while still surfacing obvious misconfigurations.
    """

    if not isinstance(schema, dict):
        raise TypeError(
            f"Schema {schema_id or ''} must be a dict, received {type(schema).__name__}"
        )
    type_value = schema.get("type")
    if type_value is not None and not isinstance(type_value, str):
        raise TypeError(
            f"Schema {schema_id or ''} has invalid 'type' field: {type_value!r}"
        )
    return schema
