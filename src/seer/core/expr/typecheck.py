"""
Static validation for `${...}` references against known JSON schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

from seer.core.expr.parser import (
    IndexSegment,
    PathSegment,
    PropertySegment,
    ReferenceExpr,
)
from seer.core.schema.jsonschema_adapter import dereference_schema
from seer.core.schema.models import (
    JsonSchema,
    OutputContract,
    OutputMode,
)
from seer.core.schema.schema_registry import SchemaRegistry


class TypeCheckError(ValueError):
    pass


def schema_from_output_contract(contract: OutputContract, registry: SchemaRegistry) -> JsonSchema:
    if contract.mode == OutputMode.text:
        return {"type": "string"}
    if not contract.schema:
        raise TypeCheckError("Output contract missing schema for json mode")
    return registry.resolve_spec(contract.schema)


class TypeEnvironment:
    def __init__(self) -> None:
        self._schemas: Dict[str, JsonSchema] = {}

    def register(self, symbol: str, schema: JsonSchema) -> None:
        existing = self._schemas.get(symbol)
        if existing is not None and not _schemas_equivalent(existing, schema):
            raise TypeCheckError(
                f"Symbol '{symbol}' already registered with incompatible schema"
            )
        self._schemas[symbol] = schema

    def get(self, symbol: str) -> JsonSchema | None:
        return self._schemas.get(symbol)

    def require(self, symbol: str) -> JsonSchema:
        schema = self.get(symbol)
        if schema is None:
            raise TypeCheckError(f"No schema registered for '{symbol}'")
        return schema

    def as_dict(self) -> Mapping[str, JsonSchema]:
        return dict(self._schemas)


@dataclass
class Scope:
    """
    Tracks schemas for local temporaries (e.g. loop variables) layered on top of
    the global type environment.
    """

    env: TypeEnvironment
    locals: MutableMapping[str, JsonSchema] = field(default_factory=dict)

    def resolve(self, symbol: str) -> JsonSchema:
        if symbol in self.locals:
            return self.locals[symbol]
        return self.env.require(symbol)

    def nested(self) -> "Scope":
        return Scope(env=self.env, locals=dict(self.locals))


def resolve_schema_path(
    schema: JsonSchema, segments: Sequence[PathSegment], *, root: JsonSchema | None = None
) -> JsonSchema:
    root_schema = root or schema
    current = schema
    for segment in segments:
        current = _resolve_single_segment(current, segment, root_schema)
    return dereference_schema(current, root=root_schema)


def _normalize_schema_type(schema_type):
    """Normalize schema type from list to single type."""
    if isinstance(schema_type, list):
        if "object" in schema_type:
            return "object"
        if "array" in schema_type:
            return "array"
        if len(schema_type) == 1:
            return schema_type[0]
    return schema_type


def _try_anyof_oneof(schema: JsonSchema, segment: PathSegment, root: JsonSchema) -> JsonSchema | None:
    """Try to resolve segment against anyOf/oneOf branches."""
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            errors = []
            for candidate in schema[keyword]:
                try:
                    return _resolve_single_segment(candidate, segment, root)
                except TypeCheckError as exc:
                    errors.append(str(exc))
            raise TypeCheckError("; ".join(errors))
    return None


def _resolve_property(schema: JsonSchema, key: str, schema_type) -> JsonSchema:
    """Resolve property access on object schema."""
    if schema_type not in (None, "object"):
        raise TypeCheckError(f"Cannot access property '{key}' on {schema_type or 'value'}")
    properties = schema.get("properties", {})
    if key in properties:
        return properties[key]
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        return additional
    raise TypeCheckError(f"Property '{key}' not declared in schema")


def _resolve_numeric_index(schema: JsonSchema, schema_type) -> JsonSchema:
    """Resolve numeric array index."""
    if schema_type != "array":
        raise TypeCheckError("Numeric index is only valid on array schemas")
    items = schema.get("items")
    if not isinstance(items, dict):
        raise TypeCheckError("Array schema is missing 'items'")
    return items


def _resolve_string_index(schema: JsonSchema, index: str, schema_type) -> JsonSchema:
    """Resolve string index on object schema."""
    if schema_type not in (None, "object"):
        raise TypeCheckError("String index only valid on object schemas")
    properties = schema.get("properties", {})
    if index in properties:
        return properties[index]
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        return additional
    raise TypeCheckError(f"Key '{index}' not present in schema")


def _resolve_single_segment(
    schema: JsonSchema, segment: PathSegment, root: JsonSchema
) -> JsonSchema:
    schema = dereference_schema(schema, root=root)
    schema_type = _normalize_schema_type(schema.get("type"))

    anyof_result = _try_anyof_oneof(schema, segment, root)
    if anyof_result is not None:
        return anyof_result

    if isinstance(segment, PropertySegment):
        return _resolve_property(schema, segment.key, schema_type)

    if isinstance(segment, IndexSegment):
        if isinstance(segment.index, int):
            return _resolve_numeric_index(schema, schema_type)
        return _resolve_string_index(schema, segment.index, schema_type)

    raise TypeCheckError(f"Unsupported segment type {type(segment)!r}")


def typecheck_reference(reference: ReferenceExpr, scope: Scope) -> JsonSchema:
    schema = scope.resolve(reference.root)
    return resolve_schema_path(schema, reference.segments, root=schema)


def ensure_references_valid(references: Iterable[ReferenceExpr], scope: Scope) -> None:
    for reference in references:
        try:
            typecheck_reference(reference, scope)
        except TypeCheckError as exc:
            raise TypeCheckError(f"Reference '{reference.raw}' is invalid: {exc}") from exc


def _schemas_equivalent(first: JsonSchema, second: JsonSchema) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
