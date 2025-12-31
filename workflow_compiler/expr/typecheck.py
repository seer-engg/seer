"""
Static validation for `${...}` references against known JSON schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

from workflow_compiler.expr.parser import IndexSegment, PathSegment, PropertySegment, ReferenceExpr
from workflow_compiler.schema.jsonschema_adapter import dereference_schema
from workflow_compiler.schema.models import InputDef, InputType, JsonSchema, OutputContract, OutputMode
from workflow_compiler.schema.schema_registry import SchemaRegistry


class TypeCheckError(ValueError):
    pass


def _type_from_input(definition: InputDef) -> JsonSchema:
    mapping = {
        InputType.string: {"type": "string"},
        InputType.integer: {"type": "integer"},
        InputType.number: {"type": "number"},
        InputType.boolean: {"type": "boolean"},
        InputType.object: {"type": "object"},
        InputType.array: {"type": "array"},
    }
    base = dict(mapping[definition.type])
    if definition.default is not None:
        base["default"] = definition.default
    return base


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


def register_inputs(env: TypeEnvironment, inputs: Mapping[str, InputDef]) -> None:
    properties: Dict[str, JsonSchema] = {}
    required = []
    for name, definition in inputs.items():
        schema = _type_from_input(definition)
        env.register(f"inputs.{name}", schema)
        properties[name] = schema
        if definition.required:
            required.append(name)
    env.register(
        "inputs",
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        },
    )


def resolve_schema_path(
    schema: JsonSchema, segments: Sequence[PathSegment], *, root: JsonSchema | None = None
) -> JsonSchema:
    root_schema = root or schema
    current = schema
    for segment in segments:
        current = _resolve_single_segment(current, segment, root_schema)
    return dereference_schema(current, root=root_schema)


def _resolve_single_segment(
    schema: JsonSchema, segment: PathSegment, root: JsonSchema
) -> JsonSchema:
    schema = dereference_schema(schema, root=root)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "object" in schema_type:
            schema_type = "object"
        elif "array" in schema_type:
            schema_type = "array"
        elif len(schema_type) == 1:
            schema_type = schema_type[0]

    # anyOf / oneOf: succeed if any branch is valid
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            errors = []
            for candidate in schema[keyword]:
                try:
                    return _resolve_single_segment(candidate, segment, root)
                except TypeCheckError as exc:
                    errors.append(str(exc))
            raise TypeCheckError("; ".join(errors))

    if isinstance(segment, PropertySegment):
        if schema_type not in (None, "object"):
            raise TypeCheckError(f"Cannot access property '{segment.key}' on {schema_type or 'value'}")
        properties = schema.get("properties", {})
        if segment.key in properties:
            return properties[segment.key]
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            return additional
        raise TypeCheckError(f"Property '{segment.key}' not declared in schema")

    if isinstance(segment, IndexSegment):
        if isinstance(segment.index, int):
            if schema_type != "array":
                raise TypeCheckError("Numeric index is only valid on array schemas")
            items = schema.get("items")
            if not isinstance(items, dict):
                raise TypeCheckError("Array schema is missing 'items'")
            return items
        if schema_type not in (None, "object"):
            raise TypeCheckError("String index only valid on object schemas")
        properties = schema.get("properties", {})
        if segment.index in properties:
            return properties[segment.index]
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            return additional
        raise TypeCheckError(f"Key '{segment.index}' not present in schema")

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


