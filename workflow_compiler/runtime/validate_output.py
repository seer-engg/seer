"""
Lightweight runtime JSON schema validation.

The implementation intentionally focuses on the subset of JSON Schema features
used by the workflow specs (type checks, required properties, nested objects,
and homogeneous arrays). Additional keywords can be added as the compiler
grows.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from workflow_compiler.errors import ExecutionError
from workflow_compiler.schema.models import JsonSchema


def validate_against_schema(schema: JsonSchema, value: Any, *, path: str = "$") -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if any(_matches_type(t, value) for t in schema_type):
            return
        raise ExecutionError(f"{path} must be one of {schema_type}, got {type(value).__name__}")

    if schema_type is not None:
        if not _matches_type(schema_type, value):
            raise ExecutionError(f"{path} must be {schema_type}, got {type(value).__name__}")

    if schema_type == "object" or (schema_type is None and isinstance(value, dict)):
        _validate_object(schema, value, path)
        return

    if schema_type == "array" or (schema_type is None and isinstance(value, list)):
        _validate_array(schema, value, path)
        return


def _matches_type(expected: str, value: Any) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    python_type = mapping.get(expected)
    if python_type is None:
        return True  # unknown type hint – best effort
    if expected == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _validate_object(schema: JsonSchema, value: Dict[str, Any], path: str) -> None:
    required = schema.get("required") or []
    for key in required:
        if key not in value:
            raise ExecutionError(f"{path}.{key} is required")

    properties = schema.get("properties") or {}
    for key, subschema in properties.items():
        if key in value:
            validate_against_schema(subschema, value[key], path=f"{path}.{key}")


def _validate_array(schema: JsonSchema, value: Iterable[Any], path: str) -> None:
    items = schema.get("items")
    if not isinstance(items, dict):
        return
    for idx, item in enumerate(value):
        validate_against_schema(items, item, path=f"{path}[{idx}]")


