from __future__ import annotations

import json
from threading import Lock
from typing import Any, Dict, Iterable

from jsonschema import ValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator, validator_for

from workflow_compiler.schema.models import JsonSchema

ValidatorType = Draft202012Validator

_validator_cache: Dict[str, ValidatorType] = {}
_cache_lock = Lock()


def _cache_key(schema: JsonSchema, schema_id: str | None) -> str:
    if schema_id:
        return schema_id
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def get_validator(schema: JsonSchema, *, schema_id: str | None = None) -> ValidatorType:
    """
    Compile (and cache) a jsonschema validator for the provided schema.
    """

    key = _cache_key(schema, schema_id)
    with _cache_lock:
        validator = _validator_cache.get(key)
        if validator is None:
            validator_cls = validator_for(schema, default=Draft202012Validator)
            validator = validator_cls(schema)
            _validator_cache[key] = validator
    return validator


def validate_instance(schema: JsonSchema, instance: Any, *, schema_id: str | None = None) -> None:
    """
    Validate an instance against the provided schema.
    """

    validator = get_validator(schema, schema_id=schema_id)
    validator.validate(instance)


def check_schema(schema: JsonSchema) -> None:
    """
    Ensure the provided schema is itself valid JSON Schema.
    """

    validator_cls = validator_for(schema, default=Draft202012Validator)
    validator_cls.check_schema(schema)


def dereference_schema(
    schema: JsonSchema,
    *,
    root: JsonSchema | None = None,
    schema_id: str | None = None,
) -> JsonSchema:
    """
    Resolve any local $ref within the provided schema using jsonschema's resolver.
    """

    if not isinstance(schema, dict):
        return schema

    ref = schema.get("$ref")
    if not ref:
        return schema

    root_schema = root or schema
    validator = get_validator(root_schema, schema_id=schema_id)
    with validator.resolver.resolving(ref) as resolved:
        return resolved


def format_validation_error(error: ValidationError, *, prefix: str = "$") -> str:
    """
    Convert a jsonschema.ValidationError into a human-friendly error string.
    """

    path = prefix
    for token in error.absolute_path:
        if isinstance(token, int):
            path += f"[{token}]"
        else:
            path += f".{token}"
    return f"{path}: {error.message}"


__all__ = [
    "SchemaError",
    "ValidationError",
    "check_schema",
    "dereference_schema",
    "format_validation_error",
    "get_validator",
    "validate_instance",
]

