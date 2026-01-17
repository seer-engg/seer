from __future__ import annotations

from typing import Any

from jsonschema import ValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator, validator_for

from seer.core.schema.models import JsonSchema

ValidatorType = Draft202012Validator


def get_validator(schema: JsonSchema, *, schema_id: str | None = None) -> ValidatorType:
    """
    Compile a jsonschema validator for the provided schema.
    """
    validator_cls = validator_for(schema, default=Draft202012Validator)
    return validator_cls(schema)


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
