"""
Lightweight runtime JSON schema validation.

The implementation intentionally focuses on the subset of JSON Schema features
used by the workflow specs (type checks, required properties, nested objects,
and homogeneous arrays). Additional keywords can be added as the compiler
grows.
"""

from __future__ import annotations

from typing import Any

from seer.core.errors import ExecutionError
from seer.core.schema.jsonschema_adapter import (
    ValidationError,
    format_validation_error,
    validate_instance,
)
from seer.core.schema.models import JsonSchema


def validate_against_schema(
    schema: JsonSchema,
    value: Any,
    *,
    path: str = "$",
    schema_id: str | None = None,
) -> None:
    """
    Validate the provided value using jsonschema, wrapping any failures
    in our ExecutionError type for clearer workflow diagnostics.
    """

    try:
        validate_instance(schema, value, schema_id=schema_id)
    except ValidationError as exc:
        message = format_validation_error(exc, prefix=path)
        raise ExecutionError(message) from exc
