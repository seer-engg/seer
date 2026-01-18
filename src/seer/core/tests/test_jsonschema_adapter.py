from __future__ import annotations

import pytest

from seer.core.errors import ExecutionError
from seer.core.runtime.validate_output import validate_against_schema
from seer.core.schema.jsonschema_adapter import (
    check_schema,
    get_validator,
    validate_instance,
)


def test_get_validator_caches_by_schema_id() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    first = get_validator(schema, schema_id="test.schema")
    second = get_validator(schema, schema_id="test.schema")
    assert first is second


def test_validate_against_schema_formats_path() -> None:
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    with pytest.raises(ExecutionError) as excinfo:
        validate_against_schema(schema, {"count": "oops"}, schema_id="count-test")

    assert "$.count" in str(excinfo.value)


def test_nested_refs_validate_successfully() -> None:
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
        "type": "object",
        "properties": {"payload": {"$ref": "#/$defs/Node"}},
        "required": ["payload"],
    }

    validate_instance(schema, {"payload": {"value": 42}})


# def test_gmail_message_schema_is_valid_jsonschema() -> None:
#     check_schema(GMAIL_MESSAGE_SCHEMA)
