from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from workflow_compiler.errors import WorkflowCompilerError
from workflow_compiler.schema.models import InputDef, InputType, WorkflowSpec


def coerce_inputs(
    spec: WorkflowSpec, provided_inputs: Mapping[str, Any] | None
) -> Dict[str, Any]:
    """
    Validate and coerce runtime inputs against the workflow spec.

    - Applies defaults declared on each InputDef.
    - Attempts type coercion for common literal formats (strings for numbers, etc.).
    - Raises WorkflowCompilerError if a required input is missing or cannot be coerced.
    """

    incoming: Dict[str, Any] = dict(provided_inputs or {})
    coerced: Dict[str, Any] = {}
    errors: list[str] = []
    spec_inputs = spec.inputs or {}

    for name, definition in spec_inputs.items():
        has_value = name in incoming
        source_value = incoming.get(name, None)

        if not has_value:
            if definition.default is not None:
                try:
                    coerced[name] = _coerce_value(
                        definition.default, definition, input_name=name
                    )
                except WorkflowCompilerError as exc:
                    errors.append(str(exc))
                continue
            if definition.required:
                errors.append(f"Input '{name}' is required but was not provided")
            continue

        try:
            coerced[name] = _coerce_value(source_value, definition, input_name=name)
        except WorkflowCompilerError as exc:
            errors.append(str(exc))

    if errors:
        raise WorkflowCompilerError("; ".join(errors))

    # Preserve additional inputs (even if spec doesn't declare them) to avoid breaking callers.
    for extra_name, extra_value in incoming.items():
        if extra_name not in coerced:
            coerced[extra_name] = extra_value

    return coerced


def _coerce_value(
    value: Any,
    definition: InputDef,
    *,
    input_name: str | None = None,
) -> Any:
    if value is None:
        if definition.required:
            raise _input_error(input_name, "cannot be null")
        return None

    expected_type = definition.type

    if expected_type == InputType.string:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        raise _input_error(input_name, "must be a string-compatible value")

    if expected_type == InputType.integer:
        if _is_int(value):
            return int(value)
        if isinstance(value, str):
            value_str = value.strip()
            if value_str == "":
                raise _input_error(input_name, "must be a valid integer")
            try:
                parsed = int(value_str, 10)
            except ValueError:
                raise _input_error(input_name, f"'{value}' is not a valid integer")
            return parsed
        raise _input_error(input_name, "must be an integer")

    if expected_type == InputType.number:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            value_str = value.strip()
            if value_str == "":
                raise _input_error(input_name, "must be a valid number")
            try:
                parsed = float(value_str)
            except ValueError:
                raise _input_error(input_name, f"'{value}' is not a valid number")
            return parsed
        raise _input_error(input_name, "must be a number")

    if expected_type == InputType.boolean:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            raise _input_error(input_name, f"'{value}' is not a valid boolean literal")
        if _is_int(value):
            if value in (0, 1):
                return bool(value)
            raise _input_error(input_name, "integer boolean inputs must be 0 or 1")
        raise _input_error(input_name, "must be a boolean")

    if expected_type == InputType.object:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return _parse_json_literal(value, dict, input_name)
        raise _input_error(input_name, "must be an object/dict")

    if expected_type == InputType.array:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return _parse_json_literal(value, list, input_name)
        raise _input_error(input_name, "must be an array/list")

    # Unknown type enum – fall back to original value
    return value


def _parse_json_literal(value: str, expected_type: type, input_name: str | None) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise _input_error(
            input_name, f"invalid JSON literal: {exc.msg}"
        ) from exc

    if not isinstance(parsed, expected_type):
        type_name = "object" if expected_type is dict else "array"
        raise _input_error(
            input_name, f"JSON literal must decode to a {type_name}"
        )
    return parsed


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _input_error(input_name: str | None, detail: str) -> WorkflowCompilerError:
    if input_name:
        return WorkflowCompilerError(f"Input '{input_name}' {detail}")
    return WorkflowCompilerError(detail)

