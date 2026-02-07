"""
Schema-driven type coercion for tool arguments.

=============================================================================
BACKGROUND: 2024-02 RCA - LLM Output Inconsistency Bug
=============================================================================
PROBLEM: Google Sheets API failed with:
    "Unable to parse range: Sheet1!'E3'"

The LLM was inconsistent in its outputs:
    - First iteration: returned "E2" (correct, no quotes)
    - Second iteration: returned "'E3'" (incorrect, with quotes)

This module provides intelligent parsing of LLM-generated values based on
JSON Schema type declarations. It acts as a defense layer between LLM outputs
and tool execution, normalizing various quirky LLM output formats.

HANDLED QUIRKS:
- Strings wrapped in quotes: 'E3' or "E3" instead of E3
- Arrays as JSON strings: "[1, 2, 3]" instead of [1, 2, 3]
- Numbers as strings: "42" instead of 42
- Booleans as strings: "true" instead of true

USAGE:
This module is used in two places:
1. src/seer/tools/executor.py - For API-level tool execution
2. src/seer/core/runtime/nodes.py - For workflow runtime tool execution

Both paths MUST apply coercion to prevent LLM output quirks from reaching tools.
=============================================================================

The coercion strategy is determined by the schema's "type" field:
- string: strip surrounding quotes
- integer/number: strip quotes, cast to int/float, enforce min/max bounds
- boolean: handle "true"/"false"/"True"/"False"/1/0/etc.
- array: try json.loads(), fall back to ast.literal_eval(), then comma-split
- object: try json.loads(), fall back to ast.literal_eval()
"""

import ast
import json
from typing import Any, Dict, List, Optional

from seer.logger import get_logger

logger = get_logger("shared.tools.coercion")


def _strip_outer_quotes(value: str) -> str:
    """
    Strip matching outer quotes from a string.

    Only strips if the value starts and ends with the same quote character.
    Does not strip if quotes are unmatched.

    Examples:
        "'E3'"  -> "E3"
        '"E3"'  -> "E3"
        "E3"    -> "E3" (no change)
        "'E3"   -> "'E3" (unmatched, no change)
        '""'    -> "" (empty string result)

    Args:
        value: String value to potentially strip quotes from

    Returns:
        String with outer quotes stripped if they matched
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _coerce_string(value: Any, field_name: str) -> str:
    """
    Coerce value to string, stripping outer quotes if present.

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)

    Returns:
        Coerced string value
    """
    if isinstance(value, str):
        result = _strip_outer_quotes(value)
        if result != value:
            logger.debug("Coerced %s: stripped quotes %r -> %r", field_name, value, result)
        return result
    return str(value)


def _coerce_integer(value: Any, field_name: str, schema: Dict[str, Any]) -> int:
    """
    Coerce value to integer with bounds checking.

    Handles:
    - int values (pass through)
    - float values (if they're whole numbers)
    - string values (strip quotes, then parse)

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)
        schema: Field schema (may contain minimum/maximum)

    Returns:
        Coerced integer value

    Raises:
        ValueError: If value cannot be converted to int
    """
    min_val = schema.get("minimum")
    max_val = schema.get("maximum")

    if isinstance(value, int) and not isinstance(value, bool):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        # Strip quotes first, then convert
        cleaned = _strip_outer_quotes(value.strip())
        result = int(cleaned)
    else:
        result = int(value)

    # Apply bounds
    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)

    logger.debug("Coerced %s to integer: %r -> %d", field_name, value, result)
    return result


def _coerce_number(value: Any, field_name: str, schema: Dict[str, Any]) -> float:
    """
    Coerce value to float with bounds checking.

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)
        schema: Field schema (may contain minimum/maximum)

    Returns:
        Coerced float value

    Raises:
        ValueError: If value cannot be converted to float
    """
    min_val = schema.get("minimum")
    max_val = schema.get("maximum")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        cleaned = _strip_outer_quotes(value.strip())
        result = float(cleaned)
    else:
        result = float(value)

    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)

    logger.debug("Coerced %s to number: %r -> %f", field_name, value, result)
    return result


def _coerce_boolean(value: Any, field_name: str) -> bool:
    """
    Coerce value to boolean, handling various string representations.

    Recognizes: true/false, True/False, yes/no, y/n, on/off, 1/0

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)

    Returns:
        Coerced boolean value

    Raises:
        ValueError: If string value is not a recognized boolean representation
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = _strip_outer_quotes(value.strip()).lower()
        if v in ("true", "1", "yes", "y", "on"):
            logger.debug("Coerced %s to boolean: %r -> True", field_name, value)
            return True
        if v in ("false", "0", "no", "n", "off"):
            logger.debug("Coerced %s to boolean: %r -> False", field_name, value)
            return False
        raise ValueError(f"Cannot coerce '{value}' to boolean")
    if isinstance(value, (int, float)):
        result = bool(value)
        logger.debug("Coerced %s to boolean: %r -> %s", field_name, value, result)
        return result
    return bool(value)


def _parse_string_as_array(trimmed: str) -> List[Any]:
    """
    Parse a string value into a list using multiple fallback strategies.

    Fallback chain:
    1. Try json.loads() (handles: ["a", "b"])
    2. Try ast.literal_eval() (handles: ['a', 'b'])
    3. Try comma-separated parsing
    4. Return single-element list or empty list

    Args:
        trimmed: Trimmed string value to parse

    Returns:
        Parsed list value
    """
    # Try JSON first (handles double-quoted strings: ["a", "b"])
    try:
        parsed = json.loads(trimmed)
        return list(parsed) if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    # Fall back to ast.literal_eval for Python literals (handles single quotes: ['a', 'b'])
    try:
        parsed = ast.literal_eval(trimmed)
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    except (ValueError, SyntaxError):
        pass

    # Last resort: comma-separated values
    if "," in trimmed:
        return [p.strip() for p in trimmed.split(",") if p.strip()]
    return [trimmed] if trimmed else []


def _coerce_array(value: Any, field_name: str, items_schema: Optional[Dict[str, Any]]) -> List[Any]:
    """
    Coerce value to array, parsing JSON/Python literals if needed.

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)
        items_schema: Schema for array items (for recursive coercion)

    Returns:
        Coerced list value
    """
    if isinstance(value, list):
        result = value
    elif isinstance(value, str):
        result = _parse_string_as_array(value.strip())
    else:
        result = [value] if value is not None else []

    # Recursively coerce array items if schema provided
    if items_schema and result:
        result = [_coerce_value(item, f"{field_name}[{i}]", items_schema) for i, item in enumerate(result)]

    logger.debug("Coerced %s to array: %r -> %r", field_name, value, result)
    return result


def _coerce_object(value: Any, field_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce value to object/dict, parsing JSON/Python literals if needed.

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging)
        schema: Field schema (may contain properties for recursive coercion)

    Returns:
        Coerced dict value

    Raises:
        ValueError: If value cannot be parsed as object
    """
    if isinstance(value, dict):
        result = value
    elif isinstance(value, str):
        trimmed = value.strip()
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                result = parsed
            else:
                raise ValueError(f"Expected object, got {type(parsed).__name__}")
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(trimmed)
                if isinstance(parsed, dict):
                    result = parsed
                else:
                    # Not caused by JSONDecodeError - suppress exception chaining
                    raise ValueError(f"Expected object, got {type(parsed).__name__}") from None
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Cannot parse '{value}' as object: {e}") from e
    else:
        raise ValueError(f"Cannot coerce {type(value).__name__} to object")

    # Recursively coerce object properties if schema has properties defined
    properties_schema = schema.get("properties", {})
    if properties_schema:
        result = {
            k: _coerce_value(v, f"{field_name}.{k}", properties_schema.get(k, {}))
            for k, v in result.items()
        }

    logger.debug("Coerced %s to object: %r -> %r", field_name, value, result)
    return result


def _coerce_value(value: Any, field_name: str, schema: Dict[str, Any]) -> Any:
    """
    Coerce a single value based on its schema type.

    Args:
        value: The value to coerce
        field_name: Name of the field (for logging and error messages)
        schema: JSON schema for this field

    Returns:
        Coerced value, or original value if no type specified
    """
    if value is None:
        return None

    schema_type = schema.get("type")

    # Dispatch table for type coercion - avoids long elif chain
    coercion_handlers = {
        "string": lambda: _coerce_string(value, field_name),
        "integer": lambda: _coerce_integer(value, field_name, schema),
        "number": lambda: _coerce_number(value, field_name, schema),
        "boolean": lambda: _coerce_boolean(value, field_name),
        "array": lambda: _coerce_array(value, field_name, schema.get("items")),
        "object": lambda: _coerce_object(value, field_name, schema),
    }

    handler = coercion_handlers.get(schema_type)
    if handler:
        return handler()
    # Unknown type or no type specified - return as-is
    return value


def coerce_arguments(
    arguments: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Coerce tool arguments based on JSON schema type declarations.

    This is the main entry point for the coercion layer. It takes raw arguments
    (potentially from LLM output) and coerces them based on the tool's schema.

    Args:
        arguments: Raw arguments dict from LLM or workflow
        schema: JSON schema with 'properties' defining expected types

    Returns:
        New dict with coerced values (original dict is not modified)

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "range": {"type": "string"},
        ...         "count": {"type": "integer"}
        ...     }
        ... }
        >>> coerce_arguments({"range": "'E3'", "count": "42"}, schema)
        {"range": "E3", "count": 42}
    """
    if not schema or "properties" not in schema:
        # No schema - just strip quotes from strings (conservative mode)
        return {k: _coerce_string(v, k) if isinstance(v, str) else v for k, v in arguments.items()}

    properties = schema.get("properties", {})
    result: Dict[str, Any] = {}

    for key, value in arguments.items():
        field_schema = properties.get(key, {})
        try:
            result[key] = _coerce_value(value, key, field_schema)
        except (ValueError, TypeError) as e:
            # Log warning but don't fail - pass original value through
            logger.warning("Coercion failed for %s: %s (passing through original)", key, e)
            result[key] = value

    return result
