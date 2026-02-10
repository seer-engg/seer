"""
Unit tests for schema-driven type coercion.

Tests the coercion module which handles LLM-generated values that may contain
unexpected formats like quoted strings, stringified arrays, etc.
"""

import pytest

from seer.tools.coercion import (
    _strip_outer_quotes,
    _coerce_string,
    _coerce_integer,
    _coerce_number,
    _coerce_boolean,
    _coerce_array,
    _coerce_object,
    coerce_arguments,
)


@pytest.mark.unit
class TestStripOuterQuotes:
    """Tests for the _strip_outer_quotes helper function."""

    def test_strips_single_quotes(self):
        assert _strip_outer_quotes("'E3'") == "E3"

    def test_strips_double_quotes(self):
        assert _strip_outer_quotes('"E3"') == "E3"

    def test_no_change_without_quotes(self):
        assert _strip_outer_quotes("E3") == "E3"

    def test_no_change_unmatched_quotes_single_double(self):
        assert _strip_outer_quotes("'E3\"") == "'E3\""

    def test_no_change_unmatched_quotes_double_single(self):
        assert _strip_outer_quotes("\"E3'") == "\"E3'"

    def test_empty_result_from_empty_single_quotes(self):
        assert _strip_outer_quotes("''") == ""

    def test_empty_result_from_empty_double_quotes(self):
        assert _strip_outer_quotes('""') == ""

    def test_preserves_inner_quotes_with_double_outer(self):
        assert _strip_outer_quotes("\"hello 'world'\"") == "hello 'world'"

    def test_preserves_inner_quotes_with_single_outer(self):
        assert _strip_outer_quotes("'hello \"world\"'") == 'hello "world"'

    def test_single_character_single_quote(self):
        assert _strip_outer_quotes("'") == "'"

    def test_single_character_double_quote(self):
        assert _strip_outer_quotes('"') == '"'

    def test_single_character_letter(self):
        assert _strip_outer_quotes("a") == "a"

    def test_empty_string(self):
        assert _strip_outer_quotes("") == ""

    def test_nested_quotes_strips_one_layer_double_single(self):
        assert _strip_outer_quotes("\"'nested'\"") == "'nested'"

    def test_nested_quotes_strips_one_layer_single_double(self):
        assert _strip_outer_quotes("'\"nested\"'") == '"nested"'

    def test_preserves_internal_quotes(self):
        assert _strip_outer_quotes('hello "world"') == 'hello "world"'

    def test_preserves_internal_single_quotes(self):
        assert _strip_outer_quotes("hello 'world'") == "hello 'world'"


@pytest.mark.unit
class TestCoerceString:
    """Tests for string coercion."""

    def test_strips_single_quotes(self):
        assert _coerce_string("'value'", "field") == "value"

    def test_strips_double_quotes(self):
        assert _coerce_string('"value"', "field") == "value"

    def test_no_change_for_unquoted(self):
        assert _coerce_string("value", "field") == "value"

    def test_converts_int_to_string(self):
        assert _coerce_string(42, "field") == "42"

    def test_converts_float_to_string(self):
        assert _coerce_string(3.14, "field") == "3.14"

    def test_converts_bool_to_string(self):
        assert _coerce_string(True, "field") == "True"

    def test_google_sheets_range_scenario(self):
        """Real-world test case: Google Sheets range with LLM-generated quotes."""
        assert _coerce_string("'Sheet1!E3'", "range") == "Sheet1!E3"


@pytest.mark.unit
class TestCoerceInteger:
    """Tests for integer coercion."""

    def test_int_passthrough(self):
        assert _coerce_integer(42, "field", {}) == 42

    def test_float_to_int_when_whole(self):
        assert _coerce_integer(42.0, "field", {}) == 42

    def test_string_to_int(self):
        assert _coerce_integer("42", "field", {}) == 42

    def test_quoted_string_to_int(self):
        assert _coerce_integer("'42'", "field", {}) == 42

    def test_double_quoted_string_to_int(self):
        assert _coerce_integer('"100"', "field", {}) == 100

    def test_string_with_whitespace(self):
        assert _coerce_integer("  42  ", "field", {}) == 42

    def test_minimum_bound(self):
        assert _coerce_integer(5, "field", {"minimum": 10}) == 10

    def test_maximum_bound(self):
        assert _coerce_integer(100, "field", {"maximum": 50}) == 50

    def test_both_bounds(self):
        assert _coerce_integer(5, "field", {"minimum": 10, "maximum": 50}) == 10
        assert _coerce_integer(100, "field", {"minimum": 10, "maximum": 50}) == 50
        assert _coerce_integer(30, "field", {"minimum": 10, "maximum": 50}) == 30

    def test_bool_false_not_treated_as_int(self):
        # bool is subclass of int, but we should convert it, not pass through
        result = _coerce_integer(False, "field", {})
        assert result == 0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _coerce_integer("not a number", "field", {})


@pytest.mark.unit
class TestCoerceNumber:
    """Tests for number (float) coercion."""

    def test_float_passthrough(self):
        assert _coerce_number(3.14, "field", {}) == 3.14

    def test_int_to_float(self):
        assert _coerce_number(42, "field", {}) == 42.0

    def test_string_to_float(self):
        assert _coerce_number("3.14", "field", {}) == 3.14

    def test_quoted_string_to_float(self):
        assert _coerce_number("'3.14'", "field", {}) == 3.14

    def test_minimum_bound(self):
        assert _coerce_number(1.0, "field", {"minimum": 5.0}) == 5.0

    def test_maximum_bound(self):
        assert _coerce_number(100.0, "field", {"maximum": 50.0}) == 50.0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _coerce_number("not a number", "field", {})


@pytest.mark.unit
class TestCoerceBoolean:
    """Tests for boolean coercion."""

    def test_bool_passthrough_true(self):
        assert _coerce_boolean(True, "field") is True

    def test_bool_passthrough_false(self):
        assert _coerce_boolean(False, "field") is False

    def test_string_true_lowercase(self):
        assert _coerce_boolean("true", "field") is True

    def test_string_false_lowercase(self):
        assert _coerce_boolean("false", "field") is False

    def test_string_true_capitalized(self):
        assert _coerce_boolean("True", "field") is True

    def test_string_false_capitalized(self):
        assert _coerce_boolean("False", "field") is False

    def test_string_yes(self):
        assert _coerce_boolean("yes", "field") is True

    def test_string_no(self):
        assert _coerce_boolean("no", "field") is False

    def test_string_y(self):
        assert _coerce_boolean("y", "field") is True

    def test_string_n(self):
        assert _coerce_boolean("n", "field") is False

    def test_string_on(self):
        assert _coerce_boolean("on", "field") is True

    def test_string_off(self):
        assert _coerce_boolean("off", "field") is False

    def test_string_1(self):
        assert _coerce_boolean("1", "field") is True

    def test_string_0(self):
        assert _coerce_boolean("0", "field") is False

    def test_quoted_true(self):
        assert _coerce_boolean("'true'", "field") is True

    def test_quoted_false(self):
        assert _coerce_boolean('"false"', "field") is False

    def test_int_1(self):
        assert _coerce_boolean(1, "field") is True

    def test_int_0(self):
        assert _coerce_boolean(0, "field") is False

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _coerce_boolean("maybe", "field")


@pytest.mark.unit
class TestCoerceArray:
    """Tests for array coercion."""

    def test_list_passthrough(self):
        assert _coerce_array([1, 2, 3], "field", None) == [1, 2, 3]

    def test_json_array_string(self):
        assert _coerce_array('[1, 2, 3]', "field", None) == [1, 2, 3]

    def test_json_string_array(self):
        assert _coerce_array('["a", "b", "c"]', "field", None) == ["a", "b", "c"]

    def test_python_literal_single_quotes(self):
        """Test ast.literal_eval fallback for Python-style single quotes."""
        assert _coerce_array("['a', 'b', 'c']", "field", None) == ["a", "b", "c"]

    def test_comma_separated_fallback(self):
        assert _coerce_array("a, b, c", "field", None) == ["a", "b", "c"]

    def test_comma_separated_no_spaces(self):
        assert _coerce_array("a,b,c", "field", None) == ["a", "b", "c"]

    def test_single_value_wrapped(self):
        assert _coerce_array("single", "field", None) == ["single"]

    def test_empty_string_returns_empty_list(self):
        assert _coerce_array("", "field", None) == []

    def test_none_value_returns_empty_list(self):
        assert _coerce_array(None, "field", None) == []

    def test_non_list_wrapped(self):
        assert _coerce_array(42, "field", None) == [42]

    def test_recursive_item_coercion(self):
        """Test that array items are coerced based on items schema."""
        items_schema = {"type": "integer"}
        result = _coerce_array('["1", "2", "3"]', "field", items_schema)
        assert result == [1, 2, 3]

    def test_recursive_string_item_coercion(self):
        """Test that array string items have quotes stripped."""
        items_schema = {"type": "string"}
        result = _coerce_array(["'a'", "'b'"], "field", items_schema)
        assert result == ["a", "b"]

    def test_gmail_recipients_scenario(self):
        """Real-world test case: email recipients from LLM."""
        result = _coerce_array("['user@example.com', 'other@example.com']", "to", None)
        assert result == ["user@example.com", "other@example.com"]


@pytest.mark.unit
class TestCoerceObject:
    """Tests for object/dict coercion."""

    def test_dict_passthrough(self):
        assert _coerce_object({"key": "value"}, "field", {}) == {"key": "value"}

    def test_json_object_string(self):
        assert _coerce_object('{"key": "value"}', "field", {}) == {"key": "value"}

    def test_python_literal_dict(self):
        """Test ast.literal_eval fallback for Python-style dicts."""
        assert _coerce_object("{'key': 'value'}", "field", {}) == {"key": "value"}

    def test_nested_json(self):
        result = _coerce_object('{"outer": {"inner": 42}}', "field", {})
        assert result == {"outer": {"inner": 42}}

    def test_recursive_property_coercion(self):
        """Test that object properties are coerced based on properties schema."""
        schema = {
            "properties": {
                "count": {"type": "integer"},
                "name": {"type": "string"}
            }
        }
        result = _coerce_object({"count": "42", "name": "'test'"}, "field", schema)
        assert result == {"count": 42, "name": "test"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _coerce_object("not valid json", "field", {})

    def test_non_dict_type_raises(self):
        with pytest.raises(ValueError):
            _coerce_object(42, "field", {})

    def test_json_array_raises(self):
        """Parsing an array when expecting object should raise."""
        with pytest.raises(ValueError):
            _coerce_object("[1, 2, 3]", "field", {})


@pytest.mark.unit
class TestCoerceArguments:
    """Tests for the main coerce_arguments function."""

    def test_string_field_strips_quotes(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        result = coerce_arguments({"name": "'John'"}, schema)
        assert result["name"] == "John"

    def test_integer_field_parses_string(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }
        result = coerce_arguments({"count": "42"}, schema)
        assert result["count"] == 42

    def test_boolean_field_parses_string(self):
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"}
            }
        }
        result = coerce_arguments({"enabled": "true"}, schema)
        assert result["enabled"] is True

    def test_array_field_parses_string(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            }
        }
        result = coerce_arguments({"items": '["a", "b"]'}, schema)
        assert result["items"] == ["a", "b"]

    def test_mixed_types(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }
        result = coerce_arguments({
            "name": "'test'",
            "count": "'42'",
            "enabled": "yes",
            "tags": "['a', 'b']"
        }, schema)
        assert result == {
            "name": "test",
            "count": 42,
            "enabled": True,
            "tags": ["a", "b"]
        }

    def test_no_schema_strips_string_quotes(self):
        """Without schema, conservatively strip quotes from strings."""
        result = coerce_arguments({"field": "'value'"}, None)
        assert result["field"] == "value"

    def test_no_schema_preserves_non_strings(self):
        result = coerce_arguments({"count": 42}, None)
        assert result["count"] == 42

    def test_unknown_field_passed_through(self):
        """Fields not in schema should pass through."""
        schema = {
            "type": "object",
            "properties": {
                "known": {"type": "string"}
            }
        }
        result = coerce_arguments({"known": "'value'", "unknown": "'other'"}, schema)
        assert result["known"] == "value"
        # Unknown field has no type, so passed as-is
        assert result["unknown"] == "'other'"

    def test_coercion_failure_passes_through(self):
        """If coercion fails, original value should be preserved."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }
        result = coerce_arguments({"count": "not a number"}, schema)
        # Should pass through original value without raising
        assert result["count"] == "not a number"

    def test_preserves_original_dict(self):
        """Original arguments dict should not be modified."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        original = {"name": "'John'"}
        result = coerce_arguments(original, schema)
        assert original["name"] == "'John'"  # Original unchanged
        assert result["name"] == "John"  # Result coerced

    def test_empty_arguments(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        result = coerce_arguments({}, schema)
        assert result == {}

    def test_google_sheets_real_scenario(self):
        """Real-world test case: Google Sheets write with quoted range."""
        schema = {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string"},
                "range": {"type": "string"},
                "values": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
        result = coerce_arguments({
            "spreadsheet_id": "abc123",
            "range": "'Sheet1!E3'",
            "values": [["'value1'", "'value2'"]]
        }, schema)
        assert result["spreadsheet_id"] == "abc123"
        assert result["range"] == "Sheet1!E3"
        # Nested array items should also be coerced
        assert result["values"] == [["value1", "value2"]]

    def test_none_value_preserved(self):
        """None values should pass through unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "optional": {"type": "string"}
            }
        }
        result = coerce_arguments({"optional": None}, schema)
        assert result["optional"] is None
