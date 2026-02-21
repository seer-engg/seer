# pylint: disable=unused-argument
# Reason: Test fixtures may not be directly used in all tests
"""
Unit tests for the static condition expression validator.

Tests the TypeInfo class, function type rules, and the validate_condition_expression
function in isolation.
"""

from __future__ import annotations

import pytest

from seer.core.expr.static_validator import (
    FUNCTION_TYPE_RULES,
    TypeInfo,
    validate_condition_expression,
)
from seer.core.expr.typecheck import Scope, TypeEnvironment

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


# =============================================================================
# TypeInfo Tests
# =============================================================================


class TestTypeInfo:
    """Tests for the TypeInfo class."""

    def test_from_schema_string(self) -> None:
        """Test TypeInfo.from_schema with string type."""
        schema = {"type": "string"}
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_string()
        assert not type_info.is_numeric()
        assert not type_info.is_array()
        assert type_info.is_iterable()  # Strings are iterable
        assert type_info.is_comparable()  # Strings can be compared
        assert not type_info.nullable

    def test_from_schema_number(self) -> None:
        """Test TypeInfo.from_schema with number type."""
        schema = {"type": "number"}
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_numeric()
        assert not type_info.is_string()
        assert type_info.is_comparable()
        assert not type_info.is_iterable()

    def test_from_schema_integer(self) -> None:
        """Test TypeInfo.from_schema with integer type."""
        schema = {"type": "integer"}
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_numeric()
        assert type_info.is_comparable()

    def test_from_schema_boolean(self) -> None:
        """Test TypeInfo.from_schema with boolean type."""
        schema = {"type": "boolean"}
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_boolean()
        assert not type_info.is_numeric()
        assert not type_info.is_comparable()

    def test_from_schema_array(self) -> None:
        """Test TypeInfo.from_schema with array type."""
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_array()
        assert type_info.is_iterable()
        assert type_info.is_container()
        assert type_info.is_sliceable()
        assert type_info.item_type is not None
        assert type_info.item_type.is_string()

    def test_from_schema_object(self) -> None:
        """Test TypeInfo.from_schema with object type."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"}
            }
        }
        type_info = TypeInfo.from_schema(schema)

        assert type_info.is_object()
        assert type_info.is_container()  # Objects support `in`
        assert not type_info.is_iterable()
        assert "name" in type_info.properties
        assert type_info.properties["name"].is_string()

    def test_from_schema_nullable(self) -> None:
        """Test TypeInfo.from_schema with nullable type."""
        schema = {"type": ["string", "null"]}
        type_info = TypeInfo.from_schema(schema)

        assert type_info.nullable
        assert type_info.is_string()

    def test_from_schema_union_types(self) -> None:
        """Test TypeInfo.from_schema with multiple types."""
        schema = {"type": ["string", "number"]}
        type_info = TypeInfo.from_schema(schema)

        # Both types should be recognized
        assert type_info.is_string()
        assert type_info.is_numeric()

    def test_from_python_value_string(self) -> None:
        """Test TypeInfo.from_python_value with string."""
        type_info = TypeInfo.from_python_value("hello")
        assert type_info.is_string()

    def test_from_python_value_int(self) -> None:
        """Test TypeInfo.from_python_value with int."""
        type_info = TypeInfo.from_python_value(42)
        assert type_info.is_numeric()

    def test_from_python_value_float(self) -> None:
        """Test TypeInfo.from_python_value with float."""
        type_info = TypeInfo.from_python_value(3.14)
        assert type_info.is_numeric()

    def test_from_python_value_bool(self) -> None:
        """Test TypeInfo.from_python_value with bool."""
        type_info = TypeInfo.from_python_value(True)
        assert type_info.is_boolean()

    def test_from_python_value_none(self) -> None:
        """Test TypeInfo.from_python_value with None."""
        type_info = TypeInfo.from_python_value(None)
        assert type_info.nullable

    def test_from_python_value_list(self) -> None:
        """Test TypeInfo.from_python_value with list."""
        type_info = TypeInfo.from_python_value([1, 2, 3])
        assert type_info.is_array()

    def test_type_description(self) -> None:
        """Test type_description returns human-readable string."""
        string_type = TypeInfo.from_schema({"type": "string"})
        assert string_type.type_description() == "string"

        number_type = TypeInfo.from_schema({"type": "number"})
        assert number_type.type_description() == "number"


# =============================================================================
# Function Type Rules Tests
# =============================================================================


class TestFunctionTypeRules:
    """Tests for the FUNCTION_TYPE_RULES configuration."""

    def test_len_rule(self) -> None:
        """Test len function requires iterable."""
        rule = FUNCTION_TYPE_RULES["len"]
        assert "iterable" in rule.accepts
        assert rule.returns == "integer"

    def test_any_all_rules(self) -> None:
        """Test any/all functions require iterable, return boolean."""
        for func_name in ["any", "all"]:
            rule = FUNCTION_TYPE_RULES[func_name]
            assert "iterable" in rule.accepts
            assert rule.returns == "boolean"

    def test_str_rule(self) -> None:
        """Test str function accepts anything."""
        rule = FUNCTION_TYPE_RULES["str"]
        assert "any" in rule.accepts
        assert rule.returns == "string"

    def test_int_float_rules(self) -> None:
        """Test int/float require numeric or string."""
        for func_name in ["int", "float"]:
            rule = FUNCTION_TYPE_RULES[func_name]
            assert "numeric" in rule.accepts or "string" in rule.accepts


# =============================================================================
# Scope Helper
# =============================================================================


def _create_test_scope(**schemas) -> Scope:
    """Create a test scope with the given symbol -> schema mappings."""
    env = TypeEnvironment()
    for symbol, schema in schemas.items():
        env.register(symbol, schema)
    return Scope(env=env)


# =============================================================================
# validate_condition_expression Tests
# =============================================================================


class TestValidateConditionExpression:
    """Tests for the validate_condition_expression function."""

    def test_valid_numeric_comparison(self) -> None:
        """Valid comparison: ${count} > 10 where count is number."""
        scope = _create_test_scope(count={"type": "number"})
        errors = validate_condition_expression("${count} > 10", scope)
        assert errors == []

    def test_valid_string_equality(self) -> None:
        """Valid comparison: ${status} == 'active' where status is string."""
        scope = _create_test_scope(status={"type": "string"})
        errors = validate_condition_expression("${status} == 'active'", scope)
        assert errors == []

    def test_invalid_string_greater_than_number(self) -> None:
        """Invalid: ${name} > 100 where name is string."""
        scope = _create_test_scope(name={"type": "string"})
        errors = validate_condition_expression("${name} > 100", scope)

        assert len(errors) >= 1
        assert any("compare" in err.lower() or "ordering" in err.lower() for err in errors)

    def test_valid_len_array(self) -> None:
        """Valid: len(${items}) > 0 where items is array."""
        scope = _create_test_scope(items={"type": "array", "items": {"type": "string"}})
        errors = validate_condition_expression("len(${items}) > 0", scope)
        assert errors == []

    def test_invalid_len_number(self) -> None:
        """Invalid: len(${count}) where count is number."""
        scope = _create_test_scope(count={"type": "number"})
        errors = validate_condition_expression("len(${count}) > 0", scope)

        assert len(errors) >= 1
        assert any("len" in err.lower() or "iterable" in err.lower() for err in errors)

    def test_valid_in_operator_array(self) -> None:
        """Valid: 'x' in ${items} where items is array."""
        scope = _create_test_scope(items={"type": "array", "items": {"type": "string"}})
        errors = validate_condition_expression("'x' in ${items}", scope)
        assert errors == []

    def test_invalid_in_operator_number(self) -> None:
        """Invalid: 'x' in ${count} where count is number."""
        scope = _create_test_scope(count={"type": "number"})
        errors = validate_condition_expression("'x' in ${count}", scope)

        assert len(errors) >= 1
        assert any("`in`" in err or "container" in err.lower() for err in errors)

    def test_valid_chained_comparison(self) -> None:
        """Valid: 0 < ${num} < 100 where num is number."""
        scope = _create_test_scope(num={"type": "number"})
        errors = validate_condition_expression("0 < ${num} < 100", scope)
        assert errors == []

    def test_invalid_chained_comparison_string(self) -> None:
        """Invalid: 0 < ${text} < 100 where text is string."""
        scope = _create_test_scope(text={"type": "string"})
        errors = validate_condition_expression("0 < ${text} < 100", scope)

        assert len(errors) >= 1
        # Should report type mismatch for comparing string with numbers
        assert any("compare" in err.lower() or "string" in err.lower() for err in errors)

    def test_unknown_variable_true(self) -> None:
        """Unknown variable 'true' should suggest 'True'."""
        scope = _create_test_scope(flag={"type": "boolean"})
        errors = validate_condition_expression("${flag} == true", scope)

        assert len(errors) >= 1
        assert any("true" in err and "True" in err for err in errors)

    def test_unknown_variable_false(self) -> None:
        """Unknown variable 'false' should suggest 'False'."""
        scope = _create_test_scope(flag={"type": "boolean"})
        errors = validate_condition_expression("${flag} == false", scope)

        assert len(errors) >= 1
        assert any("false" in err and "False" in err for err in errors)

    def test_unknown_variable_null(self) -> None:
        """Unknown variable 'null' should suggest 'None'."""
        scope = _create_test_scope(data={"type": ["object", "null"]})
        errors = validate_condition_expression("${data} == null", scope)

        assert len(errors) >= 1
        assert any("null" in err and "None" in err for err in errors)

    def test_unknown_variable_arbitrary(self) -> None:
        """Unknown arbitrary variable should report error."""
        scope = _create_test_scope(valid={"type": "boolean"})
        errors = validate_condition_expression("foo and ${valid}", scope)

        assert len(errors) >= 1
        assert any("foo" in err for err in errors)

    def test_method_call_error(self) -> None:
        """Method calls should produce helpful error."""
        scope = _create_test_scope(text={"type": "string"})
        errors = validate_condition_expression("${text}.lower() == 'test'", scope)

        assert len(errors) >= 1
        assert any("method" in err.lower() or "lower" in err for err in errors)

    def test_slice_on_array(self) -> None:
        """Valid: slicing an array."""
        scope = _create_test_scope(items={"type": "array", "items": {"type": "string"}})
        errors = validate_condition_expression("${items}[0]", scope)
        assert errors == []

    def test_slice_on_object_error(self) -> None:
        """Invalid: slicing an object with numeric range."""
        scope = _create_test_scope(data={"type": "object"})
        errors = validate_condition_expression("${data}[0]", scope)

        assert len(errors) >= 1

    def test_nullable_type_comparison(self) -> None:
        """Nullable types should still allow comparisons."""
        scope = _create_test_scope(value={"type": ["string", "null"]})
        errors = validate_condition_expression("${value} == 'test'", scope)
        assert errors == []

    def test_empty_expression_returns_no_errors(self) -> None:
        """Empty expression should be handled gracefully."""
        scope = _create_test_scope()
        errors = validate_condition_expression("", scope)
        assert errors == []

    def test_syntax_error_in_expression(self) -> None:
        """Syntax errors should be caught."""
        scope = _create_test_scope(x={"type": "number"})
        errors = validate_condition_expression("${x} > > 10", scope)

        assert len(errors) >= 1
        assert any("syntax" in err.lower() for err in errors)

    def test_valid_boolean_operators(self) -> None:
        """Valid: combining conditions with and/or."""
        scope = _create_test_scope(
            a={"type": "number"},
            b={"type": "number"}
        )
        errors = validate_condition_expression("${a} > 0 and ${b} < 100", scope)
        assert errors == []

    def test_valid_not_operator(self) -> None:
        """Valid: using not operator."""
        scope = _create_test_scope(flag={"type": "boolean"})
        errors = validate_condition_expression("not ${flag}", scope)
        assert errors == []

    def test_arithmetic_on_numbers(self) -> None:
        """Valid: arithmetic operations on numbers."""
        scope = _create_test_scope(
            a={"type": "number"},
            b={"type": "number"}
        )
        errors = validate_condition_expression("${a} + ${b} > 100", scope)
        assert errors == []

    def test_reference_resolution_failure_skips_type_check(self) -> None:
        """When reference resolution fails, type checking is skipped."""
        scope = _create_test_scope()  # No symbols registered
        # This should return empty errors because reference validation
        # will fail first (and we skip type checking to avoid cascading)
        errors = validate_condition_expression("${nonexistent} > 10", scope)
        assert errors == []  # Skipped due to reference error

    def test_string_in_string(self) -> None:
        """Valid: checking substring."""
        scope = _create_test_scope(text={"type": "string"})
        errors = validate_condition_expression("'hello' in ${text}", scope)
        assert errors == []

    def test_str_function_accepts_any(self) -> None:
        """str() function should accept any type."""
        scope = _create_test_scope(num={"type": "number"})
        errors = validate_condition_expression("str(${num}) == '42'", scope)
        assert errors == []


# =============================================================================
# Parameterized Tests - TypeInfo Schema Conversion
# =============================================================================


@pytest.mark.parametrize("schema,expected_checks", [
    # Basic types
    ({"type": "string"}, {"is_string": True, "is_numeric": False, "is_iterable": True, "is_comparable": True}),
    ({"type": "number"}, {"is_string": False, "is_numeric": True, "is_comparable": True, "is_iterable": False}),
    ({"type": "integer"}, {"is_numeric": True, "is_comparable": True}),
    ({"type": "boolean"}, {"is_boolean": True, "is_comparable": False, "is_numeric": False}),
    ({"type": "array", "items": {"type": "string"}}, {"is_array": True, "is_iterable": True, "is_container": True, "is_sliceable": True}),
    ({"type": "object"}, {"is_object": True, "is_container": True, "is_iterable": False}),
    # Nullable types
    ({"type": ["string", "null"]}, {"is_string": True, "nullable": True}),
    ({"type": ["number", "null"]}, {"is_numeric": True, "nullable": True}),
    # Union types
    ({"type": ["string", "number"]}, {"is_string": True, "is_numeric": True}),
])
def test_type_info_from_schema_parameterized(schema, expected_checks):
    """Parameterized test for TypeInfo.from_schema with various schema types."""
    type_info = TypeInfo.from_schema(schema)
    for check_name, expected_value in expected_checks.items():
        if check_name == "nullable":
            assert type_info.nullable == expected_value, f"Failed {check_name} for schema {schema}"
        else:
            method = getattr(type_info, check_name)
            assert method() == expected_value, f"Failed {check_name} for schema {schema}"


# =============================================================================
# Parameterized Tests - Comparison Operators (Valid)
# =============================================================================


@pytest.mark.parametrize("left_type,op,right_value", [
    # Numeric comparisons with all operators
    ("number", ">", "10"),
    ("number", "<", "10"),
    ("number", ">=", "10"),
    ("number", "<=", "10"),
    ("number", "==", "10"),
    ("number", "!=", "10"),
    ("integer", ">", "10"),
    ("integer", "<", "100"),
    # String comparisons
    ("string", "==", "'active'"),
    ("string", "!=", "'inactive'"),
    ("string", ">", "'abc'"),
    ("string", "<", "'xyz'"),
    # Boolean equality (no ordering)
    ("boolean", "==", "True"),
    ("boolean", "!=", "False"),
    ("boolean", "==", "None"),
])
def test_comparison_operators_valid(left_type, op, right_value):
    """Parameterized test for valid comparison operators."""
    scope = _create_test_scope(value={"type": left_type})
    expression = f"${{value}} {op} {right_value}"
    errors = validate_condition_expression(expression, scope)
    assert errors == [], f"Expected no errors for '{expression}', got: {errors}"


# =============================================================================
# Parameterized Tests - Comparison Operators (Invalid Type Mismatches)
# =============================================================================


@pytest.mark.parametrize("left_type,op,right_value,expected_error_contains", [
    # String vs number ordering
    ("string", ">", "100", "compare"),
    ("string", "<", "50", "compare"),
    ("string", ">=", "25", "compare"),
    ("string", "<=", "75", "compare"),
    # Boolean ordering not allowed
    ("boolean", ">", "10", "ordering"),
    ("boolean", "<", "5", "ordering"),
    # Object/array ordering not allowed
    ("object", "<", "5", "ordering"),
    ("array", ">=", "10", "ordering"),
])
def test_comparison_operators_invalid(left_type, op, right_value, expected_error_contains):
    """Parameterized test for invalid comparison operators (type mismatches)."""
    schema = {"type": left_type}
    if left_type == "array":
        schema["items"] = {"type": "string"}
    scope = _create_test_scope(value=schema)
    expression = f"${{value}} {op} {right_value}"
    errors = validate_condition_expression(expression, scope)

    assert len(errors) >= 1, f"Expected error for '{expression}'"
    assert any(expected_error_contains.lower() in err.lower() for err in errors), \
        f"Expected '{expected_error_contains}' in errors: {errors}"


# =============================================================================
# Parameterized Tests - Common Typos with Hints
# =============================================================================


@pytest.mark.parametrize("typo,hint_text", [
    ("true", "True"),
    ("false", "False"),
    ("null", "None"),
    ("none", "None"),
    ("undefined", "None"),
    ("nil", "None"),
])
def test_common_typo_hints_parameterized(typo, hint_text):
    """Parameterized test for common typo detection with hints."""
    scope = _create_test_scope(flag={"type": "boolean"})
    expression = f"${{flag}} == {typo}"
    errors = validate_condition_expression(expression, scope)

    assert len(errors) >= 1, f"Expected error for typo '{typo}'"
    assert any(typo in err and hint_text in err for err in errors), \
        f"Expected hint for '{typo}' → '{hint_text}' in errors: {errors}"


# =============================================================================
# Parameterized Tests - Function Argument Validation
# =============================================================================


def _get_type_schema(type_name):
    """Helper to get a schema for a type name."""
    if type_name == "array":
        return {"type": "array", "items": {"type": "string"}}
    return {"type": type_name}


@pytest.mark.parametrize("func,arg_type,should_pass", [
    # len requires iterable (array or string)
    ("len", "array", True),
    ("len", "string", True),
    ("len", "number", False),
    ("len", "boolean", False),
    ("len", "object", False),
    # any/all require iterable
    ("any", "array", True),
    ("all", "array", True),
    ("any", "string", True),
    ("all", "string", True),
    ("any", "number", False),
    ("all", "boolean", False),
    # str accepts any type
    ("str", "number", True),
    ("str", "boolean", True),
    ("str", "string", True),
    ("str", "array", True),
    ("str", "object", True),
    # int/float accept numeric or string
    ("int", "number", True),
    ("int", "string", True),
    ("int", "integer", True),
    ("float", "number", True),
    ("float", "string", True),
])
def test_function_argument_validation(func, arg_type, should_pass):
    """Parameterized test for function argument type validation."""
    scope = _create_test_scope(value=_get_type_schema(arg_type))
    expression = f"{func}(${{value}})"
    errors = validate_condition_expression(expression, scope)

    if should_pass:
        # Filter out any errors that aren't about the function
        func_errors = [e for e in errors if func.lower() in e.lower() or "iterable" in e.lower()]
        assert func_errors == [], f"Expected no function errors for {func}({arg_type}), got: {errors}"
    else:
        assert len(errors) >= 1, f"Expected error for {func}({arg_type})"


# =============================================================================
# Parameterized Tests - Container Operators (in/not in)
# =============================================================================


@pytest.mark.parametrize("container_type,should_pass", [
    ("array", True),
    ("string", True),
    ("object", True),
    ("number", False),
    ("boolean", False),
    ("integer", False),
])
def test_in_operator_container_types(container_type, should_pass):
    """Parameterized test for 'in' operator with different container types."""
    scope = _create_test_scope(container=_get_type_schema(container_type))
    expression = "'x' in ${container}"
    errors = validate_condition_expression(expression, scope)

    if should_pass:
        in_errors = [e for e in errors if "`in`" in e or "container" in e.lower()]
        assert in_errors == [], f"Expected no 'in' errors for {container_type}, got: {errors}"
    else:
        assert len(errors) >= 1, f"Expected error for 'in' on {container_type}"
        assert any("`in`" in e or "container" in e.lower() for e in errors), \
            f"Expected 'in' container error, got: {errors}"


@pytest.mark.parametrize("container_type,should_pass", [
    ("array", True),
    ("string", True),
    ("object", True),
    ("number", False),
    ("boolean", False),
])
def test_not_in_operator_container_types(container_type, should_pass):
    """Parameterized test for 'not in' operator with different container types."""
    scope = _create_test_scope(container=_get_type_schema(container_type))
    expression = "'x' not in ${container}"
    errors = validate_condition_expression(expression, scope)

    if should_pass:
        in_errors = [e for e in errors if "`in`" in e or "container" in e.lower()]
        assert in_errors == [], f"Expected no 'not in' errors for {container_type}, got: {errors}"
    else:
        assert len(errors) >= 1, f"Expected error for 'not in' on {container_type}"


# =============================================================================
# Parameterized Tests - Valid Expression Patterns
# =============================================================================


@pytest.mark.parametrize("expression,schemas", [
    # Simple comparisons
    ("${count} > 10", {"count": {"type": "number"}}),
    ("${count} >= 0", {"count": {"type": "integer"}}),
    ("${status} == 'active'", {"status": {"type": "string"}}),
    ("${status} != 'deleted'", {"status": {"type": "string"}}),
    # Function calls
    ("len(${items}) > 0", {"items": {"type": "array", "items": {"type": "string"}}}),
    ("len(${text}) < 100", {"text": {"type": "string"}}),
    ("str(${num}) == '42'", {"num": {"type": "number"}}),
    # Boolean operations
    ("${a} > 0 and ${b} < 100", {"a": {"type": "number"}, "b": {"type": "number"}}),
    ("${x} == 1 or ${y} == 2", {"x": {"type": "integer"}, "y": {"type": "integer"}}),
    ("not ${flag}", {"flag": {"type": "boolean"}}),
    # Container membership
    ("'key' in ${data}", {"data": {"type": "object"}}),
    ("'item' in ${items}", {"items": {"type": "array", "items": {"type": "string"}}}),
    ("'sub' in ${text}", {"text": {"type": "string"}}),
    # Chained comparisons
    ("0 < ${score} < 100", {"score": {"type": "number"}}),
    ("1 <= ${rank} <= 10", {"rank": {"type": "integer"}}),
    # Nullable type comparisons
    ("${value} == None", {"value": {"type": ["string", "null"]}}),
    ("${value} != None", {"value": {"type": ["object", "null"]}}),
    ("${value} == 'test'", {"value": {"type": ["string", "null"]}}),
    # Arithmetic in conditions
    ("${a} + ${b} > 100", {"a": {"type": "number"}, "b": {"type": "number"}}),
    ("${x} * 2 == ${y}", {"x": {"type": "integer"}, "y": {"type": "integer"}}),
])
def test_valid_expression_patterns(expression, schemas):
    """Parameterized test for valid expression patterns that should pass validation."""
    scope = _create_test_scope(**schemas)
    errors = validate_condition_expression(expression, scope)
    assert errors == [], f"Expected no errors for '{expression}', got: {errors}"


# =============================================================================
# Parameterized Tests - Unsupported Patterns with Helpful Errors
# =============================================================================


@pytest.mark.parametrize("expression,schemas,error_contains", [
    # Method calls
    ("${text}.lower() == 'test'", {"text": {"type": "string"}}, "method"),
    ("${text}.upper()", {"text": {"type": "string"}}, "method"),
    ("${text}.strip()", {"text": {"type": "string"}}, "method"),
    ("${items}.append('x')", {"items": {"type": "array", "items": {"type": "string"}}}, "method"),
])
def test_unsupported_method_calls(expression, schemas, error_contains):
    """Parameterized test for unsupported method call patterns."""
    scope = _create_test_scope(**schemas)
    errors = validate_condition_expression(expression, scope)

    assert len(errors) >= 1, f"Expected error for '{expression}'"
    assert any(error_contains.lower() in err.lower() for err in errors), \
        f"Expected '{error_contains}' in errors: {errors}"


# =============================================================================
# Parameterized Tests - Arithmetic Operators
# =============================================================================


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "//", "%"])
def test_arithmetic_operators_on_numbers(op):
    """Parameterized test for arithmetic operators on numeric types."""
    scope = _create_test_scope(a={"type": "number"}, b={"type": "number"})
    expression = f"${{a}} {op} ${{b}} > 0"
    errors = validate_condition_expression(expression, scope)
    assert errors == [], f"Expected no errors for arithmetic '{op}', got: {errors}"


# =============================================================================
# Parameterized Tests - Edge Cases
# =============================================================================


@pytest.mark.parametrize("expression,schemas", [
    # Multiple references in one expression
    ("${a} > ${b}", {"a": {"type": "number"}, "b": {"type": "number"}}),
    ("${x} == ${y} and ${z} > 0", {"x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "number"}}),
    # Nested function calls
    ("len(${items}) > 0 and len(${items}) < 100", {"items": {"type": "array", "items": {"type": "string"}}}),
    # Complex boolean expressions
    ("(${a} > 0 and ${b} > 0) or ${c} == True", {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "boolean"}}),
])
def test_complex_valid_expressions(expression, schemas):
    """Parameterized test for complex valid expressions."""
    scope = _create_test_scope(**schemas)
    errors = validate_condition_expression(expression, scope)
    assert errors == [], f"Expected no errors for '{expression}', got: {errors}"
