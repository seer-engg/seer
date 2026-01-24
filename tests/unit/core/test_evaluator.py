# pylint: disable=too-many-lines,unused-argument,duplicate-code
# Reason: Expression evaluator tests need many cases; schema snippets intentionally mirror examples
"""
Unit tests for expression evaluator.

Tests runtime evaluation of ${...} expressions, template rendering, and conditions.
Target coverage: 95%+
"""
import pytest

from seer.core.expr.evaluator import (
    EvaluationContext,
    EvaluationError,
    evaluate_value,
    evaluate_condition,
    render_template,
    resolve_reference,
)
from seer.core.expr.parser import parse_reference_string


# =============================================================================
# EvaluationContext Tests
# =============================================================================


def test_evaluation_context_creation():
    """Test creating evaluation context."""
    ctx = EvaluationContext(
        state={"key": "value"},
        locals={"local_key": "local_value"}
    )
    assert ctx.state == {"key": "value"}
    assert ctx.locals == {"local_key": "local_value"}


def test_evaluation_context_with_locals():
    """Test creating context with additional locals."""
    ctx = EvaluationContext(
        state={"key": "value"},
        locals={"old": "value"}
    )

    new_ctx = ctx.with_locals({"new": "added", "old": "overridden"})

    assert new_ctx.locals["new"] == "added"
    assert new_ctx.locals["old"] == "overridden"
    assert new_ctx.state == ctx.state  # State unchanged


# =============================================================================
# Reference Resolution Tests
# =============================================================================


def test_resolve_reference_from_locals():
    """Test resolving reference from locals."""
    ctx = EvaluationContext(
        state={},
        locals={"myvar": "local_value"}
    )
    ref = parse_reference_string("myvar")

    result = resolve_reference(ctx, ref)
    assert result == "local_value"


def test_resolve_reference_from_state():
    """Test resolving reference from state."""
    ctx = EvaluationContext(
        state={"myvar": "state_value"},
        locals={}
    )
    ref = parse_reference_string("myvar")

    result = resolve_reference(ctx, ref)
    assert result == "state_value"


def test_resolve_reference_locals_precedence():
    """Test that locals take precedence over state."""
    ctx = EvaluationContext(
        state={"myvar": "state_value"},
        locals={"myvar": "local_value"}
    )
    ref = parse_reference_string("myvar")

    result = resolve_reference(ctx, ref)
    assert result == "local_value"


def test_resolve_reference_property_access():
    """Test resolving reference with property access."""
    ctx = EvaluationContext(
        state={"obj": {"name": "John", "age": 30}},
        locals={}
    )
    ref = parse_reference_string("obj.name")

    result = resolve_reference(ctx, ref)
    assert result == "John"


def test_resolve_reference_nested_property():
    """Test resolving reference with nested property access."""
    ctx = EvaluationContext(
        state={
            "user": {
                "profile": {
                    "email": "test@example.com"
                }
            }
        },
        locals={}
    )
    ref = parse_reference_string("user.profile.email")

    result = resolve_reference(ctx, ref)
    assert result == "test@example.com"


def test_resolve_reference_array_index():
    """Test resolving reference with array index."""
    ctx = EvaluationContext(
        state={"items": ["first", "second", "third"]},
        locals={}
    )
    ref = parse_reference_string("items[1]")

    result = resolve_reference(ctx, ref)
    assert result == "second"


def test_resolve_reference_string_index():
    """Test resolving reference with string index."""
    ctx = EvaluationContext(
        state={"data": {"key1": "value1", "key2": "value2"}},
        locals={}
    )
    ref = parse_reference_string('data["key1"]')

    result = resolve_reference(ctx, ref)
    assert result == "value1"


def test_resolve_reference_mixed_access():
    """Test resolving reference with mixed property and index access."""
    ctx = EvaluationContext(
        state={
            "users": [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30}
            ]
        },
        locals={}
    )
    ref = parse_reference_string("users[0].name")

    result = resolve_reference(ctx, ref)
    assert result == "Alice"


def test_resolve_reference_from_config():
    """Test resolving reference from config."""
    ctx = EvaluationContext(
        state={},
        locals={},
        config={"API_KEY": "secret_key"}
    )
    ref = parse_reference_string("API_KEY")

    result = resolve_reference(ctx, ref)
    assert result == "secret_key"


def test_resolve_reference_config_root():
    """Test resolving 'config' root reference."""
    ctx = EvaluationContext(
        state={},
        locals={},
        config={"API_KEY": "secret"}
    )
    ref = parse_reference_string("config")

    result = resolve_reference(ctx, ref)
    assert result == {"API_KEY": "secret"}


def test_resolve_reference_from_trigger():
    """Test resolving reference from trigger."""
    ctx = EvaluationContext(
        state={},
        locals={},
        trigger={"trigger_id": "t1", "data": {"message": "hello"}}
    )
    ref = parse_reference_string("t1")

    result = resolve_reference(ctx, ref)
    assert result == {"trigger_id": "t1", "data": {"message": "hello"}}


def test_resolve_reference_trigger_property():
    """Test resolving trigger property reference."""
    ctx = EvaluationContext(
        state={},
        locals={},
        trigger={"trigger_id": "t1", "data": {"message": "hello"}}
    )
    ref = parse_reference_string("t1.data")

    result = resolve_reference(ctx, ref)
    assert result == {"message": "hello"}


# =============================================================================
# Reference Resolution Error Tests
# =============================================================================


def test_resolve_reference_unknown_root():
    """Test that unknown root raises EvaluationError."""
    ctx = EvaluationContext(state={}, locals={})
    ref = parse_reference_string("unknown")

    with pytest.raises(EvaluationError, match="Unknown reference root"):
        resolve_reference(ctx, ref)


def test_resolve_reference_wrong_trigger():
    """Test error when referencing wrong trigger ID."""
    ctx = EvaluationContext(
        state={},
        locals={},
        trigger={"id": "t1", "data": {}}
    )
    ref = parse_reference_string("t2")

    with pytest.raises(EvaluationError, match="does not match the active trigger"):
        resolve_reference(ctx, ref)


def test_resolve_reference_missing_property():
    """Test error when property doesn't exist."""
    ctx = EvaluationContext(
        state={"obj": {"name": "John"}},
        locals={}
    )
    ref = parse_reference_string("obj.missing")

    with pytest.raises(EvaluationError, match="Property 'missing' not found"):
        resolve_reference(ctx, ref)


def test_resolve_reference_property_on_non_object():
    """Test error when accessing property on non-object."""
    ctx = EvaluationContext(
        state={"value": "string"},
        locals={}
    )
    ref = parse_reference_string("value.property")

    with pytest.raises(EvaluationError, match="Cannot access property .* on non-object"):
        resolve_reference(ctx, ref)


def test_resolve_reference_index_out_of_range():
    """Test error when array index is out of range."""
    ctx = EvaluationContext(
        state={"items": ["a", "b"]},
        locals={}
    )
    ref = parse_reference_string("items[10]")

    with pytest.raises(EvaluationError, match="Index .* out of range"):
        resolve_reference(ctx, ref)


def test_resolve_reference_index_on_non_list():
    """Test error when using numeric index on non-list."""
    ctx = EvaluationContext(
        state={"value": "string"},
        locals={}
    )
    ref = parse_reference_string("value[0]")

    with pytest.raises(EvaluationError, match="Cannot use numeric index on non-list"):
        resolve_reference(ctx, ref)


def test_resolve_reference_string_index_missing_key():
    """Test error when string index key doesn't exist."""
    ctx = EvaluationContext(
        state={"data": {"key1": "value"}},
        locals={}
    )
    ref = parse_reference_string('data["missing"]')

    with pytest.raises(EvaluationError, match="Key 'missing' not found"):
        resolve_reference(ctx, ref)


# =============================================================================
# Template Rendering Tests
# =============================================================================


def test_render_template_plain_text():
    """Test rendering template with no references."""
    ctx = EvaluationContext(state={}, locals={})
    result = render_template(ctx, "Hello World")
    assert result == "Hello World"


def test_render_template_single_reference():
    """Test rendering template with single reference returns value directly."""
    ctx = EvaluationContext(
        state={"value": 42},
        locals={}
    )
    result = render_template(ctx, "${value}")
    # Single reference returns actual value, not string
    assert result == 42


def test_render_template_with_interpolation():
    """Test rendering template with interpolated references."""
    ctx = EvaluationContext(
        state={"first": "John", "last": "Doe"},
        locals={}
    )
    result = render_template(ctx, "Hello ${first} ${last}!")
    assert result == "Hello John Doe!"


def test_render_template_multiple_refs():
    """Test rendering template with multiple references."""
    ctx = EvaluationContext(
        state={"x": 10, "y": 20},
        locals={}
    )
    result = render_template(ctx, "x=${x}, y=${y}")
    assert result == "x=10, y=20"


def test_render_template_with_none_value():
    """Test rendering template with None value."""
    ctx = EvaluationContext(
        state={"value": None},
        locals={}
    )
    result = render_template(ctx, "Value: ${value}")
    # None is rendered as empty string
    assert result == "Value: "


def test_render_template_with_boolean():
    """Test rendering template with boolean value."""
    ctx = EvaluationContext(
        state={"flag": True},
        locals={}
    )
    result = render_template(ctx, "Flag: ${flag}")
    assert result == "Flag: True"


# =============================================================================
# Value Evaluation Tests
# =============================================================================


def test_evaluate_value_primitive():
    """Test evaluating primitive values (pass-through)."""
    ctx = EvaluationContext(state={}, locals={})

    assert evaluate_value(ctx, 42) == 42
    assert evaluate_value(ctx, 3.14) == 3.14
    assert evaluate_value(ctx, True) is True
    assert evaluate_value(ctx, None) is None


def test_evaluate_value_string_with_ref():
    """Test evaluating string value with reference."""
    ctx = EvaluationContext(
        state={"name": "Alice"},
        locals={}
    )
    result = evaluate_value(ctx, "Hello ${name}")
    assert result == "Hello Alice"


def test_evaluate_value_list():
    """Test evaluating list with references."""
    ctx = EvaluationContext(
        state={"a": 1, "b": 2},
        locals={}
    )
    result = evaluate_value(ctx, ["${a}", "${b}", 3])
    assert result == [1, 2, 3]


def test_evaluate_value_nested_list():
    """Test evaluating nested list."""
    ctx = EvaluationContext(
        state={"x": "value"},
        locals={}
    )
    result = evaluate_value(ctx, [["${x}"], [1, 2]])
    assert result == [["value"], [1, 2]]


def test_evaluate_value_dict():
    """Test evaluating dict with references."""
    ctx = EvaluationContext(
        state={"name": "Alice", "age": 25},
        locals={}
    )
    result = evaluate_value(ctx, {
        "user_name": "${name}",
        "user_age": "${age}",
        "static": "value"
    })
    assert result == {
        "user_name": "Alice",
        "user_age": 25,
        "static": "value"
    }


def test_evaluate_value_nested_dict():
    """Test evaluating nested dict."""
    ctx = EvaluationContext(
        state={"email": "test@example.com"},
        locals={}
    )
    result = evaluate_value(ctx, {
        "user": {
            "contact": {
                "email": "${email}"
            }
        }
    })
    assert result["user"]["contact"]["email"] == "test@example.com"


def test_evaluate_value_complex_structure():
    """Test evaluating complex mixed structure."""
    ctx = EvaluationContext(
        state={"name": "Alice", "scores": [95, 87, 91]},
        locals={}
    )
    result = evaluate_value(ctx, {
        "student": "${name}",
        "grades": "${scores}",
        "summary": "Student ${name} has scores: ${scores}"
    })
    assert result["student"] == "Alice"
    assert result["grades"] == [95, 87, 91]
    assert "Student Alice" in result["summary"]


# =============================================================================
# Condition Evaluation Tests
# =============================================================================


def test_evaluate_condition_simple_comparison():
    """Test evaluating simple comparison condition."""
    ctx = EvaluationContext(
        state={"value": 10},
        locals={}
    )
    result = evaluate_condition(ctx, "${value} > 5")
    assert result is True


def test_evaluate_condition_equality():
    """Test evaluating equality condition."""
    ctx = EvaluationContext(
        state={"status": "active"},
        locals={}
    )
    result = evaluate_condition(ctx, "${status} == 'active'")
    assert result is True


def test_evaluate_condition_inequality():
    """Test evaluating inequality condition."""
    ctx = EvaluationContext(
        state={"status": "inactive"},
        locals={}
    )
    result = evaluate_condition(ctx, "${status} != 'active'")
    assert result is True


def test_evaluate_condition_boolean_ops():
    """Test evaluating condition with boolean operators."""
    ctx = EvaluationContext(
        state={"a": True, "b": False},
        locals={}
    )
    assert evaluate_condition(ctx, "${a} and ${b}") is False
    assert evaluate_condition(ctx, "${a} or ${b}") is True


def test_evaluate_condition_not_operator():
    """Test evaluating condition with not operator."""
    ctx = EvaluationContext(
        state={"flag": False},
        locals={}
    )
    result = evaluate_condition(ctx, "not ${flag}")
    assert result is True


def test_evaluate_condition_arithmetic():
    """Test evaluating condition with arithmetic."""
    ctx = EvaluationContext(
        state={"x": 10, "y": 5},
        locals={}
    )
    result = evaluate_condition(ctx, "${x} + ${y} == 15")
    assert result is True


def test_evaluate_condition_with_functions():
    """Test evaluating condition with safe functions."""
    ctx = EvaluationContext(
        state={"items": [1, 2, 3]},
        locals={}
    )
    result = evaluate_condition(ctx, "len(${items}) == 3")
    assert result is True


def test_evaluate_condition_multiple_refs():
    """Test evaluating condition with multiple references."""
    ctx = EvaluationContext(
        state={"a": 10, "b": 20, "c": 30},
        locals={}
    )
    result = evaluate_condition(ctx, "${a} < ${b} < ${c}")
    assert result is True


def test_evaluate_condition_subscript():
    """Test evaluating condition with subscript access."""
    ctx = EvaluationContext(
        state={"items": [10, 20, 30]},
        locals={}
    )
    result = evaluate_condition(ctx, "${items}[0] == 10")
    assert result is True


def test_evaluate_condition_in_operator():
    """Test evaluating condition with 'in' operator."""
    ctx = EvaluationContext(
        state={"value": "test", "items": ["test", "example"]},
        locals={}
    )
    result = evaluate_condition(ctx, "${value} in ${items}")
    assert result is True


def test_evaluate_condition_not_in_operator():
    """Test evaluating condition with 'not in' operator."""
    ctx = EvaluationContext(
        state={"value": "missing", "items": ["test", "example"]},
        locals={}
    )
    result = evaluate_condition(ctx, "${value} not in ${items}")
    assert result is True


# =============================================================================
# Condition Evaluation Error Tests
# =============================================================================


def test_evaluate_condition_empty_string():
    """Test that empty condition raises EvaluationError."""
    ctx = EvaluationContext(state={}, locals={})

    with pytest.raises(EvaluationError, match="Condition expression resolved to empty string"):
        evaluate_condition(ctx, "")


def test_evaluate_condition_disallowed_node():
    """Test that disallowed AST nodes raise EvaluationError."""
    ctx = EvaluationContext(state={"x": 5}, locals={})

    # Dict literal is not in ALLOWED_NODES
    with pytest.raises(EvaluationError, match="Disallowed expression node"):
        evaluate_condition(ctx, "{'key': 'value'}")


def test_evaluate_condition_unsafe_function():
    """Test that unsafe functions raise EvaluationError."""
    ctx = EvaluationContext(state={"x": 5}, locals={})

    with pytest.raises(EvaluationError, match="Only whitelisted helper functions"):
        evaluate_condition(ctx, "eval('1 + 1')")


def test_evaluate_condition_unknown_variable():
    """Test that unknown variables raise EvaluationError."""
    ctx = EvaluationContext(state={}, locals={})

    with pytest.raises(EvaluationError, match="Unknown variable"):
        evaluate_condition(ctx, "unknown_var == 5")


def test_evaluate_condition_disallowed_operator():
    """Test that disallowed operators raise EvaluationError."""
    ctx = EvaluationContext(state={"x": 5}, locals={})

    # Power operator (**) is not in ALLOWED_BINOPS
    with pytest.raises(EvaluationError, match="Operator .* is not allowed"):
        evaluate_condition(ctx, "${x} ** 2 == 25")


# =============================================================================
# Safe Functions Tests
# =============================================================================


@pytest.mark.parametrize("func_name,input_val,expected,expr_suffix", [
    ("len", [1, 2, 3], 3, "== 3"),
    ("min", [1, 2, 3], 1, "== 1"),
    ("max", [1, 2, 3], 3, "== 3"),
    ("sum", [1, 2, 3], 6, "== 6"),
    ("str", 42, "42", "== '42'"),
    ("int", "42", 42, "== 42"),
    ("float", "3.14", 3.14, "== 3.14"),
])
def test_evaluate_condition_safe_functions(func_name, input_val, expected, expr_suffix):
    """Test various safe functions in conditions."""
    ctx = EvaluationContext(
        state={"value": input_val},
        locals={}
    )
    result = evaluate_condition(ctx, f"{func_name}(${{value}}) {expr_suffix}")
    assert result is True


def test_evaluate_condition_any_function():
    """Test 'any' function in condition."""
    ctx = EvaluationContext(
        state={"flags": [False, True, False]},
        locals={}
    )
    result = evaluate_condition(ctx, "any(${flags})")
    assert result is True


def test_evaluate_condition_all_function():
    """Test 'all' function in condition."""
    ctx = EvaluationContext(
        state={"flags": [True, True, True]},
        locals={}
    )
    result = evaluate_condition(ctx, "all(${flags})")
    assert result is True


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_evaluate_value_empty_dict():
    """Test evaluating empty dict."""
    ctx = EvaluationContext(state={}, locals={})
    result = evaluate_value(ctx, {})
    assert result == {}


def test_evaluate_value_empty_list():
    """Test evaluating empty list."""
    ctx = EvaluationContext(state={}, locals={})
    result = evaluate_value(ctx, [])
    assert result == []


def test_render_template_only_literals():
    """Test rendering template with only literals after reference resolution."""
    ctx = EvaluationContext(
        state={"a": "X", "b": "Y"},
        locals={}
    )
    result = render_template(ctx, "${a}${b}")
    assert result == "XY"


def test_evaluate_condition_with_list_literal():
    """Test evaluating condition with list literal."""
    ctx = EvaluationContext(
        state={"x": 1},
        locals={}
    )
    result = evaluate_condition(ctx, "${x} in [1, 2, 3]")
    assert result is True


def test_evaluate_condition_complex_boolean_logic():
    """Test evaluating complex boolean logic."""
    ctx = EvaluationContext(
        state={"a": 10, "b": 20, "c": True},
        locals={}
    )
    result = evaluate_condition(ctx, "(${a} < ${b}) and ${c}")
    assert result is True


def test_resolve_reference_negative_index():
    """Test resolving reference with negative array index."""
    ctx = EvaluationContext(
        state={"items": ["a", "b", "c"]},
        locals={}
    )
    ref = parse_reference_string("items[-1]")
    # Python supports negative indexing
    result = resolve_reference(ctx, ref)
    assert result == "c"
