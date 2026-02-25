"""
Unit tests for expression validation and typechecking logic.

Tests the core expression parsing, typechecking, and evaluation functionality.
"""
import pytest

from seer.core.expr.parser import (
    IndexSegment,
    PropertySegment,
    ReferenceExpr,
    TemplateLiteral,
    TemplateReference,
    collect_unique_references,
    iterate_value_references,
    parse_reference_string,
    parse_template,
)
from seer.core.expr.typecheck import (
    Scope,
    TypeCheckError,
    TypeEnvironment,
    resolve_schema_path,
    typecheck_reference,
)
from seer.core.expr.evaluator import (
    EvaluationContext,
    EvaluationError,
    evaluate_condition,
    evaluate_value,
    render_template,
    resolve_reference,
)


# =============================================================================
# Expression Parser Tests
# =============================================================================


@pytest.mark.unit
class TestParseReferenceString:
    """Tests for parse_reference_string function."""

    def test_simple_root_reference(self):
        """Test parsing a simple root reference."""
        result = parse_reference_string("input")
        assert result.raw == "input"
        assert result.root == "input"
        assert result.segments == ()

    def test_single_property_access(self):
        """Test parsing single property access."""
        result = parse_reference_string("input.value")
        assert result.root == "input"
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], PropertySegment)
        assert result.segments[0].key == "value"

    def test_nested_property_access(self):
        """Test parsing nested property access."""
        result = parse_reference_string("nodes.n1.output.data")
        assert result.root == "nodes"
        assert len(result.segments) == 3
        assert result.segments[0].key == "n1"
        assert result.segments[1].key == "output"
        assert result.segments[2].key == "data"

    def test_numeric_index_access(self):
        """Test parsing numeric index access."""
        result = parse_reference_string("items[0]")
        assert result.root == "items"
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], IndexSegment)
        assert result.segments[0].index == 0

    def test_string_index_access(self):
        """Test parsing string index access with quotes."""
        result = parse_reference_string("data['key']")
        assert result.root == "data"
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], IndexSegment)
        assert result.segments[0].index == "key"

    def test_mixed_property_and_index_access(self):
        """Test parsing mixed property and index access."""
        result = parse_reference_string("nodes.n1.output[0].value")
        assert result.root == "nodes"
        assert len(result.segments) == 4
        assert isinstance(result.segments[0], PropertySegment)
        assert result.segments[0].key == "n1"
        assert isinstance(result.segments[1], PropertySegment)
        assert result.segments[1].key == "output"
        assert isinstance(result.segments[2], IndexSegment)
        assert result.segments[2].index == 0
        assert isinstance(result.segments[3], PropertySegment)
        assert result.segments[3].key == "value"

    def test_empty_reference_raises_error(self):
        """Test that empty reference raises ValueError."""
        with pytest.raises(ValueError, match="Reference cannot be empty"):
            parse_reference_string("")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only reference raises ValueError."""
        with pytest.raises(ValueError, match="Reference cannot be empty"):
            parse_reference_string("   ")

    def test_unclosed_bracket_raises_error(self):
        """Test that unclosed bracket raises ValueError."""
        with pytest.raises(ValueError, match="Unclosed"):
            parse_reference_string("items[0")

    def test_empty_bracket_raises_error(self):
        """Test that empty bracket accessor raises ValueError."""
        with pytest.raises(ValueError, match="Empty bracket"):
            parse_reference_string("items[]")

    def test_invalid_bracket_content_raises_error(self):
        """Test that invalid bracket content raises ValueError."""
        with pytest.raises(ValueError, match="Bracket accessor must be"):
            parse_reference_string("items[invalid]")


@pytest.mark.unit
class TestParseTemplate:
    """Tests for parse_template function."""

    def test_plain_text_returns_literal(self):
        """Test that plain text returns a single literal token."""
        tokens = parse_template("plain text")
        assert len(tokens) == 1
        assert isinstance(tokens[0], TemplateLiteral)
        assert tokens[0].text == "plain text"

    def test_single_reference(self):
        """Test parsing single reference in template."""
        tokens = parse_template("${input.value}")
        assert len(tokens) == 1
        assert isinstance(tokens[0], TemplateReference)
        assert tokens[0].reference.root == "input"

    def test_reference_with_surrounding_text(self):
        """Test parsing reference with surrounding text."""
        tokens = parse_template("Hello ${name}, welcome!")
        assert len(tokens) == 3
        assert isinstance(tokens[0], TemplateLiteral)
        assert tokens[0].text == "Hello "
        assert isinstance(tokens[1], TemplateReference)
        assert tokens[1].reference.root == "name"
        assert isinstance(tokens[2], TemplateLiteral)
        assert tokens[2].text == ", welcome!"

    def test_multiple_references(self):
        """Test parsing multiple references."""
        tokens = parse_template("${first} and ${second}")
        assert len(tokens) == 3
        assert isinstance(tokens[0], TemplateReference)
        assert isinstance(tokens[1], TemplateLiteral)
        assert tokens[1].text == " and "
        assert isinstance(tokens[2], TemplateReference)

    def test_adjacent_references(self):
        """Test parsing adjacent references."""
        tokens = parse_template("${a}${b}")
        assert len(tokens) == 2
        assert all(isinstance(t, TemplateReference) for t in tokens)


@pytest.mark.unit
class TestIterateValueReferences:
    """Tests for iterate_value_references function."""

    def test_string_with_reference(self):
        """Test extracting references from string."""
        refs = list(iterate_value_references("${input.value}"))
        assert len(refs) == 1
        assert refs[0].root == "input"

    def test_nested_dict(self):
        """Test extracting references from nested dict."""
        value = {
            "field1": "${input.a}",
            "nested": {
                "field2": "${input.b}"
            }
        }
        refs = list(iterate_value_references(value))
        assert len(refs) == 2
        roots = {r.root for r in refs}
        assert roots == {"input"}

    def test_list_with_references(self):
        """Test extracting references from list."""
        value = ["${item1}", "${item2}"]
        refs = list(iterate_value_references(value))
        assert len(refs) == 2


@pytest.mark.unit
class TestCollectUniqueReferences:
    """Tests for collect_unique_references function."""

    def test_deduplicates_references(self):
        """Test that duplicate references are removed."""
        values = ["${input.value}", "${input.value}", "${other}"]
        refs = collect_unique_references(values)
        assert len(refs) == 2

    def test_preserves_order(self):
        """Test that discovery order is preserved."""
        values = ["${z}", "${a}", "${m}"]
        refs = collect_unique_references(values)
        assert [r.root for r in refs] == ["z", "a", "m"]


# =============================================================================
# Type Environment Tests
# =============================================================================


@pytest.mark.unit
class TestTypeEnvironment:
    """Tests for TypeEnvironment class."""

    def test_register_and_get(self):
        """Test registering and retrieving schemas."""
        env = TypeEnvironment()
        schema = {"type": "string"}
        env.register("input", schema)
        assert env.get("input") == schema

    def test_get_unregistered_returns_none(self):
        """Test that getting unregistered symbol returns None."""
        env = TypeEnvironment()
        assert env.get("unknown") is None

    def test_require_unregistered_raises_error(self):
        """Test that require raises error for unregistered symbol."""
        env = TypeEnvironment()
        with pytest.raises(TypeCheckError, match="No schema registered"):
            env.require("unknown")

    def test_register_duplicate_same_schema(self):
        """Test that registering same schema twice is allowed."""
        env = TypeEnvironment()
        schema = {"type": "string"}
        env.register("input", schema)
        env.register("input", schema)  # Should not raise

    def test_register_duplicate_different_schema_raises(self):
        """Test that registering different schema raises error."""
        env = TypeEnvironment()
        env.register("input", {"type": "string"})
        with pytest.raises(TypeCheckError, match="already registered"):
            env.register("input", {"type": "number"})

    def test_as_dict(self):
        """Test converting environment to dict."""
        env = TypeEnvironment()
        env.register("a", {"type": "string"})
        env.register("b", {"type": "number"})
        result = env.as_dict()
        assert "a" in result
        assert "b" in result


@pytest.mark.unit
class TestScope:
    """Tests for Scope class."""

    def test_resolve_from_env(self):
        """Test resolving symbol from environment."""
        env = TypeEnvironment()
        env.register("input", {"type": "string"})
        scope = Scope(env=env)
        assert scope.resolve("input") == {"type": "string"}

    def test_resolve_from_locals(self):
        """Test resolving symbol from locals takes precedence."""
        env = TypeEnvironment()
        env.register("item", {"type": "string"})
        scope = Scope(env=env, locals={"item": {"type": "object"}})
        assert scope.resolve("item") == {"type": "object"}

    def test_nested_scope(self):
        """Test creating nested scope preserves parent locals."""
        env = TypeEnvironment()
        scope = Scope(env=env, locals={"item": {"type": "string"}})
        nested = scope.nested()
        nested.locals["index"] = {"type": "integer"}
        assert "item" in nested.locals
        assert "index" in nested.locals
        assert "index" not in scope.locals  # Parent unchanged


# =============================================================================
# Typecheck Reference Tests
# =============================================================================


@pytest.mark.unit
class TestResolveSchemaPath:
    """Tests for resolve_schema_path function."""

    def test_resolve_property(self):
        """Test resolving property from object schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        segments = [PropertySegment("name")]
        result = resolve_schema_path(schema, segments)
        assert result == {"type": "string"}

    def test_resolve_nested_property(self):
        """Test resolving nested property."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        segments = [PropertySegment("user"), PropertySegment("name")]
        result = resolve_schema_path(schema, segments)
        assert result == {"type": "string"}

    def test_resolve_array_index(self):
        """Test resolving array index."""
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        segments = [IndexSegment(0)]
        result = resolve_schema_path(schema, segments)
        assert result == {"type": "string"}

    def test_resolve_additional_properties(self):
        """Test resolving through additionalProperties."""
        schema = {
            "type": "object",
            "additionalProperties": {"type": "number"}
        }
        segments = [PropertySegment("anyKey")]
        result = resolve_schema_path(schema, segments)
        assert result == {"type": "number"}

    def test_missing_property_raises_error(self):
        """Test that missing property raises TypeCheckError."""
        schema = {
            "type": "object",
            "properties": {}
        }
        segments = [PropertySegment("missing")]
        with pytest.raises(TypeCheckError, match="not declared"):
            resolve_schema_path(schema, segments)

    def test_property_access_on_non_object_raises_error(self):
        """Test that property access on non-object raises error."""
        schema = {"type": "string"}
        segments = [PropertySegment("name")]
        with pytest.raises(TypeCheckError, match="Cannot access property"):
            resolve_schema_path(schema, segments)

    def test_numeric_index_on_non_array_raises_error(self):
        """Test that numeric index on non-array raises error."""
        schema = {"type": "object"}
        segments = [IndexSegment(0)]
        with pytest.raises(TypeCheckError, match="only valid on array"):
            resolve_schema_path(schema, segments)

    def test_resolve_property_with_additional_properties_true(self):
        """Property access allowed when additionalProperties is true (boolean)."""
        schema = {
            "type": "object",
            "additionalProperties": True
        }
        segments = [PropertySegment("anyField")]
        result = resolve_schema_path(schema, segments)
        assert result == {}  # Permissive empty schema

    def test_resolve_nested_with_additional_properties_true(self):
        """Nested property access through additionalProperties: true."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "additionalProperties": True
                }
            }
        }
        segments = [PropertySegment("data"), PropertySegment("customField")]
        result = resolve_schema_path(schema, segments)
        assert result == {}

    def test_form_hosted_trigger_pattern(self):
        """Real-world pattern: form.hosted event_schema with custom fields."""
        # This is the exact schema structure from trigger_registry.py
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {
                    "type": "object",
                    "description": "Form submission data with custom field values",
                    "additionalProperties": True
                }
            }
        }
        # Should resolve ${trigger.data.topic} for custom form fields
        segments = [PropertySegment("data"), PropertySegment("topic")]
        result = resolve_schema_path(schema, segments)
        assert result == {}

    def test_string_index_with_additional_properties_true(self):
        """String index access allowed when additionalProperties is true."""
        schema = {
            "type": "object",
            "additionalProperties": True
        }
        segments = [IndexSegment("dynamicKey")]
        result = resolve_schema_path(schema, segments)
        assert result == {}


@pytest.mark.unit
class TestTypecheckReference:
    """Tests for typecheck_reference function."""

    def test_simple_reference(self):
        """Test typechecking simple reference."""
        env = TypeEnvironment()
        env.register("input", {"type": "object", "properties": {"value": {"type": "string"}}})
        scope = Scope(env=env)
        ref = parse_reference_string("input.value")
        result = typecheck_reference(ref, scope)
        assert result == {"type": "string"}

    def test_nested_reference(self):
        """Test typechecking nested reference."""
        env = TypeEnvironment()
        env.register("nodes", {
            "type": "object",
            "properties": {
                "n1": {
                    "type": "object",
                    "properties": {
                        "output": {"type": "string"}
                    }
                }
            }
        })
        scope = Scope(env=env)
        ref = parse_reference_string("nodes.n1.output")
        result = typecheck_reference(ref, scope)
        assert result == {"type": "string"}

    def test_unknown_root_raises_error(self):
        """Test that unknown root symbol raises error."""
        env = TypeEnvironment()
        scope = Scope(env=env)
        ref = parse_reference_string("unknown.value")
        with pytest.raises(TypeCheckError):
            typecheck_reference(ref, scope)

    def test_invalid_path_raises_error(self):
        """Test that invalid path raises error."""
        env = TypeEnvironment()
        env.register("input", {"type": "object", "properties": {}})
        scope = Scope(env=env)
        ref = parse_reference_string("input.missing")
        with pytest.raises(TypeCheckError):
            typecheck_reference(ref, scope)


# =============================================================================
# Evaluation Context Tests
# =============================================================================


@pytest.mark.unit
class TestEvaluationContext:
    """Tests for EvaluationContext class."""

    def test_with_locals_merges(self):
        """Test that with_locals merges new locals."""
        ctx = EvaluationContext(state={}, locals={"a": 1})
        new_ctx = ctx.with_locals({"b": 2})
        assert new_ctx.locals["a"] == 1
        assert new_ctx.locals["b"] == 2

    def test_with_locals_does_not_mutate_original(self):
        """Test that with_locals does not mutate original context."""
        ctx = EvaluationContext(state={}, locals={"a": 1})
        ctx.with_locals({"b": 2})
        assert "b" not in ctx.locals


# =============================================================================
# Resolve Reference Tests
# =============================================================================


@pytest.mark.unit
class TestResolveReference:
    """Tests for resolve_reference function."""

    def test_resolve_from_state(self):
        """Test resolving reference from state."""
        ctx = EvaluationContext(
            state={"input": {"value": "hello"}},
            locals={}
        )
        ref = parse_reference_string("input.value")
        result = resolve_reference(ctx, ref)
        assert result == "hello"

    def test_resolve_from_locals(self):
        """Test resolving reference from locals."""
        ctx = EvaluationContext(
            state={},
            locals={"item": {"name": "test"}}
        )
        ref = parse_reference_string("item.name")
        result = resolve_reference(ctx, ref)
        assert result == "test"

    def test_locals_take_precedence_over_state(self):
        """Test that locals take precedence over state."""
        ctx = EvaluationContext(
            state={"item": {"name": "from_state"}},
            locals={"item": {"name": "from_locals"}}
        )
        ref = parse_reference_string("item.name")
        result = resolve_reference(ctx, ref)
        assert result == "from_locals"

    def test_resolve_array_index(self):
        """Test resolving array index."""
        ctx = EvaluationContext(
            state={"items": ["a", "b", "c"]},
            locals={}
        )
        ref = parse_reference_string("items[1]")
        result = resolve_reference(ctx, ref)
        assert result == "b"

    def test_resolve_string_index(self):
        """Test resolving string index."""
        ctx = EvaluationContext(
            state={"data": {"key-with-dash": "value"}},
            locals={}
        )
        ref = parse_reference_string("data['key-with-dash']")
        result = resolve_reference(ctx, ref)
        assert result == "value"

    def test_unknown_root_raises_error(self):
        """Test that unknown root raises EvaluationError."""
        ctx = EvaluationContext(state={}, locals={})
        ref = parse_reference_string("unknown")
        with pytest.raises(EvaluationError, match="Unknown reference root"):
            resolve_reference(ctx, ref)

    def test_missing_property_raises_error(self):
        """Test that missing property raises EvaluationError."""
        ctx = EvaluationContext(
            state={"input": {}},
            locals={}
        )
        ref = parse_reference_string("input.missing")
        with pytest.raises(EvaluationError, match="not found"):
            resolve_reference(ctx, ref)

    def test_index_out_of_range_raises_error(self):
        """Test that index out of range raises EvaluationError."""
        ctx = EvaluationContext(
            state={"items": ["a"]},
            locals={}
        )
        ref = parse_reference_string("items[5]")
        with pytest.raises(EvaluationError, match="out of range"):
            resolve_reference(ctx, ref)

    def test_resolve_config(self):
        """Test resolving reference from config."""
        ctx = EvaluationContext(
            state={},
            locals={},
            config={"api_key": "secret"}
        )
        ref = parse_reference_string("config.api_key")
        result = resolve_reference(ctx, ref)
        assert result == "secret"


# =============================================================================
# Evaluate Value Tests
# =============================================================================


@pytest.mark.unit
class TestEvaluateValue:
    """Tests for evaluate_value function."""

    def test_evaluate_string_template(self):
        """Test evaluating string with template."""
        ctx = EvaluationContext(
            state={"name": "World"},
            locals={}
        )
        result = evaluate_value(ctx, "Hello ${name}!")
        assert result == "Hello World!"

    def test_evaluate_plain_string(self):
        """Test evaluating plain string without template."""
        ctx = EvaluationContext(state={}, locals={})
        result = evaluate_value(ctx, "plain text")
        assert result == "plain text"

    def test_evaluate_nested_dict(self):
        """Test evaluating nested dict with templates."""
        ctx = EvaluationContext(
            state={"value": "test"},
            locals={}
        )
        result = evaluate_value(ctx, {
            "field": "${value}",
            "nested": {"inner": "${value}"}
        })
        assert result == {"field": "test", "nested": {"inner": "test"}}

    def test_evaluate_list(self):
        """Test evaluating list with templates."""
        ctx = EvaluationContext(
            state={"a": "1", "b": "2"},
            locals={}
        )
        result = evaluate_value(ctx, ["${a}", "${b}"])
        assert result == ["1", "2"]

    def test_evaluate_non_string_passthrough(self):
        """Test that non-string values pass through unchanged."""
        ctx = EvaluationContext(state={}, locals={})
        assert evaluate_value(ctx, 42) == 42
        assert evaluate_value(ctx, 3.14) == 3.14
        assert evaluate_value(ctx, True) is True
        assert evaluate_value(ctx, None) is None


@pytest.mark.unit
class TestRenderTemplate:
    """Tests for render_template function."""

    def test_single_reference_returns_original_type(self):
        """Test that single reference returns original value type."""
        ctx = EvaluationContext(
            state={"data": {"nested": "value"}},
            locals={}
        )
        result = render_template(ctx, "${data}")
        assert result == {"nested": "value"}
        assert isinstance(result, dict)

    def test_multiple_references_returns_string(self):
        """Test that multiple references returns concatenated string."""
        ctx = EvaluationContext(
            state={"a": "hello", "b": "world"},
            locals={}
        )
        result = render_template(ctx, "${a} ${b}")
        assert result == "hello world"

    def test_none_value_renders_as_empty_string(self):
        """Test that None values render as empty string."""
        ctx = EvaluationContext(
            state={"value": None},
            locals={}
        )
        result = render_template(ctx, "prefix${value}suffix")
        assert result == "prefixsuffix"


# =============================================================================
# Evaluate Condition Tests
# =============================================================================


@pytest.mark.unit
class TestEvaluateCondition:
    """Tests for evaluate_condition function."""

    def test_simple_equality(self):
        """Test simple equality condition."""
        ctx = EvaluationContext(
            state={"value": 5},
            locals={}
        )
        assert evaluate_condition(ctx, "${value} == 5") is True
        assert evaluate_condition(ctx, "${value} == 10") is False

    def test_comparison_operators(self):
        """Test comparison operators."""
        ctx = EvaluationContext(
            state={"num": 10},
            locals={}
        )
        assert evaluate_condition(ctx, "${num} > 5") is True
        assert evaluate_condition(ctx, "${num} < 5") is False
        assert evaluate_condition(ctx, "${num} >= 10") is True
        assert evaluate_condition(ctx, "${num} <= 10") is True

    def test_boolean_operators(self):
        """Test boolean operators."""
        ctx = EvaluationContext(
            state={"a": True, "b": False},
            locals={}
        )
        assert evaluate_condition(ctx, "${a} and ${b}") is False
        assert evaluate_condition(ctx, "${a} or ${b}") is True
        assert evaluate_condition(ctx, "not ${b}") is True

    def test_in_operator(self):
        """Test 'in' operator."""
        ctx = EvaluationContext(
            state={"item": "b", "items": ["a", "b", "c"]},
            locals={}
        )
        assert evaluate_condition(ctx, "${item} in ${items}") is True

    def test_safe_functions(self):
        """Test safe built-in functions."""
        ctx = EvaluationContext(
            state={"items": [1, 2, 3]},
            locals={}
        )
        assert evaluate_condition(ctx, "len(${items}) == 3") is True
        assert evaluate_condition(ctx, "sum(${items}) == 6") is True
        assert evaluate_condition(ctx, "max(${items}) == 3") is True

    def test_empty_condition_raises_error(self):
        """Test that empty condition raises error."""
        ctx = EvaluationContext(state={}, locals={})
        with pytest.raises(EvaluationError, match="empty string"):
            evaluate_condition(ctx, "")

    def test_unsafe_function_raises_error(self):
        """Test that unsafe functions raise error."""
        ctx = EvaluationContext(
            state={"cmd": "echo test"},
            locals={}
        )
        with pytest.raises(EvaluationError, match="whitelisted"):
            evaluate_condition(ctx, "__import__('os').system(${cmd})")

    def test_unknown_variable_raises_error(self):
        """Test that unknown variable raises error."""
        ctx = EvaluationContext(state={}, locals={})
        with pytest.raises(EvaluationError, match="Unknown variable"):
            evaluate_condition(ctx, "unknown_var == 5")
