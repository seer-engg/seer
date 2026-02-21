# pylint: disable=unused-argument
# Reason: Test fixtures may not be directly used in all tests
"""
Integration tests for compile-time condition type validation.

Tests the full validate_references() flow with IfNode condition expressions,
ensuring type errors are caught at compile time with proper error context.
"""

from __future__ import annotations

import pytest

from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.errors import ErrorCode, ValidationPhaseError
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


# =============================================================================
# Test Fixtures
# =============================================================================


def _create_mock_tool() -> ToolDefinition:
    """Create a mock test.tool for workflow specs."""

    def handler(inputs, config, context):
        return inputs.get("value", "")

    async def async_handler(inputs, config, context):
        return inputs.get("value", "")

    return ToolDefinition(
        name="test.tool",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": ["string", "number", "boolean", "null"]}},
            "additionalProperties": False,
        },
        output_schema={"type": ["string", "number", "boolean", "null"]},
        handler=handler,
        async_handler=async_handler,
    )


def _create_workflow_spec(trigger_schema: dict, condition: str) -> dict:
    """Create a minimal workflow spec with an if node."""
    return {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "event_schema": trigger_schema,
            }
        ],
        "nodes": [
            {
                "id": "check_condition",
                "type": "if",
                "condition": condition,
            },
            {
                "id": "true_branch",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "true_executed"},
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "check_condition", "type": "trigger"},
            {"source": "check_condition", "target": "true_branch", "type": "conditional_true"},
        ],
    }


def _validate_workflow(spec_dict: dict) -> None:
    """Parse and validate a workflow spec, raising on errors."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    tool_registry.register(_create_mock_tool())

    spec = parse_workflow_spec(spec_dict)
    type_env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )
    validate_references(spec, type_env)


def _get_validation_errors(spec_dict: dict) -> list:
    """Get validation errors from a workflow spec."""
    try:
        _validate_workflow(spec_dict)
        return []
    except ValidationPhaseError as exc:
        return exc.errors


# =============================================================================
# Valid Expression Tests
# =============================================================================


class TestValidConditionExpressions:
    """Tests for valid condition expressions that should pass validation."""

    def test_valid_numeric_comparison(self) -> None:
        """${number} > 10 should pass when number is numeric."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"count": {"type": "number"}}
            },
            condition="${test_trigger.count} > 10",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_string_equality(self) -> None:
        """${status} == 'active' should pass when status is string."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}}
            },
            condition="${test_trigger.status} == 'active'",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_len_array(self) -> None:
        """len(${array}) > 0 should pass when items is array."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}}
                }
            },
            condition="len(${test_trigger.items}) > 0",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_in_operator_array(self) -> None:
        """'x' in ${array} should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            },
            condition="'important' in ${test_trigger.tags}",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_chained_comparison(self) -> None:
        """0 < ${num} < 100 should pass for numeric type."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"score": {"type": "number"}}
            },
            condition="0 < ${test_trigger.score} < 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_nullable_comparison(self) -> None:
        """Nullable types should still allow comparisons."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"value": {"type": ["string", "null"]}}
            },
            condition="${test_trigger.value} == 'test'",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_valid_boolean_operators(self) -> None:
        """Combining conditions with and/or should work."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            },
            condition="${test_trigger.a} > 0 and ${test_trigger.b} < 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []


# =============================================================================
# Invalid Expression Tests
# =============================================================================


class TestInvalidConditionExpressions:
    """Tests for invalid condition expressions that should fail validation."""

    def test_invalid_string_greater_than_number(self) -> None:
        """${string} > 100 should fail with TYPE_MISMATCH."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}}
            },
            condition="${test_trigger.name} > 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        err = type_mismatch_errors[0]
        assert err.node_id == "check_condition"
        assert err.location == "condition"
        assert err.expression == "${test_trigger.name} > 100"

    def test_invalid_len_on_number(self) -> None:
        """len(${number}) should fail."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"count": {"type": "number"}}
            },
            condition="len(${test_trigger.count}) > 0",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("len" in e.message.lower() or "iterable" in e.message.lower()
                   for e in type_mismatch_errors)

    def test_invalid_in_on_number(self) -> None:
        """'x' in ${number} should fail."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"count": {"type": "number"}}
            },
            condition="'x' in ${test_trigger.count}",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("`in`" in e.message or "container" in e.message.lower()
                   for e in type_mismatch_errors)

    def test_invalid_chained_comparison_string(self) -> None:
        """0 < ${string} < 100 should fail."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}}
            },
            condition="0 < ${test_trigger.text} < 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1


# =============================================================================
# Common Typo Hint Tests
# =============================================================================


class TestCommonTypoHints:
    """Tests for common typo detection with helpful hints."""

    def test_true_typo_suggests_True(self) -> None:
        """'true' should suggest 'True'."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="${test_trigger.flag} == true",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("true" in e.message and "True" in e.message
                   for e in type_mismatch_errors)

    def test_false_typo_suggests_False(self) -> None:
        """'false' should suggest 'False'."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="${test_trigger.flag} == false",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("false" in e.message and "False" in e.message
                   for e in type_mismatch_errors)

    def test_null_typo_suggests_None(self) -> None:
        """'null' should suggest 'None'."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"data": {"type": ["object", "null"]}}
            },
            condition="${test_trigger.data} == null",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("null" in e.message and "None" in e.message
                   for e in type_mismatch_errors)

    def test_unknown_variable_error(self) -> None:
        """Unknown variable 'foo' should produce error."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="foo and ${test_trigger.flag}",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("foo" in e.message for e in type_mismatch_errors)


# =============================================================================
# Error Context Tests
# =============================================================================


class TestErrorContext:
    """Tests that error context (node_id, location, expression) is populated."""

    def test_error_has_node_id(self) -> None:
        """Type mismatch errors should have node_id."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}}
            },
            condition="${test_trigger.name} > 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert type_mismatch_errors[0].node_id == "check_condition"

    def test_error_has_location(self) -> None:
        """Type mismatch errors should have location='condition'."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}}
            },
            condition="${test_trigger.name} > 100",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert type_mismatch_errors[0].location == "condition"

    def test_error_has_expression(self) -> None:
        """Type mismatch errors should include the expression."""
        condition = "${test_trigger.name} > 100"
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}}
            },
            condition=condition,
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert type_mismatch_errors[0].expression == condition


# =============================================================================
# Unsupported Pattern Tests
# =============================================================================


class TestUnsupportedPatterns:
    """Tests for unsupported expression patterns with helpful errors."""

    def test_method_call_error(self) -> None:
        """Method calls should produce helpful error."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}}
            },
            condition="${test_trigger.text}.lower() == 'test'",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1
        assert any("method" in e.message.lower() or "lower" in e.message
                   for e in type_mismatch_errors)

    def test_numeric_subscript_on_object(self) -> None:
        """Numeric index on object should produce error."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "additionalProperties": {"type": "string"}}
                }
            },
            condition="${test_trigger.data}[0] == 'test'",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]

        assert len(type_mismatch_errors) >= 1


# =============================================================================
# Regression Tests - Ensure Valid Workflows Still Pass
# =============================================================================


class TestNoRegressions:
    """Regression tests to ensure previously valid workflows still pass."""

    def test_simple_boolean_condition(self) -> None:
        """Simple boolean condition should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="${test_trigger.flag}",
        )
        # Should not raise
        _validate_workflow(spec)

    def test_comparison_with_python_true(self) -> None:
        """Comparison with Python True should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="${test_trigger.flag} == True",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_comparison_with_python_none(self) -> None:
        """Comparison with Python None should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"data": {"type": ["object", "null"]}}
            },
            condition="${test_trigger.data} == None",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_negation(self) -> None:
        """not ${flag} should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"flag": {"type": "boolean"}}
            },
            condition="not ${test_trigger.flag}",
        )
        errors = _get_validation_errors(spec)
        type_mismatch_errors = [e for e in errors if e.code == ErrorCode.TYPE_MISMATCH]
        assert type_mismatch_errors == []

    def test_existing_passing_workflow(self) -> None:
        """A workflow with numeric comparison should pass."""
        spec = _create_workflow_spec(
            trigger_schema={
                "type": "object",
                "properties": {"score": {"type": "number"}}
            },
            condition="${test_trigger.score} >= 50",
        )
        # Should not raise
        _validate_workflow(spec)
