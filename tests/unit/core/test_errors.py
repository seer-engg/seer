"""
Unit tests for workflow compiler error classes.

Tests the NodeError dataclass and WorkflowCompilerError with structured errors.
"""

import pytest

from seer.core.errors import (
    ErrorCode,
    NodeError,
    WorkflowCompilerError,
    ValidationPhaseError,
    TypeEnvironmentError,
    LoweringError,
    ExecutionError,
)

pytestmark = pytest.mark.unit


# =============================================================================
# NodeError Tests
# =============================================================================


def test_node_error_creation_minimal():
    """Test creating a NodeError with minimal required fields."""
    error = NodeError(
        code=ErrorCode.UNDEFINED_REFERENCE,
        message="Symbol 'foo' is not defined"
    )

    assert error.code == ErrorCode.UNDEFINED_REFERENCE
    assert error.message == "Symbol 'foo' is not defined"
    assert error.node_id is None
    assert error.location is None
    assert error.expression is None


def test_node_error_creation_full():
    """Test creating a NodeError with all fields."""
    error = NodeError(
        code=ErrorCode.TYPE_MISMATCH,
        message="Expected array, got string",
        node_id="loop_node",
        location="items",
        expression="${data.items}"
    )

    assert error.code == ErrorCode.TYPE_MISMATCH
    assert error.message == "Expected array, got string"
    assert error.node_id == "loop_node"
    assert error.location == "items"
    assert error.expression == "${data.items}"


def test_error_codes_defined():
    """Test that all expected error codes are defined."""
    assert hasattr(ErrorCode, "UNDEFINED_REFERENCE")
    assert hasattr(ErrorCode, "TYPE_MISMATCH")
    assert hasattr(ErrorCode, "INVALID_PROPERTY")
    assert hasattr(ErrorCode, "NON_ARRAY_ITEMS")
    assert hasattr(ErrorCode, "ORPHANED_TRIGGER")
    assert hasattr(ErrorCode, "UNKNOWN_NODE_TYPE")
    assert hasattr(ErrorCode, "TOOL_NOT_FOUND")
    assert hasattr(ErrorCode, "SCHEMA_MISMATCH")
    assert hasattr(ErrorCode, "INVALID_EXPRESSION")
    assert hasattr(ErrorCode, "TRIGGER_REFERENCE_ERROR")
    assert hasattr(ErrorCode, "FILE_NOT_FOUND")
    assert hasattr(ErrorCode, "MCP_ERROR")
    assert hasattr(ErrorCode, "VALIDATION_ERROR")


# =============================================================================
# WorkflowCompilerError Tests
# =============================================================================


def test_workflow_compiler_error_without_errors():
    """Test WorkflowCompilerError with just a message."""
    exc = WorkflowCompilerError("Something went wrong")

    assert str(exc) == "Something went wrong"
    assert exc.errors == []


def test_workflow_compiler_error_with_errors():
    """Test WorkflowCompilerError with structured errors."""
    node_errors = [
        NodeError(
            code=ErrorCode.UNDEFINED_REFERENCE,
            message="Symbol 'x' not defined",
            node_id="node1",
            location="inputs"
        ),
        NodeError(
            code=ErrorCode.TYPE_MISMATCH,
            message="Expected number",
            node_id="node2",
            location="condition"
        )
    ]

    exc = WorkflowCompilerError("Multiple validation errors", errors=node_errors)

    assert str(exc) == "Multiple validation errors"
    assert len(exc.errors) == 2
    assert exc.errors[0].node_id == "node1"
    assert exc.errors[1].node_id == "node2"


def test_validation_phase_error_inherits_errors():
    """Test ValidationPhaseError carries errors from base class."""
    node_errors = [
        NodeError(
            code=ErrorCode.ORPHANED_TRIGGER,
            message="Trigger has no edges",
            node_id="trigger1"
        )
    ]

    exc = ValidationPhaseError("Orphaned trigger detected", errors=node_errors)

    assert isinstance(exc, WorkflowCompilerError)
    assert len(exc.errors) == 1
    assert exc.errors[0].code == ErrorCode.ORPHANED_TRIGGER


def test_type_environment_error_inherits_errors():
    """Test TypeEnvironmentError carries errors from base class."""
    node_errors = [
        NodeError(
            code=ErrorCode.UNKNOWN_NODE_TYPE,
            message="Unknown type 'custom'",
            node_id="custom_node"
        )
    ]

    exc = TypeEnvironmentError("Unknown node type", errors=node_errors)

    assert isinstance(exc, WorkflowCompilerError)
    assert len(exc.errors) == 1
    assert exc.errors[0].code == ErrorCode.UNKNOWN_NODE_TYPE


def test_lowering_error_inherits_errors():
    """Test LoweringError carries errors from base class."""
    exc = LoweringError("Control flow lowering failed", errors=[])

    assert isinstance(exc, WorkflowCompilerError)
    assert exc.errors == []


def test_execution_error_with_trace_data():
    """Test ExecutionError with both trace_data and errors."""
    trace = {"node": "tool1", "status": "failed"}
    node_errors = [
        NodeError(
            code=ErrorCode.TOOL_NOT_FOUND,
            message="Tool 'unknown_tool' not found",
            node_id="tool_node"
        )
    ]

    exc = ExecutionError(
        "Tool execution failed",
        trace_data=trace,
        errors=node_errors
    )

    assert str(exc) == "Tool execution failed"
    assert exc.trace_data == trace
    assert len(exc.errors) == 1
    assert exc.errors[0].node_id == "tool_node"


def test_execution_error_without_errors():
    """Test ExecutionError with only trace_data (backwards compatible)."""
    trace = {"node": "llm1", "output": "invalid"}

    exc = ExecutionError("Invalid output", trace_data=trace)

    assert exc.trace_data == trace
    assert exc.errors == []


# =============================================================================
# Error Propagation Tests
# =============================================================================


def test_errors_can_be_modified_after_creation():
    """Test that errors list can be modified on existing exception."""
    exc = ValidationPhaseError("Initial error")
    assert exc.errors == []

    # Add errors after creation (as done in type_env.py)
    exc.errors = [
        NodeError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Added later",
            node_id="dynamic_node"
        )
    ]

    assert len(exc.errors) == 1
    assert exc.errors[0].node_id == "dynamic_node"


def test_raise_and_catch_with_errors():
    """Test raising and catching exceptions preserves errors."""
    node_errors = [
        NodeError(
            code=ErrorCode.UNDEFINED_REFERENCE,
            message="Cannot find 'data'",
            node_id="process_node",
            location="inputs.data",
            expression="${data.value}"
        )
    ]

    with pytest.raises(ValidationPhaseError) as exc_info:
        raise ValidationPhaseError("Reference validation failed", errors=node_errors)

    assert len(exc_info.value.errors) == 1
    error = exc_info.value.errors[0]
    assert error.code == ErrorCode.UNDEFINED_REFERENCE
    assert error.node_id == "process_node"
    assert error.location == "inputs.data"
    assert error.expression == "${data.value}"
