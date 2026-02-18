"""
Shared exception hierarchy for the workflow compiler.
"""

from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCode:
    """Standard error codes for workflow compilation errors."""

    UNDEFINED_REFERENCE = "UNDEFINED_REFERENCE"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    INVALID_PROPERTY = "INVALID_PROPERTY"
    NON_ARRAY_ITEMS = "NON_ARRAY_ITEMS"
    ORPHANED_TRIGGER = "ORPHANED_TRIGGER"
    UNKNOWN_NODE_TYPE = "UNKNOWN_NODE_TYPE"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INVALID_EXPRESSION = "INVALID_EXPRESSION"
    TRIGGER_REFERENCE_ERROR = "TRIGGER_REFERENCE_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    MCP_ERROR = "MCP_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# =============================================================================
# Structured Error Data
# =============================================================================


@dataclass
class NodeError:
    """Structured error associated with a specific node or location.

    This allows the frontend to highlight specific nodes in the workflow editor
    when compilation or validation errors occur.
    """

    code: str
    """Error code, e.g., 'UNDEFINED_REFERENCE', 'TYPE_MISMATCH'."""

    message: str
    """Human-readable error message."""

    node_id: Optional[str] = None
    """ID of the node where the error occurred, if applicable."""

    location: Optional[str] = None
    """Location within the node, e.g., 'inputs.prompt', 'condition'."""

    expression: Optional[str] = None
    """The problematic expression, if applicable."""


# =============================================================================
# Exception Hierarchy
# =============================================================================


class WorkflowCompilerError(Exception):
    """Base class for all compiler related errors."""

    def __init__(self, message: str, errors: Optional[List[NodeError]] = None):
        super().__init__(message)
        self.errors: List[NodeError] = errors or []


class ValidationPhaseError(WorkflowCompilerError):
    """Raised when the workflow specification fails structural checks."""


class TypeEnvironmentError(WorkflowCompilerError):
    """Raised when the type environment cannot be constructed."""


class LoweringError(WorkflowCompilerError):
    """Raised when converting the workflow into executable form fails."""


class ExecutionError(WorkflowCompilerError):
    """Raised for runtime execution issues (tool failures, invalid outputs, etc)."""

    def __init__(self, message: str, trace_data: dict | None = None, errors: Optional[List[NodeError]] = None):
        super().__init__(message, errors=errors)
        self.trace_data = trace_data
