"""
Shared exception hierarchy for the workflow compiler.
"""


class WorkflowCompilerError(Exception):
    """Base class for all compiler related errors."""


class ValidationPhaseError(WorkflowCompilerError):
    """Raised when the workflow specification fails structural checks."""


class TypeEnvironmentError(WorkflowCompilerError):
    """Raised when the type environment cannot be constructed."""


class LoweringError(WorkflowCompilerError):
    """Raised when converting the workflow into executable form fails."""


class ExecutionError(WorkflowCompilerError):
    """Raised for runtime execution issues (tool failures, invalid outputs, etc)."""

    def __init__(self, message: str, trace_data: dict | None = None):
        super().__init__(message)
        self.trace_data = trace_data
