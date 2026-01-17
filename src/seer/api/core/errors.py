"""
Error handling middleware and utilities.

Provides RFC 7807 Problem Details for HTTP APIs error responses.
All API endpoints should use raise_problem() for consistent error handling.
"""
from __future__ import annotations

from typing import Optional, Sequence, Dict, Any

from typing import TYPE_CHECKING
from fastapi import HTTPException

# Import only for type checking to avoid runtime circular imports
if TYPE_CHECKING:  # pragma: no cover
    from seer.api.workflows import models as api_models

# Error type URIs
PROBLEM_BASE = "https://seer.errors"
VALIDATION_PROBLEM = f"{PROBLEM_BASE}/validation"
COMPILE_PROBLEM = f"{PROBLEM_BASE}/workflows/compile"
RUN_PROBLEM = f"{PROBLEM_BASE}/workflows/run"
AUTH_PROBLEM = f"{PROBLEM_BASE}/auth"
INTEGRATION_PROBLEM = f"{PROBLEM_BASE}/integrations"


def raise_problem(
    *,
    type_uri: str,
    title: str,
    detail: str,
    status: int,
    errors: Optional[Sequence["api_models.ProblemError"]] = None,
) -> None:
    """
    Raise an HTTP exception using RFC 7807 Problem Details format.

    Args:
        type_uri: URI identifying the problem type (e.g., VALIDATION_PROBLEM)
        title: Short, human-readable summary of the problem
        detail: Human-readable explanation specific to this occurrence
        status: HTTP status code
        errors: Optional list of field-level errors

    Raises:
        HTTPException with structured problem details in the detail payload

    Example:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404
        )
    """
    payload: Dict[str, Any] = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "errors": [error.model_dump() for error in errors] if errors else [],
    }
    raise HTTPException(status_code=status, detail=payload)
