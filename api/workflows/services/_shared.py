"""Shared utilities and constants for workflow services."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from fastapi import HTTPException

from api.workflows import models as api_models
from shared.database.models import WorkflowRun
from workflow_compiler.schema.models import WorkflowSpec


# Problem type URIs
PROBLEM_BASE = "https://seer.errors/workflows"
VALIDATION_PROBLEM = f"{PROBLEM_BASE}/validation"
COMPILE_PROBLEM = f"{PROBLEM_BASE}/compile"
RUN_PROBLEM = f"{PROBLEM_BASE}/run"


def _now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    """Convert WorkflowSpec to dictionary."""
    return spec.model_dump(mode="json")


def _raise_problem(
    *,
    type_uri: str,
    title: str,
    detail: str,
    status: int,
    errors: Optional[Sequence[api_models.ProblemError]] = None,
) -> None:
    """
    Raise an HTTPException with RFC 7807 problem details format.

    Args:
        type_uri: URI identifying the problem type
        title: Short summary of the problem
        detail: Human-readable explanation
        status: HTTP status code
        errors: Optional list of validation errors
    """
    payload = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "errors": [error.model_dump() for error in errors] if errors else [],
    }
    raise HTTPException(status_code=status, detail=payload)


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """
    Generate SHA256 hash of workflow spec for versioning.

    Args:
        spec_dict: Workflow specification dictionary

    Returns:
        Hex-encoded SHA256 hash
    """
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_run_config(run: WorkflowRun, config_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build LangGraph run configuration with thread_id for checkpoint recovery.

    IMPORTANT: Always uses run.run_id as thread_id to ensure checkpoint retrieval works.
    If config_payload contains a different thread_id, it will be overridden.

    Args:
        run: Workflow run instance
        config_payload: Optional configuration overrides

    Returns:
        Configuration dictionary with thread_id set
    """
    base_config = dict((config_payload or {}) or {})
    configurable = dict((base_config.get("configurable") or {}) or {})
    # Always use run.run_id as thread_id for checkpoint retrieval consistency
    # Don't use setdefault - explicitly set to ensure it matches execution config
    configurable["thread_id"] = run.thread_id or run.run_id
    base_config["configurable"] = configurable
    return base_config
