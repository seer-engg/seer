from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from fastapi import HTTPException

from api.workflows import models as api_models
from shared.database import (
    WorkflowRun,
    Workflow,
    parse_workflow_public_id,
    User,
)
from workflow_compiler.schema.models import WorkflowSpec


PROBLEM_BASE = "https://seer.errors/workflows"
VALIDATION_PROBLEM = f"{PROBLEM_BASE}/validation"
COMPILE_PROBLEM = f"{PROBLEM_BASE}/compile"
RUN_PROBLEM = f"{PROBLEM_BASE}/run"



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    return spec.model_dump(mode="json")


def _raise_problem(
    *,
    type_uri: str,
    title: str,
    detail: str,
    status: int,
    errors: Optional[Sequence[api_models.ProblemError]] = None,
) -> None:
    payload = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "errors": [error.model_dump() for error in errors] if errors else [],
    }
    raise HTTPException(status_code=status, detail=payload)


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()



def _build_run_config(run: WorkflowRun, config_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure LangGraph defaults (thread_id) are present so checkpoints can be recovered.

    IMPORTANT: Always uses run.run_id as thread_id to ensure checkpoint retrieval works.
    If config_payload contains a different thread_id, it will be overridden.
    """
    base_config = dict((config_payload or {}) or {})
    configurable = dict((base_config.get("configurable") or {}) or {})
    # Always use run.run_id as thread_id for checkpoint retrieval consistency
    # Don't use setdefault - explicitly set to ensure it matches execution config
    configurable["thread_id"] = run.thread_id or run.run_id
    base_config["configurable"] = configurable
    return base_config



async def _get_workflow(user: User, workflow_id: str) -> Workflow:
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow id",
            detail="Workflow id is invalid",
            status=400,
        )
    workflow = await Workflow.filter(id=pk, user=user).prefetch_related("draft", "published_version").first()
    if workflow is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )
    return workflow
