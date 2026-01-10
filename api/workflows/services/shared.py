from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from api.core.errors import VALIDATION_PROBLEM
from api.core.errors import raise_problem as _raise_problem
from shared.database.models import User
from shared.database.workflow_models import (
    Workflow,
    WorkflowRun,
    parse_workflow_public_id,
)
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import WorkflowSpec


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    return spec.model_dump(mode="json")


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


async def _compile_workflow(
    user: User,
    spec: Dict[str, Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    Compile a workflow spec using the global compiler instance.

    This is a shared helper to avoid duplicating the compile pattern across
    history.py and execution.py.
    """
    compiler = WorkflowCompilerSingleton.instance()
    return await compiler.compile(
        user,
        spec,
        checkpointer=checkpointer,
    )
