"""Workflow run execution (synchronous and asynchronous)."""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Dict, Optional

from starlette.exceptions import HTTPException

from api.agents.checkpointer import get_checkpointer
from api.workflows import models as api_models
from api.workflows.services.shared import (
    COMPILE_PROBLEM,
    RUN_PROBLEM,
    _build_run_config,
    _compile_workflow,
    _get_workflow,
    _hash_spec,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from shared.analytics import analytics
from shared.config import config as shared_config
from shared.database.models import User
from shared.database.workflow_models import (
    Workflow,
    WorkflowDraft,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
    make_workflow_public_id,
)
from worker.tasks.workflows import execute_saved_workflow as execute_saved_workflow_task
from workflow_compiler.errors import WorkflowCompilerError
from workflow_compiler.schema.models import WorkflowSpec

logger = logging.getLogger(__name__)


async def _ensure_draft_version(workflow: Workflow, user: User) -> WorkflowVersion:
    draft = await WorkflowDraft.get(workflow=workflow)
    spec_dict = json.loads(json.dumps(draft.spec or {}))
    spec_hash = _hash_spec(spec_dict)
    existing = (
        await WorkflowVersion.filter(
            workflow=workflow,
            spec_hash=spec_hash,
            status=WorkflowVersionStatus.DRAFT,
            created_from_draft_revision=draft.revision,
        )
        .order_by("-created_at")
        .first()
    )
    if existing:
        return existing
    return await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_from_draft_revision=draft.revision,
        created_by=user,
        manifest=None,
        spec_hash=spec_hash,
    )


async def _create_run_record(
    user: User,
    *,
    workflow: Optional[Workflow],
    workflow_version: Optional[WorkflowVersion],
    spec: WorkflowSpec,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    source: WorkflowRunSource = WorkflowRunSource.MANUAL,
) -> WorkflowRun:
    run = await WorkflowRun.create(
        user=user,
        workflow=workflow,
        workflow_version=workflow_version,
        spec=_spec_to_dict(spec),
        inputs=inputs or {},
        config=config_payload or {},
        source=source,
        status=WorkflowRunStatus.QUEUED,
    )
    await WorkflowRun.filter(id=run.id).update(thread_id=run.run_id)
    run.thread_id = run.run_id
    return run


def _serialize_run(run: WorkflowRun) -> api_models.RunResponse:
    workflow_public_id = (
        make_workflow_public_id(run.workflow_id) if run.workflow_id else None
    )
    return api_models.RunResponse(
        run_id=run.run_id,
        status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        workflow_id=workflow_public_id,
        workflow_version_id=run.workflow_version_id,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        progress=None,
        current_node_id=None,
        last_error=run.error,
    )


def _serialize_run_summary(run: WorkflowRun) -> api_models.WorkflowRunSummary:
    return api_models.WorkflowRunSummary(
        run_id=run.run_id,
        status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        workflow_version_id=run.workflow_version_id,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        inputs=dict(run.inputs or {}),
        output=run.output,
        error=run.error,
    )


def _capture_workflow_start(
    run: WorkflowRun, user: User, execution_mode: str, inputs: Dict[str, Any]
) -> None:
    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_run_started",
        properties={
            "run_id": run.run_id,
            "workflow_id": run.workflow.workflow_id if run.workflow else None,
            "workflow_name": run.workflow.name if run.workflow else "draft",
            "execution_mode": execution_mode,
            "has_inputs": bool(inputs),
            "input_keys": list(inputs.keys()) if inputs else [],
            "deployment_mode": shared_config.seer_mode,
        },
    )


async def _handle_run_failure(
    run: WorkflowRun,
    user: User,
    error: Exception,
    start_time: float,
    execution_mode: str,
) -> None:
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.FAILED,
        finished_at=_now(),
        error=str(error),
    )
    duration_ms = (time.time() - start_time) * 1000

    is_compiler_error = isinstance(error, WorkflowCompilerError)
    error_type = "CompilationError" if is_compiler_error else (
        "RuntimeError" if is_compiler_error else "Exception"
    )
    problem_uri = COMPILE_PROBLEM if is_compiler_error else RUN_PROBLEM
    problem_title = "Compilation failed" if is_compiler_error else "Run failed"

    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_run_failed",
        properties={
            "run_id": run.run_id,
            "workflow_id": (
                run.workflow.workflow_id if run.workflow else None
            ),
            "workflow_name": run.workflow.name if run.workflow else "draft",
            "execution_mode": execution_mode,
            "duration_ms": round(duration_ms, 2),
            "error_type": error_type,
            "error_message": str(error)[:500],
            "deployment_mode": shared_config.seer_mode,
        },
    )
    _raise_problem(
        type_uri=problem_uri,
        title=problem_title,
        detail=str(error),
        status=400,
    )


async def _execute_compiled_run(
    run: WorkflowRun,
    user: User,
    *,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    logger.debug(
        "Preparing workflow run '%s' (workflow_id=%s) inputs_keys=%s "
        "config_payload_keys=%s user_id=%s",
        run.run_id,
        getattr(run.workflow, "workflow_id", None),
        sorted((inputs or {}).keys()),
        sorted((config_payload or {}).keys()),
        getattr(user, "id", None),
    )
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.RUNNING,
        started_at=_now(),
    )
    start_time = time.time()
    execution_mode = getattr(run, '_analytics_execution_mode', 'taskiq_worker')
    _capture_workflow_start(run, user, execution_mode, inputs)

    checkpointer = await get_checkpointer()
    try:
        compiled = await _compile_workflow(user, run.spec, checkpointer=checkpointer)
    except WorkflowCompilerError as exc:
        await _handle_run_failure(run, user, exc, start_time, execution_mode)
    try:
        run_config = dict(config_payload or {})
        logger.debug(
            "Invoking compiled workflow for run '%s' with config_keys=%s "
            "user_context_id=%s",
            run.run_id,
            sorted(run_config.keys()),
            getattr(user, "id", None),
        )
        effective_config = _build_run_config(run, run_config)
        logger.info(
            "Executing workflow run '%s' with config: %s",
            run.run_id,
            effective_config,
            extra={"run_id": run.run_id, "config": effective_config},
        )
        result = await compiled.ainvoke(inputs or {}, config=effective_config)
    except WorkflowCompilerError as exc:
        print(f"{traceback.format_exc()}")
        await _handle_run_failure(run, user, exc, start_time, execution_mode)
    except Exception as exc:
        print(f"{traceback.format_exc()}")
        await _handle_run_failure(run, user, exc, start_time, execution_mode)

    run._analytics_start_time = start_time
    run._analytics_execution_mode = execution_mode
    return result


async def _complete_run(run: WorkflowRun, output: Dict[str, Any]) -> WorkflowRun:
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.SUCCEEDED,
        finished_at=_now(),
        output=output,
    )
    await run.refresh_from_db()

    # Capture workflow completion event
    if hasattr(run, '_analytics_start_time'):
        duration_ms = (time.time() - run._analytics_start_time) * 1000
        execution_mode = getattr(run, '_analytics_execution_mode', 'unknown')

        analytics.capture(
            distinct_id=run.user.user_id,
            event="workflow_run_completed",
            properties={
                "run_id": run.run_id,
                "workflow_id": run.workflow.workflow_id if run.workflow else None,
                "workflow_name": run.workflow.name if run.workflow else "draft",
                "execution_mode": execution_mode,
                "duration_ms": round(duration_ms, 2),
                "output_keys": list(output.keys()) if output else [],
                "deployment_mode": shared_config.seer_mode,
            },
        )

    return run


async def run_draft_workflow(
    user: User, payload: api_models.RunFromSpecRequest
) -> api_models.RunResponse:
    run = await _create_run_record(
        user,
        workflow=None,
        workflow_version=None,
        spec=payload.spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )
    # Mark execution mode for tracking
    run._analytics_execution_mode = "api_sync"

    output = await _execute_compiled_run(
        run, user, inputs=payload.inputs, config_payload=payload.config
    )
    run = await _complete_run(run, output)
    return _serialize_run(run)


async def list_workflow_runs(
    user: User,
    workflow_id: str,
    *,
    limit: int = 50,
) -> api_models.WorkflowRunListResponse:
    workflow = await _get_workflow(user, workflow_id)
    limit = max(1, min(limit, 100))
    runs = (
        await WorkflowRun.filter(user=user, workflow=workflow)
        .order_by("-created_at")
        .limit(limit)
    )
    return api_models.WorkflowRunListResponse(
        workflow_id=workflow.workflow_id,
        runs=[_serialize_run_summary(run) for run in runs],
    )


async def run_saved_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse:
    workflow = await _get_workflow(user, workflow_id)
    if payload.version is not None:
        version = await WorkflowVersion.filter(
            workflow=workflow,
            version_number=payload.version,
            status=WorkflowVersionStatus.RELEASED,
        ).first()
        if version is None:
            _raise_problem(
                type_uri=RUN_PROBLEM,
                title="Version not found",
                detail=f"Version '{payload.version}' not found for workflow '{workflow_id}'",
                status=404,
            )
    else:
        version = await _ensure_draft_version(workflow, user)

    spec = WorkflowSpec.model_validate(version.spec)
    run = await _create_run_record(
        user,
        workflow=workflow,
        workflow_version=version,
        spec=spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )
    try:
        await execute_saved_workflow_task.kiq(run_id=run.id, user_id=user.id)

        # Capture async workflow start event (actual execution tracked in worker)
        analytics.capture(
            distinct_id=user.user_id,
            event="workflow_run_started",
            properties={
                "run_id": run.run_id,
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "execution_mode": "api_async",
                "has_inputs": bool(payload.inputs),
                "input_keys": list((payload.inputs or {}).keys()),
                "deployment_mode": shared_config.seer_mode,
            },
        )
    except Exception as exc:
        logger.exception(
            "Failed to enqueue saved workflow run",
            extra={"workflow_id": workflow_id, "run_id": run.run_id},
        )
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error={"detail": f"Failed to enqueue workflow run: {exc}"},
        )
        await run.refresh_from_db()
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Failed to enqueue workflow run",
            detail="An error occurred while queuing the workflow execution.",
            status=500,
        )
    return _serialize_run(run)


async def execute_saved_workflow_run(*, run_id: int, user_id: int) -> None:
    """
    Execute a saved workflow run asynchronously (invoked by Taskiq worker).
    """
    run = await WorkflowRun.get(id=run_id)
    await run.fetch_related("workflow", "user")

    user = run.user
    if user is None or getattr(user, "id", None) != user_id:
        user = await User.get(id=user_id)

    inputs = dict(run.inputs or {})
    config_payload = dict(run.config or {})

    # Mark execution mode for tracking
    run._analytics_execution_mode = "taskiq_worker"

    try:
        output = await _execute_compiled_run(
            run,
            user,
            inputs=inputs,
            config_payload=config_payload,
        )
        await _complete_run(run, output)
    except HTTPException:
        logger.exception(
            "Saved workflow run failed",
            extra={"run_id": run.run_id, "workflow_id": getattr(run.workflow, "workflow_id", None)},
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error during saved workflow run",
            extra={"run_id": run.run_id, "workflow_id": getattr(run.workflow, "workflow_id", None)},
        )
        raise
