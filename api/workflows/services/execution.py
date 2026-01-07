"""Workflow run execution (synchronous and asynchronous)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from typing import Any, Dict, Optional

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.exceptions import HTTPException

from api.workflows import models as api_models
from api.workflows.services._shared import (
    COMPILE_PROBLEM,
    RUN_PROBLEM,
    _build_run_config,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from api.agents.checkpointer import get_checkpointer
from shared.analytics import analytics
from shared.config import config as shared_config
from shared.database.base import async_session_maker
from shared.database.models import (
    User,
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
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import WorkflowSpec

compiler = WorkflowCompilerSingleton.instance()
logger = logging.getLogger(__name__)


# ===== Helper Functions =====


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """Hash workflow spec for deduplication."""
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


async def _ensure_draft_version(workflow: Workflow, user: User, session: AsyncSession) -> WorkflowVersion:
    """
    Ensure a draft version exists for the workflow's current draft.

    Creates a new draft version if one doesn't exist for the current draft revision.
    """
    stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
    result = await session.execute(stmt)
    draft = result.scalar_one()

    spec_dict = json.loads(json.dumps(draft.spec or {}))
    spec_hash = _hash_spec(spec_dict)
    stmt = (
        select(WorkflowVersion)
        .where(
            WorkflowVersion.workflow_id == workflow.id,
            WorkflowVersion.spec_hash == spec_hash,
            WorkflowVersion.status == WorkflowVersionStatus.DRAFT,
            WorkflowVersion.created_from_draft_revision == draft.revision,
        )
        .order_by(WorkflowVersion.created_at.desc())
    )
    result = await session.execute(stmt)
    existing = result.scalars().first()
    if existing:
        return existing

    version = WorkflowVersion(
        workflow_id=workflow.id,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        created_from_draft_revision=draft.revision,
        created_by_id=user.id,
        manifest=None,
        spec_hash=spec_hash,
    )
    session.add(version)
    await session.commit()
    await session.refresh(version)
    return version


async def _create_run_record(
    user: User,
    *,
    workflow: Optional[Workflow],
    workflow_version: Optional[WorkflowVersion],
    spec: WorkflowSpec,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    source: WorkflowRunSource = WorkflowRunSource.MANUAL,
    session: AsyncSession,
) -> WorkflowRun:
    """Create a new workflow run record."""
    run = WorkflowRun(
        user_id=user.id,
        workflow_id=workflow.id if workflow else None,
        workflow_version_id=workflow_version.id if workflow_version else None,
        spec=_spec_to_dict(spec),
        inputs=inputs or {},
        config=config_payload or {},
        source=source,
        status=WorkflowRunStatus.QUEUED,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Update thread_id to match run_id
    run.thread_id = run.run_id
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


def _serialize_run(run: WorkflowRun) -> api_models.RunResponse:
    """Serialize workflow run to API response."""
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
    """Serialize workflow run to summary response."""
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


# ===== Phase 3 Refactored Helpers =====


async def _update_run_status_to_running(
    run: WorkflowRun,
    session: AsyncSession,
) -> None:
    """Update run status to RUNNING and set started_at timestamp."""
    run.status = WorkflowRunStatus.RUNNING
    run.started_at = _now()
    session.add(run)
    await session.commit()


def _track_workflow_execution_start(
    user: User,
    run: WorkflowRun,
    inputs: Dict[str, Any],
    execution_mode: str,
) -> float:
    """
    Track workflow execution start event.

    Returns:
        Start time (for duration calculation)
    """
    start_time = time.time()
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
    return start_time


async def _compile_workflow_for_run(
    user: User,
    run: WorkflowRun,
    checkpointer,
):
    """
    Compile workflow for execution.

    Returns:
        Compiled workflow object

    Raises:
        WorkflowCompilerError: If compilation fails
    """
    return await compiler.compile(
        user,
        run.spec,
        checkpointer=checkpointer,
    )


async def _invoke_workflow_with_config(
    compiled,
    run: WorkflowRun,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    Invoke compiled workflow with runtime config.

    Returns:
        Workflow execution result

    Raises:
        Exception: If execution fails
    """
    run_config = dict(config_payload or {})
    logger.debug(
        "Invoking compiled workflow for run '%s' with config_keys=%s",
        run.run_id,
        sorted(run_config.keys()),
    )
    effective_config = _build_run_config(run, run_config)
    logger.info(
        f"Executing workflow run '{run.run_id}' with config: {effective_config}",
        extra={"run_id": run.run_id, "config": effective_config}
    )
    return await compiled.ainvoke(inputs or {}, config=effective_config)


async def _handle_compilation_failure(
    exc: WorkflowCompilerError,
    run: WorkflowRun,
    user: User,
    start_time: float,
    execution_mode: str,
    session: AsyncSession,
) -> None:
    """
    Handle compilation failure with analytics tracking.

    Updates run status, tracks analytics, and raises HTTPException.
    """
    run.status = WorkflowRunStatus.FAILED
    run.finished_at = _now()
    run.error = str(exc)
    session.add(run)
    await session.commit()

    duration_ms = (time.time() - start_time) * 1000
    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_run_failed",
        properties={
            "run_id": run.run_id,
            "workflow_id": run.workflow.workflow_id if run.workflow else None,
            "workflow_name": run.workflow.name if run.workflow else "draft",
            "execution_mode": execution_mode,
            "duration_ms": round(duration_ms, 2),
            "error_type": "CompilationError",
            "error_message": str(exc)[:500],
            "deployment_mode": shared_config.seer_mode,
        },
    )

    _raise_problem(
        type_uri=COMPILE_PROBLEM,
        title="Compilation failed",
        detail=str(exc),
        status=400,
    )


async def _handle_execution_failure(
    exc: Exception,
    run: WorkflowRun,
    user: User,
    start_time: float,
    execution_mode: str,
    session: AsyncSession,
    is_compiler_error: bool = False,
) -> None:
    """
    Handle execution failure with analytics tracking.

    Consolidates error handling for WorkflowCompilerError and general exceptions.
    """
    print(f"{traceback.format_exc()}")

    run.status = WorkflowRunStatus.FAILED
    run.finished_at = _now()
    run.error = str(exc)
    session.add(run)
    await session.commit()

    duration_ms = (time.time() - start_time) * 1000
    error_type = "RuntimeError" if is_compiler_error else "Exception"
    problem_type = RUN_PROBLEM
    title = "Run failed"

    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_run_failed",
        properties={
            "run_id": run.run_id,
            "workflow_id": run.workflow.workflow_id if run.workflow else None,
            "workflow_name": run.workflow.name if run.workflow else "draft",
            "execution_mode": execution_mode,
            "duration_ms": round(duration_ms, 2),
            "error_type": error_type,
            "error_message": str(exc)[:500],
            "deployment_mode": shared_config.seer_mode,
        },
    )

    _raise_problem(
        type_uri=problem_type,
        title=title,
        detail=str(exc),
        status=400,
    )


# ===== Core Execution Functions =====


async def _execute_compiled_run(
    run: WorkflowRun,
    user: User,
    *,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Execute a compiled workflow run.

    Handles compilation, execution, and error tracking.
    """
    logger.debug(
        "Preparing workflow run '%s' (workflow_id=%s) inputs_keys=%s config_payload_keys=%s user_id=%s",
        run.run_id,
        getattr(run.workflow, "workflow_id", None),
        sorted((inputs or {}).keys()),
        sorted((config_payload or {}).keys()),
        getattr(user, "id", None),
    )

    # Update run status to RUNNING
    await _update_run_status_to_running(run, session)

    # Track execution start
    execution_mode = getattr(run, '_analytics_execution_mode', 'taskiq_worker')
    start_time = _track_workflow_execution_start(user, run, inputs, execution_mode)

    # Compile workflow
    checkpointer = await get_checkpointer()
    try:
        compiled = await _compile_workflow_for_run(user, run, checkpointer)
    except WorkflowCompilerError as exc:
        await _handle_compilation_failure(exc, run, user, start_time, execution_mode, session)

    # Execute workflow
    try:
        result = await _invoke_workflow_with_config(compiled, run, inputs, config_payload, logger)
    except WorkflowCompilerError as exc:
        await _handle_execution_failure(exc, run, user, start_time, execution_mode, session, is_compiler_error=True)
    except Exception as exc:
        await _handle_execution_failure(exc, run, user, start_time, execution_mode, session, is_compiler_error=False)

    # Store timing info for _complete_run
    run._analytics_start_time = start_time
    run._analytics_execution_mode = execution_mode

    return result


async def _complete_run(run: WorkflowRun, output: Dict[str, Any], session: AsyncSession) -> WorkflowRun:
    """Mark workflow run as succeeded and capture analytics."""
    run.status = WorkflowRunStatus.SUCCEEDED
    run.finished_at = _now()
    run.output = output
    session.add(run)
    await session.commit()
    await session.refresh(run)

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


# ===== Public API Functions =====


async def run_draft_workflow(user: User, payload: api_models.RunFromSpecRequest) -> api_models.RunResponse:
    """
    Execute a draft workflow specification synchronously.

    Creates a run record and executes immediately without queueing.
    """
    async with async_session_maker() as session:
        run = await _create_run_record(
            user,
            workflow=None,
            workflow_version=None,
            spec=payload.spec,
            inputs=payload.inputs,
            config_payload=payload.config,
            session=session,
        )
        # Mark execution mode for tracking
        run._analytics_execution_mode = "api_sync"

        output = await _execute_compiled_run(run, user, inputs=payload.inputs, config_payload=payload.config, session=session)
        run = await _complete_run(run, output, session)
        return _serialize_run(run)


async def list_workflow_runs(
    user: User,
    workflow_id: str,
    *,
    limit: int = 50,
) -> api_models.WorkflowRunListResponse:
    """List workflow runs for a specific workflow."""
    # Import here to avoid circular dependency
    from api.workflows.services.workflows import _get_workflow

    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        limit = max(1, min(limit, 100))
        stmt = (
            select(WorkflowRun)
            .where(WorkflowRun.user_id == user.id, WorkflowRun.workflow_id == workflow.id)
            .order_by(WorkflowRun.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        runs = result.scalars().all()
        return api_models.WorkflowRunListResponse(
            workflow_id=workflow.workflow_id,
            runs=[_serialize_run_summary(run) for run in runs],
        )


async def run_saved_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse:
    """
    Queue a saved workflow for asynchronous execution.

    Creates a run record and enqueues it for execution by the worker.
    """
    # Import here to avoid circular dependency
    from api.workflows.services.workflows import _get_workflow

    async with async_session_maker() as session:
        workflow = await _get_workflow(user, workflow_id, session)
        if payload.version is not None:
            stmt = select(WorkflowVersion).where(
                WorkflowVersion.workflow_id == workflow.id,
                WorkflowVersion.version_number == payload.version,
                WorkflowVersion.status == WorkflowVersionStatus.RELEASED,
            )
            result = await session.execute(stmt)
            version = result.scalars().first()
            if version is None:
                _raise_problem(
                    type_uri=RUN_PROBLEM,
                    title="Version not found",
                    detail=f"Version '{payload.version}' not found for workflow '{workflow_id}'",
                    status=404,
                )
        else:
            version = await _ensure_draft_version(workflow, user, session)

        spec = WorkflowSpec.model_validate(version.spec)
        run = await _create_run_record(
            user,
            workflow=workflow,
            workflow_version=version,
            spec=spec,
            inputs=payload.inputs,
            config_payload=payload.config,
            session=session,
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
            run.status = WorkflowRunStatus.FAILED
            run.finished_at = _now()
            run.error = {"detail": f"Failed to enqueue workflow run: {exc}"}
            session.add(run)
            await session.commit()
            await session.refresh(run)
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
    async with async_session_maker() as session:
        stmt = select(WorkflowRun).where(WorkflowRun.id == run_id).options(selectinload(WorkflowRun.workflow), selectinload(WorkflowRun.user))
        result = await session.execute(stmt)
        run = result.scalar_one()

        user = run.user
        if user is None or getattr(user, "id", None) != user_id:
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one()

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
                session=session,
            )
            await _complete_run(run, output, session)
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
