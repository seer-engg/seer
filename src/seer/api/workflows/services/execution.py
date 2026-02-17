"""Workflow run execution (synchronous and asynchronous)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.core.errors import COMPILE_PROBLEM, RUN_PROBLEM, raise_problem
from seer.core.errors import WorkflowCompilerError
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton
from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    _get_draft_version,
    _get_workflow,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from seer.core.schema.models import TriggerSpec, WorkflowSpec
from seer.database import (
    TriggerSubscription,
    User,
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
    make_workflow_public_id,
)
from seer.worker.tasks.workflows import workflow_execution_task
from seer.services.workflows.execution import (
    get_workflow_run_interrupt as _get_workflow_run_interrupt,
    resume_workflow_run as _resume_workflow_run,
)

logger = logging.getLogger(__name__)



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


async def _validate_workflow_spec(user: User, spec: WorkflowSpec) -> None:
    """
    Validate workflow spec by running the compiler pipeline.
    Raises HTTP 400 if validation fails.
    """
    compiler = WorkflowCompilerSingleton.instance()
    spec_dict = _spec_to_dict(spec)
    checkpointer = await get_checkpointer()
    try:
        await compiler.compile(user, spec_dict, checkpointer=checkpointer)
    except WorkflowCompilerError as exc:
        raise_problem(
            type_uri=COMPILE_PROBLEM,
            title="Workflow validation failed",
            detail=str(exc),
            status=400,
        )


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


def _validate_trigger_envelope(envelope: Dict[str, Any]) -> None:
    """
    Validate that a trigger envelope has required fields for execution.
    Raises HTTP 400 if validation fails.
    """
    required_fields = ["trigger_key", "data"]
    missing = [field for field in required_fields if field not in envelope]
    if missing:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Invalid trigger envelope",
            detail=f"Missing required fields: {', '.join(missing)}",
            status=400,
        )
    if not isinstance(envelope.get("data"), dict):
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Invalid trigger envelope",
            detail="'data' field must be an object",
            status=400,
        )


async def _generate_sample_trigger_envelope(
    trigger: Union["TriggerSubscription", TriggerSpec],
) -> Optional[Dict[str, Any]]:
    """
    Generate a sample event envelope for a trigger (subscription or spec).
    Returns None if sample event is unavailable.

    Supports both TriggerSubscription (DB model) and TriggerSpec (from workflow spec).
    """
    from seer.core.registry.trigger_registry import trigger_registry  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency
    from seer.core.triggers.events import (  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency
        TriggerEventEnvelopeInput,
        build_event_envelope,
    )

    # Extract fields based on type
    if isinstance(trigger, TriggerSpec):
        trigger_id = trigger.id
        trigger_key = trigger.key
        title = trigger.ui_meta.get("title", trigger.key)
        provider_connection_id = trigger.provider_config.get("provider_connection_id")

        # Get definition from registry FIRST (needed for fallback and provider)
        definition = trigger_registry.maybe_get(trigger_key)
        if definition is None:
            logger.warning(
                "Cannot generate sample event: unknown trigger_key",
                extra={
                    "trigger_id": trigger_id,
                    "trigger_key": trigger_key,
                }
            )
            return None

        provider = definition.provider
        # Fallback to registry's sample_event if spec doesn't have one
        sample_event = trigger.meta.sample_event or definition.meta.sample_event
    else:
        # TriggerSubscription
        trigger_id = trigger.trigger_id
        trigger_key = trigger.trigger_key
        title = trigger.title or trigger.trigger_id
        provider_connection_id = trigger.provider_connection_id

        # Load trigger definition from registry
        definition = trigger_registry.maybe_get(trigger_key)
        if definition is None:
            logger.warning(
                "Cannot generate sample event: unknown trigger_key",
                extra={
                    "subscription_id": trigger.id,
                    "trigger_key": trigger_key,
                }
            )
            return None

        provider = definition.provider
        sample_event = definition.meta.sample_event

    # Get sample event
    if sample_event is None:
        logger.warning(
            "Cannot generate sample event: no sample_event in trigger definition",
            extra={
                "trigger_id": trigger_id,
                "trigger_key": trigger_key,
            }
        )
        return None

    # Build event envelope (reuse existing helper)
    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=title,
            provider=provider,
            provider_connection_id=provider_connection_id,
            payload=sample_event.get("data", sample_event),  # Handle both wrapped and unwrapped formats
            raw=sample_event.get("raw"),
            occurred_at=None,  # Uses current time
        )
    )

    return envelope


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


async def _handle_trigger_event_override(
    user: User,
    *,
    workflow: Workflow,
    workflow_id: str,
    version: WorkflowVersion,
    spec: WorkflowSpec,
    payload: api_models.RunFromWorkflowRequest,
    trigger_specs: list[TriggerSpec],
) -> api_models.RunResponse:
    """Handle custom trigger_event_override for testing with real events."""
    _validate_trigger_envelope(payload.trigger_event_override)

    # Determine target trigger
    target_trigger = None
    if payload.trigger_id:
        target_trigger = next((t for t in trigger_specs if t.id == payload.trigger_id), None)
        if not target_trigger:
            _raise_problem(
                type_uri=RUN_PROBLEM,
                title="Trigger not found",
                detail=f"No trigger with id '{payload.trigger_id}' found in workflow",
                status=404,
            )
    elif len(trigger_specs) == 1:
        target_trigger = trigger_specs[0]
    elif len(trigger_specs) > 1:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Ambiguous trigger",
            detail="trigger_id required when workflow has multiple triggers",
            status=400,
        )

    # Build effective envelope with trigger_id for expression resolution
    effective_envelope = dict(payload.trigger_event_override)
    if target_trigger:
        effective_envelope["trigger_id"] = target_trigger.id
        if not effective_envelope.get("title"):
            effective_envelope["title"] = target_trigger.ui_meta.get("title", target_trigger.key)

    run = await _create_run_record(
        user,
        workflow=workflow,
        workflow_version=version,
        spec=spec,
        inputs=payload.inputs,
        config_payload=payload.config,
        source=WorkflowRunSource.MANUAL,
    )

    try:
        await workflow_execution_task.kiq(
            run_id=run.id, user_id=user.id, trigger_envelope=effective_envelope
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Catch all enqueue failures
        logger.exception(
            "Failed to enqueue run with trigger_event_override",
            extra={"workflow_id": workflow_id, "run_id": run.run_id}
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

    logger.info(
        "Created run with trigger_event_override",
        extra={
            "workflow_id": workflow_id,
            "run_id": run.run_id,
            "trigger_key": effective_envelope.get("trigger_key"),
            "trigger_id": effective_envelope.get("trigger_id"),
        }
    )
    return _serialize_run(run)


async def _handle_multi_trigger_runs(
    user: User,
    *,
    workflow: Workflow,
    workflow_id: str,
    version: WorkflowVersion,
    spec: WorkflowSpec,
    payload: api_models.RunFromWorkflowRequest,
    trigger_specs: list[TriggerSpec],
) -> api_models.MultiRunResponse:
    """Create multiple runs, one per trigger with sample event data."""
    runs = []
    for trigger_spec in trigger_specs:
        trigger_envelope = await _generate_sample_trigger_envelope(trigger_spec)
        if trigger_envelope is None:
            logger.warning(
                "Skipping trigger without sample event",
                extra={"trigger_id": trigger_spec.id, "workflow_id": workflow_id}
            )
            continue

        run = await _create_run_record(
            user,
            workflow=workflow,
            workflow_version=version,
            spec=spec,
            inputs=payload.inputs,
            config_payload=payload.config,
            source=WorkflowRunSource.MANUAL,
        )

        try:
            await workflow_execution_task.kiq(
                run_id=run.id, user_id=user.id, trigger_envelope=trigger_envelope
            )
            trigger_title = trigger_spec.ui_meta.get("title", trigger_spec.key)
            runs.append({"run": run, "trigger_title": trigger_title})
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Continue processing other triggers
            logger.exception(
                "Failed to enqueue trigger-based run",
                extra={"workflow_id": workflow_id, "run_id": run.run_id, "trigger_id": trigger_spec.id}
            )
            await WorkflowRun.filter(id=run.id).update(
                status=WorkflowRunStatus.FAILED,
                finished_at=_now(),
                error={"detail": f"Failed to enqueue workflow run: {exc}"},
            )

    if not runs:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="No valid triggers",
            detail="Workflow has triggers but none have valid sample events",
            status=400,
        )

    logger.info(
        "Created multiple runs for workflow with triggers",
        extra={
            "workflow_id": workflow_id,
            "run_count": len(runs),
            "trigger_titles": [r["trigger_title"] for r in runs],
        }
    )
    return api_models.MultiRunResponse(
        runs=[
            api_models.RunWithTrigger(**_serialize_run(r["run"]).model_dump(), trigger_title=r["trigger_title"])
            for r in runs
        ]
    )


async def _handle_manual_run(
    user: User,
    *,
    workflow: Workflow,
    workflow_id: str,
    version: WorkflowVersion,
    spec: WorkflowSpec,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse:
    """Create a single manual run without trigger data."""
    run = await _create_run_record(
        user,
        workflow=workflow,
        workflow_version=version,
        spec=spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )

    try:
        await workflow_execution_task.kiq(run_id=run.id, user_id=user.id)
    except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError) as exc:
        logger.exception(
            "Failed to enqueue workflow task",
            extra={"workflow_id": workflow_id, "run_id": run.run_id}
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
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Catch all Taskiq broker failures
        logger.exception(
            "UNEXPECTED: Task enqueue failed",
            extra={"workflow_id": workflow_id, "run_id": run.run_id}
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


async def run_saved_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse | api_models.MultiRunResponse:
    """
    Run a workflow. If the workflow has enabled trigger subscriptions,
    automatically creates one run per trigger with sample event data.
    Otherwise, creates a single manual run.
    """
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
        version = await _get_draft_version(workflow, create_if_missing=True, user=user)
        if not version:
            _raise_problem(
                type_uri=RUN_PROBLEM,
                title="No draft version",
                detail="Workflow has no draft version to run",
                status=500,
            )

        # Sync triggers for DRAFT
        spec = WorkflowSpec.model_validate(version.spec)
        if spec.triggers:
            # pylint: disable-next=import-outside-toplevel # Reason: Avoid circular import
            from seer.api.workflows.services.triggers import sync_trigger_subscriptions
            await sync_trigger_subscriptions(user, workflow, spec, skip_validation=True)

    spec = WorkflowSpec.model_validate(version.spec)
    await _validate_workflow_spec(user, spec)
    trigger_specs = spec.triggers or []

    if payload.trigger_event_override:
        return await _handle_trigger_event_override(
            user,
            workflow=workflow,
            workflow_id=workflow_id,
            version=version,
            spec=spec,
            payload=payload,
            trigger_specs=trigger_specs,
        )

    if trigger_specs:
        return await _handle_multi_trigger_runs(
            user,
            workflow=workflow,
            workflow_id=workflow_id,
            version=version,
            spec=spec,
            payload=payload,
            trigger_specs=trigger_specs,
        )

    return await _handle_manual_run(
        user,
        workflow=workflow,
        workflow_id=workflow_id,
        version=version,
        spec=spec,
        payload=payload,
    )


async def resume_workflow_run(
    user: User,
    run_id: str,
    responses: Dict[str, Any],
) -> api_models.RunResponse:
    """
    Resume a workflow run that is paused at an HITL interrupt.

    Delegates to the core service and returns API response.
    """
    await _resume_workflow_run(user, run_id, responses)

    # Fetch the updated run for response
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
    run_pk = parse_run_public_id(run_id)
    run = await WorkflowRun.get(id=run_pk)
    return _serialize_run(run)


async def get_workflow_run_interrupt(
    user: User,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get pending HITL interrupt data for a workflow run.

    Returns interrupt data if run is interrupted, None otherwise.
    """
    return await _get_workflow_run_interrupt(user, run_id)
