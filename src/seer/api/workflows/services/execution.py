"""Workflow run execution (synchronous and asynchronous)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from seer.api.core.errors import  RUN_PROBLEM
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
        sample_event = trigger.meta.sample_event

        # Get provider from registry
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


async def run_saved_workflow(  # pylint: disable=too-complex # Reason: Complex workflow routing logic for triggers vs manual runs
    user: User,
    workflow_id: str,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse | api_models.MultiRunResponse:
    """
    Run a workflow. If the workflow has enabled trigger subscriptions,
    automatically creates one run per trigger with sample event data.
    Otherwise, creates a single manual run.
    """
    # Run limit check moved to UsageLimitMiddleware
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
        # Always run latest DRAFT (create from published if needed)
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
        has_triggers = bool(spec.triggers)
        if has_triggers:
            # pylint: disable=import-outside-toplevel
            from seer.api.workflows.services.triggers import sync_trigger_subscriptions
            await sync_trigger_subscriptions(user, workflow, spec, skip_validation=True)

    spec = WorkflowSpec.model_validate(version.spec)

    # Read triggers directly from WorkflowSpec
    trigger_specs = spec.triggers if spec.triggers else []

    # If triggers exist, create multiple runs (one per trigger)
    if trigger_specs:
        runs = []
        for trigger_spec in trigger_specs:
            # Generate sample trigger envelope
            trigger_envelope = await _generate_sample_trigger_envelope(trigger_spec)
            if trigger_envelope is None:
                logger.warning(
                    "Skipping trigger without sample event",
                    extra={
                        "trigger_id": trigger_spec.id,
                        "workflow_id": workflow_id,
                    }
                )
                continue

            # Create run record
            run = await _create_run_record(
                user,
                workflow=workflow,
                workflow_version=version,
                spec=spec,
                inputs=payload.inputs,
                config_payload=payload.config,
                source=WorkflowRunSource.MANUAL,  # Still manual, but with trigger data
            )

            # Enqueue with trigger envelope
            try:
                await workflow_execution_task.kiq(
                    run_id=run.id,
                    user_id=user.id,
                    trigger_envelope=trigger_envelope
                )

                # Get trigger title from ui_meta or fallback to key
                trigger_title = trigger_spec.ui_meta.get("title", trigger_spec.key)
                runs.append({
                    "run": run,
                    "trigger_title": trigger_title,
                })
            except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Catch all task enqueue failures to continue processing other triggers
                logger.exception(
                    "Failed to enqueue trigger-based run",
                    extra={
                        "workflow_id": workflow_id,
                        "run_id": run.run_id,
                        "trigger_id": trigger_spec.id,
                    }
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
                api_models.RunWithTrigger(
                    **_serialize_run(r["run"]).model_dump(),
                    trigger_title=r["trigger_title"],
                )
                for r in runs
            ]
        )

    # EXISTING: No triggers, create single manual run
    run = await _create_run_record(
        user,
        workflow=workflow,
        workflow_version=version,
        spec=spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )

    # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with worker.tasks.workflows

    try:
        await workflow_execution_task.kiq(run_id=run.id, user_id=user.id)
    except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError) as exc:
        logger.exception(
            "Failed to enqueue workflow task",
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
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Catch all Taskiq broker failures to gracefully handle enqueue errors
        # Unexpected error - taskiq or broker issue
        logger.exception(
            "UNEXPECTED: Task enqueue failed",
            extra={"workflow_id": workflow_id, "run_id": run.run_id},
        )
        # Still update run to failed state
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
