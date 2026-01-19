"""Workflow run execution (synchronous and asynchronous)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    _ensure_draft_version,
    _get_workflow,
    _now,
    _raise_problem,
    _spec_to_dict,
)
from seer.api.core.errors import  RUN_PROBLEM
from seer.analytics import analytics
from seer.config import config as shared_config
from seer.database import (
    TriggerSubscription,
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
from seer.core.schema.models import WorkflowSpec
import asyncio

logger = logging.getLogger(__name__)
from seer.worker.tasks.workflows import workflow_execution_task



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
    subscription: "TriggerSubscription",
) -> Optional[Dict[str, Any]]:
    """
    Generate a sample event envelope for a trigger subscription.
    Returns None if sample event is unavailable.

    Reuses existing logic from test_trigger_subscription.
    """
    from seer.core.registry.trigger_registry import trigger_registry
    from seer.core.triggers.events import build_event_envelope

    # Load trigger definition from registry
    definition = trigger_registry.maybe_get(subscription.trigger_key)
    if definition is None:
        logger.warning(
            "Cannot generate sample event: unknown trigger_key",
            extra={
                "subscription_id": subscription.id,
                "trigger_key": subscription.trigger_key,
            }
        )
        return None

    # Get sample event from trigger metadata
    sample_event = definition.meta.sample_event
    if sample_event is None:
        logger.warning(
            "Cannot generate sample event: no sample_event in trigger definition",
            extra={
                "subscription_id": subscription.id,
                "trigger_key": subscription.trigger_key,
            }
        )
        return None

    # Build event envelope (reuse existing helper)
    envelope = build_event_envelope(
        trigger_id=subscription.trigger_id,
        trigger_key=subscription.trigger_key,
        title=subscription.title or subscription.trigger_id,
        provider=definition.provider,
        provider_connection_id=subscription.provider_connection_id,
        payload=sample_event.get("data", sample_event),  # Handle both wrapped and unwrapped formats
        raw=sample_event.get("raw"),
        occurred_at=None,  # Uses current time
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
        # NEW: Check for triggers BEFORE calling _ensure_draft_version
        # to determine if we should skip validation (since we'll use sample events)
        draft = await WorkflowDraft.get(workflow=workflow)
        draft_spec = WorkflowSpec.model_validate(draft.spec or {})
        has_triggers = bool(draft_spec.triggers)

        # Pass skip_validation=True when triggers exist (we'll use sample events)
        version = await _ensure_draft_version(workflow, user, skip_validation=has_triggers)

    spec = WorkflowSpec.model_validate(version.spec)

    # NEW: Query enabled trigger subscriptions
    subscriptions = await TriggerSubscription.filter(
        workflow=workflow,
        enabled=True
    ).all()

    # NEW: If triggers exist, create multiple runs (one per trigger)
    if subscriptions:
        runs = []
        for subscription in subscriptions:
            # Generate sample trigger envelope
            trigger_envelope = await _generate_sample_trigger_envelope(subscription)
            if trigger_envelope is None:
                logger.warning(
                    "Skipping trigger subscription without sample event",
                    extra={
                        "subscription_id": subscription.id,
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

                analytics.capture(
                    distinct_id=user.user_id,
                    event="workflow_run_started",
                    properties={
                        "run_id": run.run_id,
                        "workflow_id": workflow.id,
                        "workflow_name": workflow.name,
                        "execution_mode": "api_async_with_sample_trigger",
                        "trigger_key": subscription.trigger_key,
                        "trigger_title": subscription.title,
                        "deployment_mode": shared_config.seer_mode,
                    },
                )
                runs.append({
                    "run": run,
                    "trigger_title": subscription.title or subscription.trigger_id,
                })
            except Exception as exc:
                logger.exception(
                    "Failed to enqueue trigger-based run",
                    extra={
                        "workflow_id": workflow_id,
                        "run_id": run.run_id,
                        "trigger_id": subscription.trigger_id,
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
                title="No valid trigger subscriptions",
                detail="Workflow has trigger subscriptions but none have valid sample events",
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
    except Exception as exc:
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
