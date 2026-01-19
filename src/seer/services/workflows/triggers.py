from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.analytics.workflows import WorkflowAnalytics
from seer.api.workflows.services import _create_run_record
from seer.core.schema.models import WorkflowSpec
from seer.database import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
    WorkflowRun,
    WorkflowRunSource,
)
from seer.database.workflow_models import WorkflowRunStatus
from seer.logger import get_logger
from seer.services.workflows.execution import _execute_run, _now

logger = get_logger(__name__)


def _lookup_filter_value(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current

def _filters_match(filters: Optional[Dict[str, Any]], envelope: Dict[str, Any]) -> bool:
    if not filters:
        return True
    data = envelope.get("data") or {}
    if not isinstance(data, dict):
        return False
    for key, expected in filters.items():
        actual = _lookup_filter_value(data, key)
        if actual != expected:
            return False
    return True




async def process_trigger_event(subscription_id: int, event_id: int) -> None:
    """
    Execute a trigger event workflow run synchronously.

    Invoked by Taskiq worker tasks to convert stored trigger events into workflow runs.
    """
    subscription = await TriggerSubscription.get(id=subscription_id)
    await subscription.fetch_related("workflow", "workflow__published_version", "user")
    event = await TriggerEvent.get(id=event_id)

    logger.info(
        "Processing trigger job",
        extra={
            "subscription_id": subscription_id,
            "event_id": event_id,
            "trigger_key": subscription.trigger_key,
        }
    )

    if not subscription.enabled:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.PROCESSED,
            error={"detail": "Subscription disabled"},
        )
        logger.info(
            "Trigger job skipped: subscription disabled",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "trigger_key": subscription.trigger_key,
            }
        )
        return

    workflow = subscription.workflow
    user = subscription.user
    if not workflow or not user:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Workflow or user missing for subscription"},
        )
        logger.error(
            "Trigger job failed: workflow or user missing",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "trigger_key": subscription.trigger_key,
                "has_workflow": workflow is not None,
                "has_user": user is not None,
            }
        )
        return

    envelope = event.event or {}
    if not _filters_match(subscription.filters, envelope):
        await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
        logger.info(
            "Trigger job skipped: event filtered out",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "trigger_key": subscription.trigger_key,
                "filters": subscription.filters,
                "event_data": envelope.get("data"),
            }
        )
        return

    published_version = workflow.published_version
    if published_version is None:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Workflow has no published version"},
        )
        logger.error(
            "Trigger job failed: no published version",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "workflow_id": workflow.id,
                "trigger_key": subscription.trigger_key,
            }
        )
        return

    spec = WorkflowSpec.model_validate(published_version.spec)

    # Trigger data is now accessed directly via ${trigger.data.*} expressions in the workflow.
    # No binding evaluation or input validation is needed.
    run = await _create_run_record(
        user,
        workflow=workflow,
        workflow_version=published_version,
        spec=spec,
        inputs={},
        config_payload={},
        source=WorkflowRunSource.TRIGGER,
    )
    await WorkflowRun.filter(id=run.id).update(
        subscription=subscription,
        trigger_event=event,
    )
    run.subscription = subscription
    run.trigger_event = event
    logger.info(
        "Trigger job succeeded: workflow run created",
        extra={
            "subscription_id": subscription_id,
            "event_id": event_id,
            "workflow_id": workflow.id,
            "run_id": run.id,
        }
    )
    try:
        output, metrics = await _execute_run(
            run,
            user,
            inputs={},
            config_payload={},
            execution_mode="trigger",
            trigger_envelope=envelope,
        )
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            finished_at=_now(),
            output=output,
        )
        await run.refresh_from_db()
        await WorkflowAnalytics._complete_run(run, output, metrics)  # pylint: disable=protected-access  # use internal analytics hook until public API exists
        await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
        logger.info(
            "Trigger job completed: workflow execution succeeded",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "workflow_id": workflow.id,
                "run_id": run.id,
            }
        )
    except HTTPException as exc:
        logger.error(
            "Trigger job failed: workflow execution error",
            extra={
                "subscription_id": subscription_id,
                "event_id": event_id,
                "workflow_id": workflow.id,
                "run_id": run.id,
                "error_detail": getattr(exc, "detail", str(exc)),
            }
        )
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": getattr(exc, "detail", str(exc))},
        )
