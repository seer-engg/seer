from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from fastapi import HTTPException, status
from tortoise.exceptions import DoesNotExist, IntegrityError

from shared.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
)
from workflow_compiler.registry.trigger_registry import trigger_registry
from workflow_compiler.schema.models import WorkflowSpec

from api.workflows import services as workflow_services
from worker.tasks.triggers import process_trigger_event as process_trigger_event_task
from shared.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_active_subscription(subscription_id: int) -> TriggerSubscription:
    try:
        subscription = await TriggerSubscription.get(id=subscription_id)
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        ) from None
    if not subscription.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not active",
        )
    return subscription


def _verify_secret(subscription: TriggerSubscription, provided: Optional[str]) -> None:
    expected = subscription.secret_token
    if not expected:
        return
    if not provided or provided.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )


def _load_trigger_provider(trigger_key: str) -> str:
    definition = trigger_registry.maybe_get(trigger_key)
    if definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trigger '{trigger_key}' is not registered",
        )
    return definition.provider


def _build_event_envelope(
    *,
    trigger_key: str,
    provider: str,
    provider_connection_id: Optional[int],
    payload: Dict[str, Any],
    raw: Optional[Dict[str, Any]],
    occurred_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    occurred = occurred_at or _utcnow()
    return {
        "id": f"evt_{uuid4().hex}",
        "trigger_key": trigger_key,
        "provider": provider,
        "account_id": provider_connection_id,
        "occurred_at": occurred.isoformat(),
        "received_at": _utcnow().isoformat(),
        "data": payload,
        "raw": raw,
    }


async def _persist_event(
    *,
    subscription: TriggerSubscription,
    envelope: Dict[str, Any],
    provider_event_id: Optional[str],
    event_hash: Optional[str],
    raw: Optional[Dict[str, Any]],
) -> tuple[TriggerEvent, bool]:
    occurred_at_str = envelope.get("occurred_at")
    occurred_at = (
        datetime.fromisoformat(occurred_at_str)
        if isinstance(occurred_at_str, str)
        else _utcnow()
    )
    try:
        event = await TriggerEvent.create(
            trigger_key=subscription.trigger_key,
            provider_connection_id=subscription.provider_connection_id,
            provider_event_id=provider_event_id,
            event_hash=event_hash,
            occurred_at=occurred_at,
            event=envelope,
            raw_payload=raw,
            status=TriggerEventStatus.RECEIVED,
        )
        return event, True
    except IntegrityError:
        dedupe_filters = {
            "trigger_key": subscription.trigger_key,
            "provider_connection_id": subscription.provider_connection_id,
        }
        dedupe_key = None
        if provider_event_id:
            dedupe_filters["provider_event_id"] = provider_event_id
            dedupe_key = ("provider_event_id", provider_event_id)
        elif event_hash:
            dedupe_filters["event_hash"] = event_hash
            dedupe_key = ("event_hash", event_hash)
        if dedupe_key:
            existing = await TriggerEvent.get(**dedupe_filters)
            logger.info(
                "Deduped trigger event",
                extra={
                    "trigger_key": subscription.trigger_key,
                    "subscription_id": subscription.id,
                    dedupe_key[0]: dedupe_key[1],
                },
            )
            return existing, False
        raise


async def handle_generic_webhook(
    subscription_id: int,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    secret: Optional[str],
    provider_event_id: Optional[str],
) -> TriggerEvent:
    logger.info(
        "Handling generic webhook",
        extra={"subscription_id": subscription_id, "provider_event_id": provider_event_id},
    )
    subscription = await _get_active_subscription(subscription_id)
    _verify_secret(subscription, secret)
    provider = _load_trigger_provider(subscription.trigger_key)
    raw_payload = {
        "headers": dict(headers),
        "body": payload,
    }
    envelope = _build_event_envelope(
        trigger_key=subscription.trigger_key,
        provider=provider,
        provider_connection_id=subscription.provider_connection_id,
        payload=payload,
        raw=raw_payload,
        occurred_at=_utcnow(),
    )
    event, created = await _persist_event(
        subscription=subscription,
        envelope=envelope,
        provider_event_id=provider_event_id,
        event_hash=None,
        raw=raw_payload,
    )
    if created:
        await _dispatch_trigger_event(subscription, event, envelope)
    return event


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


async def process_trigger_run_job(subscription_id: int, event_id: int) -> None:
    """
    Execute a trigger event workflow run synchronously.

    Invoked by Taskiq worker tasks to convert stored trigger events into workflow runs.
    """
    subscription = await TriggerSubscription.get(id=subscription_id)
    await subscription.fetch_related("workflow", "user")
    event = await TriggerEvent.get(id=event_id)

    if not subscription.enabled:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.PROCESSED,
            error={"detail": "Subscription disabled"},
        )
        logger.error(
            "Trigger run job processed (subscription disabled)",
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
            "Trigger run job processed (workflow or user missing)",
        )
        return

    envelope = event.event or {}
    if not _filters_match(subscription.filters, envelope):
        await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
        logger.error(
            "Trigger run job processed (event filtered out)",
        )
        return

    spec = WorkflowSpec.model_validate(workflow.spec)
    bindings = dict(subscription.bindings or {})
    try:
        resolved_inputs = workflow_services._evaluate_bindings(bindings, envelope)
    except ValueError as exc:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": str(exc)},
        )
        logger.error(
            f"Trigger run job processed (invalid bindings) with error: {exc}",
        )
        return

    validation_errors = workflow_services._validate_resolved_inputs(resolved_inputs, spec)
    if validation_errors:
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Invalid inputs", "errors": validation_errors},
        )
        logger.info(
            "Trigger run job processed (invalid inputs)",
        )
        return

    run = await WorkflowRun.create(
        user=user,
        workflow=workflow,
        workflow_version=workflow.version,
        spec=workflow.spec,
        inputs=resolved_inputs,
        config={},
        status=WorkflowRunStatus.QUEUED,
        source=WorkflowRunSource.TRIGGER,
        subscription=subscription,
        trigger_event=event,
    )
    logger.info(
        "Trigger run job processed (run created)"
    )
    try:
        output = await workflow_services._execute_compiled_run(
            run,
            user,
            inputs=resolved_inputs,
            config_payload={},
        )
        await workflow_services._complete_run(run, output)
        await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
    except HTTPException as exc:
        logger.error(
            "Triggered workflow run failed",
            extra={"run_id": run.id, "subscription_id": subscription.id, "event_id": event.id},
        )
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": getattr(exc, "detail", str(exc))},
        )


async def _dispatch_trigger_event(
    subscription: TriggerSubscription,
    event: TriggerEvent,
    envelope: Dict[str, Any],
) -> None:
    if not _filters_match(subscription.filters, envelope):
        await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
        logger.debug(
            "Trigger event filtered out",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )
        return
    logger.info(
        "Trigger event matched filters",
        extra={"event_id": event.id, "subscription_id": subscription.id},
    )
    try:
        await process_trigger_event_task.kiq(subscription_id=subscription.id, event_id=event.id)
    except Exception:
        logger.exception(
            "Failed to enqueue trigger event",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )
        await TriggerEvent.filter(id=event.id).update(
            status=TriggerEventStatus.FAILED,
            error={"detail": "Failed to enqueue trigger event"},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue trigger event",
        )
    logger.info(
        "Trigger event enqueued",
        extra={"event_id": event.id, "subscription_id": subscription.id},
    )
    await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.ROUTED)

