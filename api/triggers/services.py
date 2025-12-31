from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)


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
) -> Dict[str, Any]:
    occurred_at = _utcnow()
    return {
        "id": f"evt_{uuid4().hex}",
        "trigger_key": trigger_key,
        "provider": provider,
        "account_id": provider_connection_id,
        "occurred_at": occurred_at.isoformat(),
        "received_at": occurred_at.isoformat(),
        "data": payload,
        "raw": raw,
    }


async def _persist_event(
    *,
    subscription: TriggerSubscription,
    envelope: Dict[str, Any],
    provider_event_id: Optional[str],
    raw: Optional[Dict[str, Any]],
) -> TriggerEvent:
    try:
        return await TriggerEvent.create(
            trigger_key=subscription.trigger_key,
            provider_connection_id=subscription.provider_connection_id,
            provider_event_id=provider_event_id,
            occurred_at=_utcnow(),
            event=envelope,
            raw_payload=raw,
            status=TriggerEventStatus.RECEIVED,
        )
    except IntegrityError:
        if provider_event_id:
            existing = await TriggerEvent.get(
                trigger_key=subscription.trigger_key,
                provider_connection_id=subscription.provider_connection_id,
                provider_event_id=provider_event_id,
            )
            logger.info(
                "Deduped trigger event",
                extra={
                    "trigger_key": subscription.trigger_key,
                    "subscription_id": subscription.id,
                    "provider_event_id": provider_event_id,
                },
            )
            return existing
        raise


async def handle_generic_webhook(
    subscription_id: int,
    *,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    secret: Optional[str],
    provider_event_id: Optional[str],
) -> TriggerEvent:
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
    )
    event = await _persist_event(
        subscription=subscription,
        envelope=envelope,
        provider_event_id=provider_event_id,
        raw=raw_payload,
    )
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


@dataclass
class TriggerRunJob:
    subscription_id: int
    event_id: int


class TriggerRunDispatcher:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[TriggerRunJob] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._worker())

    async def enqueue(self, job: TriggerRunJob) -> None:
        self._ensure_worker()
        await self._queue.put(job)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process_job(job)
            except Exception:
                logger.exception(
                    "Trigger run job failed",
                    extra={"subscription_id": job.subscription_id, "event_id": job.event_id},
                )
            finally:
                self._queue.task_done()

    async def _process_job(self, job: TriggerRunJob) -> None:
        subscription = await TriggerSubscription.get(id=job.subscription_id)
        await subscription.fetch_related("workflow", "user")
        event = await TriggerEvent.get(id=job.event_id)

        if not subscription.enabled:
            await TriggerEvent.filter(id=event.id).update(
                status=TriggerEventStatus.PROCESSED,
                error={"detail": "Subscription disabled"},
            )
            return

        workflow = subscription.workflow
        user = subscription.user
        if not workflow or not user:
            await TriggerEvent.filter(id=event.id).update(
                status=TriggerEventStatus.FAILED,
                error={"detail": "Workflow or user missing for subscription"},
            )
            return

        envelope = event.event or {}
        if not _filters_match(subscription.filters, envelope):
            await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.PROCESSED)
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
            return

        validation_errors = workflow_services._validate_resolved_inputs(resolved_inputs, spec)
        if validation_errors:
            await TriggerEvent.filter(id=event.id).update(
                status=TriggerEventStatus.FAILED,
                error={"detail": "Invalid inputs", "errors": validation_errors},
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
            logger.exception(
                "Triggered workflow run failed",
                extra={"run_id": run.id, "subscription_id": subscription.id, "event_id": event.id},
            )
            await TriggerEvent.filter(id=event.id).update(
                status=TriggerEventStatus.FAILED,
                error={"detail": getattr(exc, "detail", str(exc))},
            )


trigger_run_dispatcher = TriggerRunDispatcher()


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
    await trigger_run_dispatcher.enqueue(
        TriggerRunJob(subscription_id=subscription.id, event_id=event.id)
    )
    await TriggerEvent.filter(id=event.id).update(status=TriggerEventStatus.ROUTED)

