from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from fastapi import HTTPException, status
from sqlmodel import select
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError

from shared.database.models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
    WorkflowRun,
    WorkflowRunSource,
)
from shared.database.base import async_session_maker
from workflow_compiler.registry.trigger_registry import trigger_registry
from workflow_compiler.schema.models import WorkflowSpec

from api.workflows import services as workflow_services
from worker.tasks.triggers import process_trigger_event as process_trigger_event_task
from shared.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_active_subscription(subscription_id: int) -> TriggerSubscription:
    async with async_session_maker() as session:
        stmt = select(TriggerSubscription).where(TriggerSubscription.id == subscription_id)
        result = await session.execute(stmt)
        subscription = result.scalar_one_or_none()

        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )
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
    async with async_session_maker() as session:
        try:
            event = TriggerEvent(
                trigger_key=subscription.trigger_key,
                provider_connection_id=subscription.provider_connection_id,
                provider_event_id=provider_event_id,
                event_hash=event_hash,
                occurred_at=occurred_at,
                event=envelope,
                raw_payload=raw,
                status=TriggerEventStatus.RECEIVED,
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)
            return event, True
        except IntegrityError:
            await session.rollback()
            # Build query for deduplication
            stmt = select(TriggerEvent).where(
                TriggerEvent.trigger_key == subscription.trigger_key,
                TriggerEvent.provider_connection_id == subscription.provider_connection_id
            )
            dedupe_key = None
            if provider_event_id:
                stmt = stmt.where(TriggerEvent.provider_event_id == provider_event_id)
                dedupe_key = ("provider_event_id", provider_event_id)
            elif event_hash:
                stmt = stmt.where(TriggerEvent.event_hash == event_hash)
                dedupe_key = ("event_hash", event_hash)

            if dedupe_key:
                result = await session.execute(stmt)
                existing = result.scalar_one()
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
    async with async_session_maker() as session:
        # Get subscription with relationships
        stmt = (
            select(TriggerSubscription)
            .where(TriggerSubscription.id == subscription_id)
            .options(
                selectinload(TriggerSubscription.workflow).selectinload(
                    lambda w: w.published_version
                ),
                selectinload(TriggerSubscription.user)
            )
        )
        result = await session.execute(stmt)
        subscription = result.scalar_one()

        # Get event
        stmt = select(TriggerEvent).where(TriggerEvent.id == event_id)
        result = await session.execute(stmt)
        event = result.scalar_one()

        if not subscription.enabled:
            event.status = TriggerEventStatus.PROCESSED
            event.error = {"detail": "Subscription disabled"}
            session.add(event)
            await session.commit()
            logger.error(
                "Trigger run job processed (subscription disabled)",
            )
            return

        workflow = subscription.workflow
        user = subscription.user
        if not workflow or not user:
            event.status = TriggerEventStatus.FAILED
            event.error = {"detail": "Workflow or user missing for subscription"}
            session.add(event)
            await session.commit()
            logger.error(
                "Trigger run job processed (workflow or user missing)",
            )
            return

        envelope = event.event or {}
        if not _filters_match(subscription.filters, envelope):
            event.status = TriggerEventStatus.PROCESSED
            session.add(event)
            await session.commit()
            logger.error(
                "Trigger run job processed (event filtered out)",
            )
            return

        published_version = workflow.published_version
        if published_version is None:
            event.status = TriggerEventStatus.FAILED
            event.error = {"detail": "Workflow has no published version"}
            session.add(event)
            await session.commit()
            logger.error("Trigger run job processed (no published version)")
            return

        spec = WorkflowSpec.model_validate(published_version.spec)
        bindings = dict(subscription.bindings or {})
        try:
            resolved_inputs = workflow_services._evaluate_bindings(bindings, envelope)
        except ValueError as exc:
            event.status = TriggerEventStatus.FAILED
            event.error = {"detail": str(exc)}
            session.add(event)
            await session.commit()
            logger.error(
                f"Trigger run job processed (invalid bindings) with error: {exc}",
            )
            return

        validation_errors = workflow_services._validate_resolved_inputs(resolved_inputs, spec)
        if validation_errors:
            event.status = TriggerEventStatus.FAILED
            event.error = {"detail": "Invalid inputs", "errors": validation_errors}
            session.add(event)
            await session.commit()
            logger.info(
                "Trigger run job processed (invalid inputs)",
            )
            return

        run = await workflow_services._create_run_record(
            user,
            workflow=workflow,
            workflow_version=published_version,
            spec=spec,
            inputs=resolved_inputs,
            config_payload={},
            source=WorkflowRunSource.TRIGGER,
        )

        # Update run with subscription and event
        stmt = select(WorkflowRun).where(WorkflowRun.id == run.id)
        result = await session.execute(stmt)
        run = result.scalar_one()
        run.subscription_id = subscription.id
        run.trigger_event_id = event.id
        session.add(run)
        await session.commit()

        run.subscription = subscription
        run.trigger_event = event
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

            # Update event status
            stmt = select(TriggerEvent).where(TriggerEvent.id == event_id)
            result = await session.execute(stmt)
            event = result.scalar_one()
            event.status = TriggerEventStatus.PROCESSED
            session.add(event)
            await session.commit()
        except HTTPException as exc:
            logger.error(
                "Triggered workflow run failed",
                extra={"run_id": run.id, "subscription_id": subscription.id, "event_id": event.id},
            )
            stmt = select(TriggerEvent).where(TriggerEvent.id == event_id)
            result = await session.execute(stmt)
            event = result.scalar_one()
            event.status = TriggerEventStatus.FAILED
            event.error = {"detail": getattr(exc, "detail", str(exc))}
            session.add(event)
            await session.commit()


async def _dispatch_trigger_event(
    subscription: TriggerSubscription,
    event: TriggerEvent,
    envelope: Dict[str, Any],
) -> None:
    async with async_session_maker() as session:
        if not _filters_match(subscription.filters, envelope):
            stmt = select(TriggerEvent).where(TriggerEvent.id == event.id)
            result = await session.execute(stmt)
            event_db = result.scalar_one()
            event_db.status = TriggerEventStatus.PROCESSED
            session.add(event_db)
            await session.commit()
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
            stmt = select(TriggerEvent).where(TriggerEvent.id == event.id)
            result = await session.execute(stmt)
            event_db = result.scalar_one()
            event_db.status = TriggerEventStatus.FAILED
            event_db.error = {"detail": "Failed to enqueue trigger event"}
            session.add(event_db)
            await session.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue trigger event",
            )

        logger.info(
            "Trigger event enqueued",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )
        stmt = select(TriggerEvent).where(TriggerEvent.id == event.id)
        result = await session.execute(stmt)
        event_db = result.scalar_one()
        event_db.status = TriggerEventStatus.ROUTED
        session.add(event_db)
        await session.commit()


async def handle_form_submission(
    subscription_id: int,
    form_data: Dict[str, Any],
    headers: Mapping[str, str],
    provider_event_id: Optional[str] = None,
) -> TriggerEvent:
    """
    Handle form submission from public form page.

    Creates a trigger event with form data and routes it to the workflow.
    No authentication required - public endpoint.
    """
    subscription = await _get_active_subscription(subscription_id)

    # Verify this is a form trigger
    if subscription.trigger_key != "trigger.form":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This subscription is not a form trigger",
        )

    # Build event payload with form data
    event_id = provider_event_id or f"form_{uuid4().hex}"
    occurred_at = _utcnow()

    event_payload = {
        "id": event_id,
        "trigger_key": "trigger.form",
        "provider": "form",
        "account_id": None,
        "occurred_at": occurred_at.isoformat(),
        "received_at": _utcnow().isoformat(),
        "data": form_data,  # Form fields directly as data
        "raw": {"headers": dict(headers), "form_data": form_data},
    }

    async with async_session_maker() as session:
        # Create trigger event
        event = TriggerEvent(
            subscription_id=subscription.id,
            provider_event_id=event_id,
            occurred_at=occurred_at,
            payload=event_payload,
            status=TriggerEventStatus.PENDING,
        )
        session.add(event)

        try:
            await session.commit()
            await session.refresh(event)
        except IntegrityError:
            logger.warning(
                "Duplicate form submission event",
                extra={"provider_event_id": event_id, "subscription_id": subscription.id},
            )
            await session.rollback()
            stmt = select(TriggerEvent).where(
                TriggerEvent.subscription_id == subscription.id,
                TriggerEvent.provider_event_id == event_id,
            )
            result = await session.execute(stmt)
            return result.scalar_one()

        logger.info(
            "Form submission event created",
            extra={"event_id": event.id, "subscription_id": subscription.id},
        )

        # Route event to workflow
        try:
            await process_trigger_event_task.kiq(subscription_id=subscription.id, event_id=event.id)
        except Exception:
            logger.exception(
                "Failed to enqueue form submission event",
                extra={"event_id": event.id, "subscription_id": subscription.id},
            )
            stmt = select(TriggerEvent).where(TriggerEvent.id == event.id)
            result = await session.execute(stmt)
            event_db = result.scalar_one()
            event_db.status = TriggerEventStatus.FAILED
            event_db.error = {"detail": "Failed to enqueue form submission"}
            session.add(event_db)
            await session.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process form submission",
            )

        stmt = select(TriggerEvent).where(TriggerEvent.id == event.id)
        result = await session.execute(stmt)
        event_db = result.scalar_one()
        event_db.status = TriggerEventStatus.ROUTED
        session.add(event_db)
        await session.commit()

        return event


async def get_form_trigger_info(subscription_id: int) -> Dict[str, Any]:
    """
    Get information about a form trigger for rendering the public form.

    Returns form fields (from workflow inputs), workflow name, description, etc.
    No authentication required - public endpoint.
    """
    subscription = await _get_active_subscription(subscription_id)

    # Verify this is a form trigger
    if subscription.trigger_key != "trigger.form":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This subscription is not a form trigger",
        )

    async with async_session_maker() as session:
        # Load workflow and draft to get spec
        stmt = (
            select(TriggerSubscription)
            .where(TriggerSubscription.id == subscription_id)
            .options(selectinload(TriggerSubscription.workflow))
        )
        result = await session.execute(stmt)
        subscription_with_workflow = result.scalar_one()
        workflow = subscription_with_workflow.workflow

        # Get workflow spec from draft
        from shared.database.models import WorkflowDraft
        stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
        result = await session.execute(stmt)
        draft = result.scalar_one_or_none()

        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow draft not found",
            )

        spec = WorkflowSpec.model_validate(draft.spec)

        # Build form fields from workflow inputs
        form_fields = []
        for name, input_def in (spec.inputs or {}).items():
            form_fields.append({
                "name": name,
                "label": input_def.description or name.replace("_", " ").title(),
                "type": input_def.type.value,
                "required": input_def.required,
                "default": input_def.default,
            })

        return {
            "subscription_id": subscription_id,
            "workflow_name": workflow.name,
            "workflow_description": workflow.description,
            "form_fields": form_fields,
            "form_url": f"/v1/forms/{subscription_id}/submit",
        }
