"""Trigger subscription management and event binding validation."""

from __future__ import annotations

import re
import secrets
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from jsonschema import Draft7Validator

from seer.core.triggers.supabase_webhook import (
    SupabaseWebhookError,
    create_database_webhook,
    delete_database_webhook,
)
from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _get_workflow,
    _raise_problem,
)
from seer.config import config as shared_config
from seer.database import (
    User,
    TriggerSubscription,
    Workflow,
    WorkflowDraft,
    make_workflow_public_id,
)
from seer.logger import get_logger
from seer.observability import (
    PollingIntervalTooFast,
    get_limits_for_user,
    resolve_user_tier,
)
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.schema.models import (
    WorkflowSpec,
)

logger = get_logger(__name__)


def _load_trigger_definition(trigger_key: str):
    definition = trigger_registry.maybe_get(trigger_key)
    if definition is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger not found",
            detail=f"Trigger '{trigger_key}' is not registered",
            status=404,
        )
    return definition


def _validate_filters_payload(filters: Dict[str, Any], definition) -> None:
    if not filters:
        return
    schema = definition.schemas.filter
    if not schema:
        return
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(filters))
    if errors:
        detail = errors[0].message
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid trigger filters",
            detail=f"Filters did not match schema: {detail}",
            status=400,
        )


def _is_expression(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("${") and value.strip().endswith("}")


def _extract_event_path(expression: str) -> List[str]:
    content = expression.strip()[2:-1].strip()
    if not content.startswith("event."):
        raise ValueError("Bindings must reference event.*")
    segments = [segment for segment in content.split(".") if segment]
    if len(segments) < 2:
        raise ValueError("Binding must reference at least one event property")
    return segments[1:]


def _generate_subscription_secret() -> str:
    return secrets.token_urlsafe(24)


def _should_emit_webhook_url(trigger_key: str) -> bool:
    return trigger_key.startswith("webhook.")


def _build_webhook_url(subscription_id: int, trigger_key: str) -> Optional[str]:
    if trigger_key == "webhook.generic":
        return f"/v1/webhooks/generic/{subscription_id}"
    if trigger_key == "webhook.supabase.db_changes":
        return f"/v1/webhooks/generic/{subscription_id}"
    return None


def _serialize_subscription(
    subscription: TriggerSubscription,
) -> api_models.TriggerSubscriptionResponse:
    webhook_url = None
    if _should_emit_webhook_url(subscription.trigger_key):
        webhook_url = _build_webhook_url(subscription.id, subscription.trigger_key)
    return api_models.TriggerSubscriptionResponse(
        subscription_id=subscription.id,
        workflow_id=make_workflow_public_id(subscription.workflow_id),
        trigger_key=subscription.trigger_key,
        provider_connection_id=subscription.provider_connection_id,
        enabled=subscription.enabled,
        filters=dict(subscription.filters or {}),
        provider_config=dict(subscription.provider_config or {}),
        secret_token=subscription.secret_token,
        webhook_url=webhook_url,
        form_suffix=subscription.form_suffix,
        form_fields=subscription.form_fields,
        form_config=(
            dict(subscription.form_config or {}) if subscription.form_config else None
        ),
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


def _resolve_event_value(payload: Dict[str, Any], segments: List[str]) -> Any:
    current = payload
    for segment in segments:
        if not isinstance(current, dict):
            raise ValueError(f"Cannot traverse into '{segment}' on non-object value")
        if segment not in current:
            raise ValueError(f"Event payload is missing '{segment}'")
        current = current[segment]
    return current


def _evaluate_bindings(bindings: Dict[str, Any], event_payload: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in (bindings or {}).items():
        if _is_expression(value):
            path = _extract_event_path(value)
            resolved[key] = _resolve_event_value(event_payload, path)
        else:
            resolved[key] = value
    return resolved


def _validate_event_payload(event_payload: Dict[str, Any], schema: Dict[str, Any]) -> None:
    if not schema:
        return
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(event_payload))
    if errors:
        detail = errors[0].message
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid event payload",
            detail=f"Event payload failed validation: {detail}",
            status=400,
        )




def _validate_form_suffix(suffix: Optional[str]) -> None:
    if not suffix:
        return
    if not re.match(r"^[a-z0-9-]+$", suffix):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid form suffix",
            detail="Invalid form suffix format. Use lowercase letters, numbers, and hyphens only.",
            status=400,
        )
    reserved = {"workflows", "settings", "sign-in", "sign-up", "api", "admin"}
    if suffix in reserved:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Reserved form suffix",
            detail=f"Form suffix '{suffix}' is reserved and cannot be used.",
            status=400,
        )


async def _validate_and_adjust_poll_interval(
    user: User,
    requested_interval: Optional[int],
) -> tuple[int, Optional[str]]:
    """
    Validate polling interval against user's tier limits.

    Returns:
        Tuple of (adjusted_interval, warning_message)
        If requested interval is too fast, it will be clamped to the minimum allowed.
    """
    if requested_interval is None:
        return 60, None  # Default to 1 minute

    # Get user's tier limits
    limits = await get_limits_for_user(user)
    min_interval = limits.poll_min_interval_seconds

    # Check if requested interval is within limits
    if requested_interval < min_interval:
        # Clamp to minimum allowed interval
        tier = await resolve_user_tier(user)
        error = PollingIntervalTooFast(
            requested_interval=requested_interval,
            min_interval=min_interval,
            tier=tier,
        )
        # Return clamped value and warning message
        return min_interval, error.message

    return requested_interval, None


async def _create_supabase_webhook(
    subscription: TriggerSubscription, secret: str
) -> None:
    webhook_base_url = shared_config.webhook_base_url or "http://localhost:8000"
    base = webhook_base_url.rstrip("/")
    full_url = f"{base}/api/v1/webhooks/generic/{subscription.id}"
    try:
        metadata = await create_database_webhook(
            subscription, full_url, secret=secret
        )
        logger.info(
            "Created Supabase webhook",
            extra={"subscription_id": subscription.id, "metadata": metadata},
        )
    except SupabaseWebhookError as exc:
        logger.error(
            "Failed to create Supabase webhook, rolling back subscription",
            extra={"subscription_id": subscription.id, "error": str(exc)},
        )
        await subscription.delete()
        raise HTTPException(
            status_code=500, detail=f"Failed to create Supabase webhook: {str(exc)}"
        ) from exc


async def list_trigger_subscriptions(
    user: User,
    *,
    workflow_id: Optional[str] = None,
) -> api_models.TriggerSubscriptionListResponse:
    query = TriggerSubscription.filter(user=user)
    if workflow_id:
        workflow = await _get_workflow(user, workflow_id)
        query = query.filter(workflow=workflow)
    subscriptions = await query.order_by("-created_at")
    return api_models.TriggerSubscriptionListResponse(
        items=[_serialize_subscription(item) for item in subscriptions],
    )


async def create_trigger_subscription(
    user: User,
    payload: api_models.TriggerSubscriptionCreateRequest,
) -> api_models.TriggerSubscriptionResponse:
    workflow = await _get_workflow(user, payload.workflow_id)
    definition = _load_trigger_definition(payload.trigger_key)
    filters = dict(payload.filters or {})
    provider_config = dict(payload.provider_config or {})
    _validate_filters_payload(filters, definition)
    secret = None
    if _should_emit_webhook_url(payload.trigger_key):
        secret = _generate_subscription_secret()
    _validate_form_suffix(payload.form_suffix)

    # Phase 2: Polling Frequency Gate
    # Validate and adjust poll interval based on user's tier
    # For now, we validate the default value (60 seconds)
    # TODO: Add poll_interval_seconds to TriggerSubscriptionCreateRequest
    adjusted_interval, warning = await _validate_and_adjust_poll_interval(user, 60)
    if warning:
        logger.warning(
            "Poll interval adjusted for user %s: %s",
            user.id,
            warning,
            extra={"user_id": user.id, "adjusted_interval": adjusted_interval},
        )

    subscription = await TriggerSubscription.create(
        user=user,
        workflow=workflow,
        trigger_key=payload.trigger_key,
        provider_connection_id=payload.provider_connection_id,
        enabled=payload.enabled,
        filters=filters,
        provider_config=provider_config,
        secret_token=secret,
        form_suffix=payload.form_suffix,
        form_fields=payload.form_fields,
        form_config=payload.form_config,
        poll_interval_seconds=adjusted_interval,
    )
    if payload.trigger_key == "webhook.supabase.db_changes" and secret:
        await _create_supabase_webhook(subscription, secret)
    return _serialize_subscription(subscription)


async def get_trigger_subscription(
    user: User,
    subscription_id: int,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    return _serialize_subscription(subscription)


def _apply_subscription_updates(
    subscription: TriggerSubscription,
    payload: api_models.TriggerSubscriptionUpdateRequest,
    definition,
) -> None:
    if payload.filters is not None:
        new_filters = dict(payload.filters or {})
        _validate_filters_payload(new_filters, definition)
        subscription.filters = new_filters
    if payload.provider_connection_id is not None:
        subscription.provider_connection_id = payload.provider_connection_id
    if payload.provider_config is not None:
        subscription.provider_config = dict(payload.provider_config or {})
    if payload.enabled is not None:
        subscription.enabled = payload.enabled
    if _should_emit_webhook_url(subscription.trigger_key) and not subscription.secret_token:
        subscription.secret_token = _generate_subscription_secret()


async def update_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionUpdateRequest,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    definition = _load_trigger_definition(subscription.trigger_key)
    _apply_subscription_updates(subscription, payload, definition)
    await subscription.save()
    return _serialize_subscription(subscription)


async def delete_trigger_subscription(user: User, subscription_id: int) -> None:
    subscription = await _get_trigger_subscription(user, subscription_id)
    # For Supabase webhook triggers, clean up the database trigger
    if subscription.trigger_key == "webhook.supabase.db_changes":
        try:
            await delete_database_webhook(subscription)
        except Exception as exc:
            # Log but don't block deletion - trigger cleanup is best-effort
            logger.warning(
                "Failed to delete Supabase webhook (non-fatal) %s", str(exc),
            )
    await subscription.delete()


async def _get_trigger_subscription(user: User, subscription_id: int) -> TriggerSubscription:
    try:
        pk = int(subscription_id)
    except (TypeError, ValueError):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid subscription id",
            detail="Subscription id must be an integer",
            status=400,
        )
    subscription = (
        await TriggerSubscription.filter(id=pk, user=user)
        .prefetch_related("workflow")
        .first()
    )
    if subscription is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger subscription not found",
            detail=f"Subscription '{subscription_id}' not found",
            status=404,
        )
    return subscription


async def test_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionTestRequest,
) -> api_models.TriggerSubscriptionTestResponse:
    """
    Test a trigger subscription by validating event payload against trigger schema.

    With the new trigger model, workflows access trigger data directly via ${trigger.data.*}
    expressions. This endpoint now validates the event payload and returns the data
    that would be available to the workflow.
    """
    subscription = await _get_trigger_subscription(user, subscription_id)
    await subscription.fetch_related("workflow")
    definition = _load_trigger_definition(subscription.trigger_key)
    event_payload = payload.event or definition.meta.sample_event
    if event_payload is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Event payload required",
            detail="Provide an event payload or configure a trigger sample.",
            status=400,
        )
    _validate_event_payload(event_payload, definition.schemas.event)
    # Return the event data that would be available via ${trigger.data.*}
    return api_models.TriggerSubscriptionTestResponse(inputs=event_payload, errors=[])

from seer.core.registry.trigger_registry import POLLING_TRIGGERS
async def sync_trigger_subscriptions(
    user: User,
    workflow: Workflow,
    spec: WorkflowSpec,
    skip_validation: bool = False,
) -> None:
    """
    Reconcile TriggerSubscription rows to match the workflow spec.

    This runs during draft version creation to keep DB state in sync with the spec payload
    provided by the frontend.

    Args:
        skip_validation: If True, skip connection validation and webhook creation.
                        Used when running with sample events.
    """

    existing: Dict[str, TriggerSubscription] = {}
    duplicate_subscriptions: List[TriggerSubscription] = []
    for sub in await TriggerSubscription.filter(workflow=workflow):
        if sub.trigger_id in existing:
            duplicate_subscriptions.append(sub)
        else:
            existing[sub.trigger_id] = sub
    # Clean up any accidental duplicates before reconciling desired state.
    for duplicate in duplicate_subscriptions:
        await delete_trigger_subscription(user, duplicate.id)
    desired = {trigger.id: trigger for trigger in spec.triggers or []}

    # Remove subscriptions no longer declared in the spec.
    for trigger_id, subscription in existing.items():
        if trigger_id not in desired:
            await delete_trigger_subscription(user, subscription.id)

    # Upsert declared triggers.
    for trigger_spec in spec.triggers or []:
        definition = _load_trigger_definition(trigger_spec.key)
        if not skip_validation:
            if definition.meta.requires_connection and trigger_spec.provider_connection_id is None:
                _raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing trigger connection",
                    detail=f"Trigger '{trigger_spec.key}' requires a provider connection.",
                    status=400,
                )
        filters = dict(trigger_spec.filters or {})
        provider_config = dict(trigger_spec.provider_config or {})

        _validate_filters_payload(filters, definition)

        adjusted_interval, warning = await _validate_and_adjust_poll_interval(user, 60)
        if warning:
            logger.warning(
                "Poll interval adjusted for user %s: %s",
                user.id,
                warning,
                extra={"user_id": user.id, "adjusted_interval": adjusted_interval},
            )

        existing_subscription = existing.get(trigger_spec.id)
        previous_secret = getattr(existing_subscription, "secret_token", None)
        secret = previous_secret
        if _should_emit_webhook_url(trigger_spec.key) and not secret:
            secret = _generate_subscription_secret()

        if existing_subscription:
            existing_subscription.trigger_key = trigger_spec.key  # Update type reference
            existing_subscription.title = trigger_spec.title  # Update title for reference resolution
            existing_subscription.provider_connection_id = trigger_spec.provider_connection_id
            existing_subscription.enabled = trigger_spec.enabled
            existing_subscription.filters = filters
            existing_subscription.provider_config = provider_config
            existing_subscription.secret_token = secret
            existing_subscription.poll_interval_seconds = adjusted_interval
            await existing_subscription.save()

            # If we generated a new secret for Supabase, ensure webhook is created.
            if (
                not skip_validation
                and trigger_spec.key == "webhook.supabase.db_changes"
                and secret
                and not previous_secret
            ):
                await _create_supabase_webhook(existing_subscription, secret)
        else:
            is_polling = False
            if trigger_spec.key in POLLING_TRIGGERS:
                is_polling = True
            subscription = await TriggerSubscription.create(
                user=user,
                workflow=workflow,
                trigger_id=trigger_spec.id,
                trigger_key=trigger_spec.key,
                title=trigger_spec.title,
                provider_connection_id=trigger_spec.provider_connection_id,
                enabled=trigger_spec.enabled,
                filters=filters,
                provider_config=provider_config,
                secret_token=secret,
                poll_interval_seconds=adjusted_interval,
                is_polling=is_polling,
            )
            if not skip_validation and trigger_spec.key == "webhook.supabase.db_changes" and secret:
                await _create_supabase_webhook(subscription, secret)
