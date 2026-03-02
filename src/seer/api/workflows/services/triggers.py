"""Trigger subscription management and event binding validation."""
# pylint: disable=too-many-lines  # Reason: Comprehensive trigger management requires extensive validation logic

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from jsonschema import Draft7Validator

from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope

from seer.core.triggers.supabase_webhook import (
    SupabaseWebhookError,
    create_database_webhook,
    delete_database_webhook,
)
from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _get_draft_version,
    _get_workflow,
    _raise_problem,
)
from seer.config import config as shared_config
from seer.database import (
    User,
    TriggerEvent,
    TriggerSubscription,
    Workflow,
    make_workflow_public_id,
    OAuthConnection
)
from seer.logger import get_logger
from seer.observability import (
    PollingIntervalTooFast,
    get_limits_for_user,
    resolve_user_tier,
)
from seer.core.registry.trigger_registry import POLLING_TRIGGERS, trigger_registry
from seer.core.schema.models import (
    JsonSchema,
    WorkflowSpec,
)
from seer.services.integrations.auth.oauth import get_oauth_provider
from seer.services.integrations.auth.helpers import get_connection_display_name

logger = get_logger(__name__)


class MultipleAccountsError(Exception):
    """Raised when multiple OAuth accounts exist and explicit selection is required."""

    def __init__(self, provider: str, account_names: List[str]):
        self.provider = provider
        self.account_names = account_names
        super().__init__(
            f"Multiple {provider} accounts available. "
            f"Please specify provider_connection_id. Available: {', '.join(account_names)}"
        )


async def _auto_select_provider_connection(
    user: User,
    trigger_definition,
) -> Optional[int]:
    """
    Auto-select an active OAuth connection for a trigger's provider.

    Returns the connection ID if exactly one account exists.
    Raises MultipleAccountsError if multiple accounts exist (requires explicit selection).
    Returns None if no accounts exist.
    """
    # Map trigger provider to OAuth provider (e.g., "gmail" -> "google")
    oauth_provider = get_oauth_provider(trigger_definition.provider)

    # Query for all active connections
    connections = await OAuthConnection.filter(
        user=user,
        provider=oauth_provider,
        status="active"
    ).order_by("-created_at").all()

    if len(connections) == 0:
        return None

    if len(connections) == 1:
        connection = connections[0]
        logger.info(
            "Auto-selected provider connection for trigger",
            extra={
                "user_id": user.id,
                "trigger_key": trigger_definition.key,
                "connection_id": connection.id,
                "provider": oauth_provider,
            }
        )
        return connection.id

    # Multiple connections - require explicit selection
    account_names = [get_connection_display_name(c) for c in connections]
    raise MultipleAccountsError(oauth_provider, account_names)


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


def _validate_provider_config(provider_config: Dict[str, Any], definition) -> None:
    """Validate provider_config against trigger's config schema.

    Note: provider_connection_id is an infrastructure field (specifies which OAuth connection
    to use) and is excluded from schema validation since it's not a user-configurable trigger option.
    """
    if not provider_config:
        return
    schema = definition.schemas.config
    if not schema:
        return
    # Exclude infrastructure fields that aren't part of the trigger's user-facing config schema
    config_to_validate = {k: v for k, v in provider_config.items() if k != "provider_connection_id"}
    if not config_to_validate:
        return
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(config_to_validate))
    if errors:
        detail = errors[0].message
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid trigger configuration",
            detail=f"Provider config did not match schema: {detail}",
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


def _generate_webhook_slug() -> str:
    """Generate a URL-safe random slug for webhook URLs (256 bits entropy)."""
    return secrets.token_urlsafe(32)


def _generate_form_suffix() -> str:
    """Generate a short, URL-safe suffix for form URLs."""
    return secrets.token_urlsafe(6)


def _is_form_trigger(trigger_key: str) -> bool:
    return trigger_key == "form.hosted"


def _should_emit_webhook_url(trigger_key: str) -> bool:
    return trigger_key.startswith("webhook.")


def _build_webhook_url(webhook_slug: str, trigger_key: str) -> Optional[str]:
    """Build webhook URL path using the unique slug."""
    if trigger_key == "webhook.generic":
        return f"/api/v1/webhooks/generic/{webhook_slug}"
    if trigger_key == "webhook.supabase.db_changes":
        return f"/api/v1/webhooks/generic/{webhook_slug}"
    return None


def _build_form_url(subscription: TriggerSubscription) -> Optional[str]:
    """Build public form URL from frontend base URL and suffix."""
    if not subscription.form_suffix:
        return None

    frontend_base_url = shared_config.frontend_url or "http://localhost:5173"
    base = frontend_base_url.rstrip("/")
    return f"{base}/forms/{subscription.form_suffix}"


async def _serialize_subscription(
    subscription: TriggerSubscription,
) -> api_models.TriggerSubscriptionResponse:
    webhook_url = None
    form_url = None

    if _should_emit_webhook_url(subscription.trigger_key) and subscription.webhook_slug:
        webhook_url = _build_webhook_url(subscription.webhook_slug, subscription.trigger_key)

    # Build form URL for form triggers
    if subscription.trigger_key == "form.hosted":
        form_url = _build_form_url(subscription)

    # Fetch connection display name if connection is set
    connection_display_name = None
    if subscription.provider_connection_id:
        conn = await OAuthConnection.get_or_none(id=subscription.provider_connection_id)
        if conn:
            connection_display_name = get_connection_display_name(conn)

    return api_models.TriggerSubscriptionResponse(
        subscription_id=subscription.id,
        workflow_id=make_workflow_public_id(subscription.workflow_id),
        trigger_key=subscription.trigger_key,
        provider_connection_id=subscription.provider_connection_id,
        connection_display_name=connection_display_name,
        enabled=subscription.enabled,
        filters=dict(subscription.filters or {}),
        provider_config=dict(subscription.provider_config or {}),
        secret_token=None,  # Deprecated: slug-based URLs don't need secrets
        webhook_url=webhook_url,
        form_url=form_url,
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
    if not re.match(r"^[a-zA-Z0-9_-]+$", suffix):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid form suffix",
            detail="Invalid form suffix format. Use letters, numbers, hyphens, and underscores only.",
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


async def _create_supabase_webhook(subscription: TriggerSubscription) -> None:
    """Create a Supabase database webhook using the subscription's webhook_slug."""
    if not subscription.webhook_slug:
        raise ValueError("webhook_slug is required for Supabase webhooks")
    webhook_base_url = shared_config.webhook_base_url or "http://localhost:8000"
    base = webhook_base_url.rstrip("/")
    full_url = f"{base}/api/v1/webhooks/generic/{subscription.webhook_slug}"
    try:
        metadata = await create_database_webhook(subscription, full_url)
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
    serialized_items = [await _serialize_subscription(item) for item in subscriptions]
    return api_models.TriggerSubscriptionListResponse(
        items=serialized_items,
    )


async def list_trigger_subscriptions_extended(
    user: User,
    *,
    workflow_id: Optional[str] = None,
    trigger_key: Optional[str] = None,
    search: Optional[str] = None,
) -> api_models.TriggerSubscriptionListItemsResponse:
    """
    List trigger subscriptions with extended info for management view.

    Includes workflow title and last event timestamp.
    """
    query = TriggerSubscription.filter(user=user).prefetch_related("workflow")

    if workflow_id:
        workflow = await _get_workflow(user, workflow_id)
        query = query.filter(workflow=workflow)

    if trigger_key:
        query = query.filter(trigger_key=trigger_key)

    if search:
        query = query.filter(title__icontains=search)

    subscriptions = await query.order_by("-created_at")

    # Build response items with last event timestamps
    items: List[api_models.TriggerSubscriptionListItem] = []
    for sub in subscriptions:
        # Get last event for this subscription
        last_event = await TriggerEvent.filter(
            subscription_id=sub.id
        ).order_by("-received_at").first()

        items.append(api_models.TriggerSubscriptionListItem(
            id=sub.id,
            trigger_id=sub.trigger_id,
            trigger_key=sub.trigger_key,
            title=sub.title or None,
            enabled=sub.enabled,
            workflow_id=make_workflow_public_id(sub.workflow_id),
            workflow_title=sub.workflow.name if sub.workflow else "Deleted Workflow",
            last_event_at=last_event.received_at if last_event else None,
            created_at=sub.created_at,
        ))

    return api_models.TriggerSubscriptionListItemsResponse(items=items)


async def create_trigger_subscription(
    user: User,
    payload: api_models.TriggerSubscriptionCreateRequest,
) -> api_models.TriggerSubscriptionResponse:
    workflow = await _get_workflow(user, payload.workflow_id)
    definition = _load_trigger_definition(payload.trigger_key)
    filters = dict(payload.filters or {})
    provider_config = dict(payload.provider_config or {})
    _validate_filters_payload(filters, definition)
    _validate_provider_config(provider_config, definition)

    # Validate provider connection for triggers that require it
    provider_connection_id = payload.provider_connection_id
    if definition.meta.requires_connection:
        if provider_connection_id is None:
            # Attempt auto-selection (only works with single account)
            try:
                provider_connection_id = await _auto_select_provider_connection(user, definition)
            except MultipleAccountsError as exc:
                _raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Multiple accounts available",
                    detail=str(exc),
                    status=422,
                )

            if provider_connection_id is None:
                _raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing trigger connection",
                    detail=(
                        f"Trigger '{payload.trigger_key}' requires a provider connection. "
                        f"Please connect your {definition.provider} account first."
                    ),
                    status=400,
                )

    webhook_slug = None
    if _should_emit_webhook_url(payload.trigger_key):
        webhook_slug = _generate_webhook_slug()
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
        provider_connection_id=provider_connection_id,
        enabled=payload.enabled,
        filters=filters,
        provider_config=provider_config,
        webhook_slug=webhook_slug,
        form_suffix=payload.form_suffix,
        form_fields=payload.form_fields,
        form_config=payload.form_config,
        poll_interval_seconds=adjusted_interval,
    )
    if payload.trigger_key == "webhook.supabase.db_changes" and webhook_slug:
        await _create_supabase_webhook(subscription)
    return await _serialize_subscription(subscription)


async def get_trigger_subscription(
    user: User,
    subscription_id: int,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    return await _serialize_subscription(subscription)


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
        new_provider_config = dict(payload.provider_config or {})
        _validate_provider_config(new_provider_config, definition)
        subscription.provider_config = new_provider_config
    if payload.enabled is not None:
        subscription.enabled = payload.enabled
    if _should_emit_webhook_url(subscription.trigger_key) and not subscription.webhook_slug:
        subscription.webhook_slug = _generate_webhook_slug()


async def update_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionUpdateRequest,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    definition = _load_trigger_definition(subscription.trigger_key)
    _apply_subscription_updates(subscription, payload, definition)
    await subscription.save()
    return await _serialize_subscription(subscription)


async def delete_trigger_subscription(user: User, subscription_id: int) -> None:
    subscription = await _get_trigger_subscription(user, subscription_id)
    # For Supabase webhook triggers, clean up the database trigger
    if subscription.trigger_key == "webhook.supabase.db_changes":
        try:
            await delete_database_webhook(subscription)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Webhook cleanup is best-effort, shouldn't block deletion
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


# ======================================================================
# Form trigger helpers
# ======================================================================


def _extract_form_config_from_spec(trigger_spec) -> tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Extract form configuration from TriggerSpec.ui_meta.

    Returns:
        (form_suffix, form_fields, form_config)
    """
    if trigger_spec.key != "form.hosted":
        return None, None, None

    ui_meta = trigger_spec.ui_meta or {}
    form_config_raw = ui_meta.get("form_config", {})

    # Extract suffix (required)
    form_suffix = form_config_raw.get("suffix")

    # Build form_config dict (metadata only)
    form_config_json = {
        "title": form_config_raw.get("title", "Form"),
        "description": form_config_raw.get("description", ""),
        "submitButtonText": form_config_raw.get("submitButtonText", "Submit"),
        "successMessage": form_config_raw.get("successMessage", "Thank you for your submission!"),
        "styling": form_config_raw.get("styling", {}),
    }

    # Convert JSON schema to form_fields array
    form_fields = _json_schema_to_form_fields(trigger_spec.event_schema)

    return form_suffix, form_fields, form_config_json


def _json_schema_to_form_fields(event_schema: JsonSchema) -> List[Dict[str, Any]]:
    """
    Convert event_schema.properties.data (JSON schema) to form_fields array.
    """
    if not event_schema:
        return []
    properties = event_schema.get("properties", {})
    data_schema = properties.get("data", {})
    data_properties = data_schema.get("properties", {})
    required_fields = set(data_schema.get("required", []))

    form_fields = []
    for field_name, field_schema in data_properties.items():
        field_type = field_schema.get("type", "string")

        # Map JSON schema format to form field type
        mapped_type = "text"
        if field_type == "string":
            format_val = field_schema.get("format")
            if format_val == "email":
                mapped_type = "email"
            elif format_val == "uri":
                mapped_type = "url"
        elif field_type in ("number", "integer"):
            mapped_type = "number"
        elif field_type == "object":
            mapped_type = "object"

        form_fields.append({
            "name": field_name,
            "displayLabel": field_schema.get("title", field_name),
            "description": field_schema.get("description", ""),
            "type": mapped_type,
            "required": field_name in required_fields,
            "placeholder": field_schema.get("examples", [None])[0] if field_schema.get("examples") else None,
        })

    return form_fields


async def _validate_form_suffix_uniqueness(
    user: User,
    form_suffix: Optional[str],
    existing_subscription: Optional[TriggerSubscription] = None,
) -> None:
    """
    Ensure form_suffix is unique across all user's forms.
    """
    if not form_suffix:
        return

    query = TriggerSubscription.filter(
        user=user,
        form_suffix=form_suffix,
        trigger_key="form.hosted",
    )

    # Exclude current subscription if updating
    if existing_subscription:
        query = query.exclude(id=existing_subscription.id)

    existing = await query.first()
    if existing:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Duplicate form suffix",
            detail=f"Form suffix '{form_suffix}' is already in use. Choose a unique suffix.",
            status=400,
        )


async def _reconcile_existing_subscriptions(
    user: User,
    workflow: Workflow,
    desired_ids: set,
) -> Dict[str, TriggerSubscription]:
    """Build a map of existing subscriptions, cleaning up duplicates and stale entries."""
    existing: Dict[str, TriggerSubscription] = {}
    duplicate_subscriptions: List[TriggerSubscription] = []
    for sub in await TriggerSubscription.filter(workflow=workflow):
        if sub.trigger_id in existing:
            duplicate_subscriptions.append(sub)
        else:
            existing[sub.trigger_id] = sub

    for duplicate in duplicate_subscriptions:
        await delete_trigger_subscription(user, duplicate.id)

    for trigger_id, subscription in existing.items():
        if trigger_id not in desired_ids:
            await delete_trigger_subscription(user, subscription.id)

    return existing


async def sync_trigger_subscriptions(  # pylint: disable=too-complex,too-many-locals,too-many-branches  # Reason: Comprehensive trigger sync requires extensive state reconciliation
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
    desired_ids = {trigger.id for trigger in spec.triggers or []}
    existing = await _reconcile_existing_subscriptions(user, workflow, desired_ids)

    # Upsert declared triggers.
    for trigger_spec in spec.triggers or []:
        definition = _load_trigger_definition(trigger_spec.key)

        # Extract or auto-select provider connection
        provider_connection_id = trigger_spec.provider_config.get("provider_connection_id")

        if not skip_validation and definition.meta.requires_connection:
            if provider_connection_id is None:
                # Attempt auto-selection (only works with single account)
                try:
                    provider_connection_id = await _auto_select_provider_connection(user, definition)
                except MultipleAccountsError as exc:
                    _raise_problem(
                        type_uri=VALIDATION_PROBLEM,
                        title="Multiple accounts available",
                        detail=str(exc),
                        status=422,
                    )

                if provider_connection_id is None:
                    _raise_problem(
                        type_uri=VALIDATION_PROBLEM,
                        title="Missing trigger connection",
                        detail=(
                            f"Trigger '{trigger_spec.key}' requires a provider connection. "
                            f"Please connect your {definition.provider} account first."
                        ),
                        status=400,
                    )
                else:
                    # Store auto-selected connection for persistence
                    trigger_spec.provider_config["provider_connection_id"] = provider_connection_id

        filters = dict(trigger_spec.filters or {})
        provider_config = dict(trigger_spec.provider_config or {})

        _validate_filters_payload(filters, definition)
        if not skip_validation:
            _validate_provider_config(provider_config, definition)

        # Extract form configuration from ui_meta (for form triggers)
        form_suffix, form_fields, form_config = _extract_form_config_from_spec(trigger_spec)

        # Validate form suffix if present
        if form_suffix and not skip_validation:
            _validate_form_suffix(form_suffix)
            await _validate_form_suffix_uniqueness(user, form_suffix, existing.get(trigger_spec.id))

        adjusted_interval, warning = await _validate_and_adjust_poll_interval(user, 60)
        if warning:
            logger.warning(
                "Poll interval adjusted for user %s: %s",
                user.id,
                warning,
                extra={"user_id": user.id, "adjusted_interval": adjusted_interval},
            )

        existing_subscription = existing.get(trigger_spec.id)
        previous_slug = getattr(existing_subscription, "webhook_slug", None)
        webhook_slug = previous_slug
        if _should_emit_webhook_url(trigger_spec.key) and not webhook_slug:
            webhook_slug = _generate_webhook_slug()

        if existing_subscription:
            existing_subscription.trigger_key = trigger_spec.key  # Update type reference
            existing_subscription.title = definition.title  # Update title for reference resolution from registry
            existing_subscription.provider_connection_id = trigger_spec.provider_config.get("provider_connection_id")
            existing_subscription.enabled = True  # Always enabled when in spec
            existing_subscription.filters = filters
            existing_subscription.provider_config = provider_config
            existing_subscription.webhook_slug = webhook_slug
            existing_subscription.poll_interval_seconds = adjusted_interval
            # Sync form data for form triggers — preserve auto-generated suffix
            if form_suffix:
                existing_subscription.form_suffix = form_suffix
            elif not existing_subscription.form_suffix and _is_form_trigger(trigger_spec.key):
                existing_subscription.form_suffix = _generate_form_suffix()
            existing_subscription.form_fields = form_fields
            existing_subscription.form_config = form_config
            await existing_subscription.save()

            # If we generated a new slug for Supabase, ensure webhook is created.
            if (
                not skip_validation
                and trigger_spec.key == "webhook.supabase.db_changes"
                and webhook_slug
                and not previous_slug
            ):
                await _create_supabase_webhook(existing_subscription)
        else:
            is_polling = trigger_spec.key in POLLING_TRIGGERS
            # Auto-generate form suffix if not provided and this is a form trigger
            effective_form_suffix = form_suffix or (_generate_form_suffix() if _is_form_trigger(trigger_spec.key) else None)
            subscription = await TriggerSubscription.create(
                user=user,
                workflow=workflow,
                trigger_id=trigger_spec.id,
                trigger_key=trigger_spec.key,
                title=definition.title,
                provider_connection_id=trigger_spec.provider_config.get("provider_connection_id"),
                enabled=True,
                filters=filters,
                provider_config=provider_config,
                webhook_slug=webhook_slug,
                poll_interval_seconds=adjusted_interval,
                is_polling=is_polling,
                # Set form data for form triggers
                form_suffix=effective_form_suffix,
                form_fields=form_fields,
                form_config=form_config,
            )
            if not skip_validation and trigger_spec.key == "webhook.supabase.db_changes" and webhook_slug:
                await _create_supabase_webhook(subscription)


async def _provision_form_listening(
    user: User,
    workflow: Workflow,
    trigger_id: str,
    trigger_spec,
    *,
    definition,
    existing: Optional[TriggerSubscription],
) -> api_models.StartListeningResponse:
    """Provision or update a form trigger subscription and return the form URL."""
    form_suffix, form_fields, form_config = _extract_form_config_from_spec(trigger_spec)

    if existing and existing.form_suffix:
        subscription = existing
        subscription.form_fields = form_fields
        subscription.form_config = form_config
        await subscription.save()
    else:
        suffix = form_suffix or _generate_form_suffix()
        if existing:
            existing.form_suffix = suffix
            existing.form_fields = form_fields
            existing.form_config = form_config
            existing.enabled = True
            await existing.save()
            subscription = existing
        else:
            subscription = await TriggerSubscription.create(
                user=user,
                workflow=workflow,
                trigger_id=trigger_id,
                trigger_key=trigger_spec.key,
                title=definition.title,
                provider_connection_id=trigger_spec.provider_config.get("provider_connection_id"),
                enabled=True,
                filters=dict(trigger_spec.filters or {}),
                provider_config=dict(trigger_spec.provider_config or {}),
                form_suffix=suffix,
                form_fields=form_fields,
                form_config=form_config,
            )

    form_url = _build_form_url(subscription)
    return api_models.StartListeningResponse(
        form_url=form_url,
        subscription_id=subscription.id,
    )


async def _provision_webhook_listening(
    user: User,
    workflow: Workflow,
    trigger_id: str,
    trigger_spec,
    *,
    definition,
    existing: Optional[TriggerSubscription],
) -> api_models.StartListeningResponse:
    """Provision or update a webhook trigger subscription and return the webhook URL."""
    if existing and existing.webhook_slug:
        subscription = existing
    else:
        webhook_slug = _generate_webhook_slug()
        if existing:
            existing.webhook_slug = webhook_slug
            existing.enabled = True
            await existing.save()
            subscription = existing
        else:
            subscription = await TriggerSubscription.create(
                user=user,
                workflow=workflow,
                trigger_id=trigger_id,
                trigger_key=trigger_spec.key,
                title=definition.title,
                provider_connection_id=trigger_spec.provider_config.get("provider_connection_id"),
                enabled=True,
                filters=dict(trigger_spec.filters or {}),
                provider_config=dict(trigger_spec.provider_config or {}),
                webhook_slug=webhook_slug,
            )

    webhook_url = _build_webhook_url(subscription.webhook_slug, subscription.trigger_key)
    if not webhook_url:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Could not build webhook URL",
            detail="Failed to generate webhook URL for this trigger type.",
            status=500,
        )

    webhook_base_url = shared_config.webhook_base_url or "http://localhost:8000"
    full_url = f"{webhook_base_url.rstrip('/')}{webhook_url}"

    return api_models.StartListeningResponse(
        webhook_url=full_url,
        secret_token=None,  # Deprecated: slug-based URLs don't need secrets
        subscription_id=subscription.id,
    )


async def start_listening_for_trigger(
    user: User,
    workflow_id: str,
    trigger_id: str,
) -> api_models.StartListeningResponse:
    """
    Provision a TriggerSubscription for a webhook or form trigger immediately (without publish).

    Idempotent: if a subscription already exists for this trigger, reuse it.
    For webhook triggers: returns the webhook URL and signing secret.
    For form triggers: returns the form URL.
    """
    workflow = await _get_workflow(user, workflow_id)

    draft_version = await _get_draft_version(workflow, create_if_missing=False)
    if not draft_version:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No draft found",
            detail="Cannot start listening without a draft version. Save the workflow first.",
            status=400,
        )

    spec = WorkflowSpec.model_validate(draft_version.spec)
    trigger_spec = next((t for t in (spec.triggers or []) if t.id == trigger_id), None)

    if trigger_spec is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger not found in spec",
            detail=f"Trigger '{trigger_id}' not found in workflow draft.",
            status=404,
        )

    if not _should_emit_webhook_url(trigger_spec.key) and not _is_form_trigger(trigger_spec.key):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Unsupported trigger type",
            detail=f"Trigger '{trigger_spec.key}' does not support start listening.",
            status=400,
        )

    definition = _load_trigger_definition(trigger_spec.key)
    existing = await TriggerSubscription.filter(
        workflow=workflow, trigger_id=trigger_id
    ).first()

    if _is_form_trigger(trigger_spec.key):
        return await _provision_form_listening(user, workflow, trigger_id, trigger_spec, definition=definition, existing=existing)

    return await _provision_webhook_listening(user, workflow, trigger_id, trigger_spec, definition=definition, existing=existing)


async def get_pending_events(
    user: User,
    workflow_id: str,
    trigger_id: str,
    since: Optional[int] = None,
) -> api_models.PendingEventsResponse:
    """
    Fetch recent webhook events for a trigger subscription.

    Used by the frontend to poll for events while the user is in "Start Listening" mode.
    Filters directly by subscription_id for per-subscription isolation.
    """
    workflow = await _get_workflow(user, workflow_id)

    subscription = await TriggerSubscription.filter(
        workflow=workflow, trigger_id=trigger_id
    ).first()

    if not subscription:
        return api_models.PendingEventsResponse(events=[], latest_event_id=None)

    # Direct indexed lookup by subscription_id — no trigger_key scan or in-memory filtering needed
    query = TriggerEvent.filter(
        subscription_id=subscription.id,
    ).order_by("-received_at")

    if since is not None:
        query = query.filter(id__gt=since)

    events = await query.limit(20)

    items: list[api_models.PendingEventItem] = []
    latest_id: Optional[int] = None
    for event in events:
        envelope = event.event or {}
        data = envelope.get("data", {})
        received_at = event.received_at.isoformat() if event.received_at else ""
        items.append(api_models.PendingEventItem(
            event_id=event.id,
            data=data if isinstance(data, dict) else {},
            received_at=received_at,
        ))
        if latest_id is None or event.id > latest_id:
            latest_id = event.id

    return api_models.PendingEventsResponse(events=items, latest_event_id=latest_id)


async def get_subscription_event_count(
    user: User,
    subscription_id: int,
) -> api_models.SubscriptionEventCountResponse:
    """
    Get the count of stored events for a trigger subscription.

    Used by the frontend to determine if "Browse events" should be shown
    for persisted triggers (webhooks, forms).
    """
    # Verify subscription exists and belongs to user
    subscription = await TriggerSubscription.get_or_none(
        id=subscription_id, user=user
    )
    if not subscription:
        raise HTTPException(
            status_code=404,
            detail=f"Subscription {subscription_id} not found",
        )

    # Count events for this subscription
    event_count = await TriggerEvent.filter(
        subscription_id=subscription_id
    ).count()

    return api_models.SubscriptionEventCountResponse(
        subscription_id=subscription_id,
        event_count=event_count,
        has_events=event_count > 0,
    )


def generate_cron_event(trigger_id: str, provider_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a synthetic cron event for immediate workflow execution.

    This allows users to trigger cron-based workflows on demand without
    waiting for the next scheduled run.
    """
    now = datetime.now(timezone.utc)
    return build_event_envelope(TriggerEventEnvelopeInput(
        trigger_id=trigger_id,
        trigger_key="schedule.cron",
        title="Manual Trigger",
        provider="schedule",
        provider_connection_id=None,
        payload={
            "scheduled_time": now.isoformat(),
            "actual_time": now.isoformat(),
            "cron_expression": provider_config.get("cron_expression", "* * * * *"),
            "timezone": provider_config.get("timezone", "UTC"),
            "manual": True,
        },
        raw=None,
        occurred_at=now,
    ))
