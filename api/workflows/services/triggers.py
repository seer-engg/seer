"""Trigger subscription management and event binding validation."""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator

from api.workflows import models as api_models
from api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _raise_problem,
    _get_workflow
)
from shared.database.workflow_models import (
    TriggerSubscription,
    User,
    Workflow,
    WorkflowDraft,
    make_workflow_public_id,
)
from workflow_compiler.registry.trigger_registry import trigger_registry
from workflow_compiler.schema.models import (
    InputDef,
    InputType,
    WorkflowSpec,
)
from shared.logger import get_logger
from api.triggers.supabase_webhook import (
    create_database_webhook,
    delete_database_webhook,
    SupabaseWebhookError,
)
from shared.config import config as shared_config
from fastapi import HTTPException
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
    schema = definition.filter_schema
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


def _resolve_schema_for_path(schema: Dict[str, Any], segments: List[str]) -> Dict[str, Any]:
    current = schema
    for segment in segments:
        schema_type = current.get("type")
        schema_types = [schema_type] if isinstance(schema_type, str) else schema_type or []
        if "object" in schema_types or not schema_types:
            props = current.get("properties", {})
            if segment in props:
                current = props[segment]
                continue
            additional = current.get("additionalProperties", True)
            if isinstance(additional, dict):
                current = additional
                continue
            if additional:
                current = {}
                continue
            raise ValueError(f"Property '{segment}' is not allowed on event")
        else:
            raise ValueError(f"Cannot descend into non-object property '{segment}'")
    return current

def _schema_fragment_matches_input(fragment: Dict[str, Any], input_def: InputDef) -> bool:
    schema_type = fragment.get("type")
    if not schema_type:
        return True
    schema_types = [schema_type] if isinstance(schema_type, str) else list(schema_type)
    if input_def.type == InputType.string:
        return "string" in schema_types
    if input_def.type == InputType.integer:
        return "integer" in schema_types
    if input_def.type == InputType.number:
        return "number" in schema_types or "integer" in schema_types
    if input_def.type == InputType.boolean:
        return "boolean" in schema_types
    if input_def.type == InputType.object:
        return "object" in schema_types
    if input_def.type == InputType.array:
        return "array" in schema_types
    return True


def _literal_value_matches_input(value: Any, input_def: InputDef) -> bool:
    if value is None:
        return not input_def.required
    if input_def.type == InputType.string:
        return isinstance(value, str)
    if input_def.type == InputType.integer:
        return isinstance(value, int) and not isinstance(value, bool)
    if input_def.type == InputType.number:
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    if input_def.type == InputType.boolean:
        return isinstance(value, bool)
    if input_def.type == InputType.object:
        return isinstance(value, dict)
    if input_def.type == InputType.array:
        return isinstance(value, list)
    return True



def _validate_bindings_against_workflow(
    bindings: Dict[str, Any],
    spec: WorkflowSpec,
    event_schema: Dict[str, Any],
) -> None:
    if not bindings:
        return
    errors: List[api_models.ProblemError] = []
    for input_name, binding in bindings.items():
        input_def = spec.inputs.get(input_name)
        if input_def is None:
            errors.append(
                api_models.ProblemError(
                    code="UNKNOWN_INPUT",
                    message=f"Input '{input_name}' is not defined on workflow",
                )
            )
            continue
        if _is_expression(binding):
            try:
                path = _extract_event_path(binding)
                fragment = _resolve_schema_for_path(event_schema or {}, path)
            except ValueError as exc:
                errors.append(
                    api_models.ProblemError(
                        code="INVALID_BINDING",
                        message=str(exc),
                        expression=binding,
                    )
                )
                continue
            if not _schema_fragment_matches_input(fragment, input_def):
                errors.append(
                    api_models.ProblemError(
                        code="TYPE_MISMATCH",
                        message=f"Binding for '{input_name}' is incompatible with expected input type '{input_def.type.value}'",
                        expression=binding,
                    )
                )
        else:
            if not _literal_value_matches_input(binding, input_def):
                errors.append(
                    api_models.ProblemError(
                        code="TYPE_MISMATCH",
                        message=f"Literal binding for '{input_name}' has incompatible type",
                    )
                )
    if errors:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid trigger bindings",
            detail="One or more bindings are invalid",
            status=400,
            errors=errors,
        )


def _generate_subscription_secret() -> str:
    return secrets.token_urlsafe(24)


def _should_emit_webhook_url(trigger_key: str) -> bool:
    return trigger_key.startswith("webhook.")


def _build_webhook_url(subscription_id: int, trigger_key: str) -> Optional[str]:
    if trigger_key == "webhook.generic":
        return f"/v1/webhooks/generic/{subscription_id}"
    elif trigger_key == "webhook.supabase.db_changes":
        return f"/v1/webhooks/generic/{subscription_id}"
    return None

def _serialize_subscription(subscription: TriggerSubscription) -> api_models.TriggerSubscriptionResponse:
    webhook_url = _build_webhook_url(subscription.id, subscription.trigger_key) if _should_emit_webhook_url(subscription.trigger_key) else None
    return api_models.TriggerSubscriptionResponse(
        subscription_id=subscription.id,
        workflow_id=make_workflow_public_id(subscription.workflow_id),
        trigger_key=subscription.trigger_key,
        provider_connection_id=subscription.provider_connection_id,
        enabled=subscription.enabled,
        filters=dict(subscription.filters or {}),
        bindings=dict(subscription.bindings or {}),
        provider_config=dict(subscription.provider_config or {}),
        secret_token=subscription.secret_token,
        webhook_url=webhook_url,
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


def _validate_resolved_inputs(resolved: Dict[str, Any], spec: WorkflowSpec) -> List[str]:
    errors: List[str] = []
    for name, input_def in (spec.inputs or {}).items():
        if input_def.required and input_def.default is None and name not in resolved:
            errors.append(f"Missing required input '{name}'")
        elif name in resolved and not _literal_value_matches_input(resolved[name], input_def):
            errors.append(f"Input '{name}' has incompatible type")
    return errors


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
    draft = workflow.draft or await WorkflowDraft.get(workflow=workflow)
    definition = _load_trigger_definition(payload.trigger_key)
    spec = WorkflowSpec.model_validate(draft.spec)
    filters = dict(payload.filters or {})
    bindings = dict(payload.bindings or {})
    provider_config = dict(payload.provider_config or {})
    _validate_filters_payload(filters, definition)
    _validate_bindings_against_workflow(bindings, spec, definition.event_schema)
    secret = _generate_subscription_secret() if _should_emit_webhook_url(payload.trigger_key) else None
    subscription = await TriggerSubscription.create(
        user=user,
        workflow=workflow,
        trigger_key=payload.trigger_key,
        provider_connection_id=payload.provider_connection_id,
        enabled=payload.enabled,
        filters=filters,
        bindings=bindings,
        provider_config=provider_config,
        secret_token=secret,
    )
    # For Supabase webhook triggers, create the database trigger
    if payload.trigger_key == "webhook.supabase.db_changes" and secret:
        try:
            # Build full webhook URL with domain and secret
            webhook_base_url = shared_config.webhook_base_url or "http://localhost:8000"
            full_webhook_url = f"{webhook_base_url.rstrip('/')}/api/v1/webhooks/generic/{subscription.id}"

            # Create the Postgres trigger in Supabase
            webhook_metadata = await create_database_webhook(subscription, full_webhook_url, secret=secret)
            logger.info(
                "Created Supabase webhook",
                extra={"subscription_id": subscription.id, "metadata": webhook_metadata}
            )
        except SupabaseWebhookError as exc:
            # Rollback: delete the subscription and re-raise
            logger.error(
                "Failed to create Supabase webhook, rolling back subscription",
                extra={"subscription_id": subscription.id, "error": str(exc)}
            )
            await subscription.delete()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create Supabase webhook: {str(exc)}"
            )
    return _serialize_subscription(subscription)


async def get_trigger_subscription(
    user: User,
    subscription_id: int,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    return _serialize_subscription(subscription)


async def update_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionUpdateRequest,
) -> api_models.TriggerSubscriptionResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    definition = _load_trigger_definition(subscription.trigger_key)
    await subscription.fetch_related("workflow")
    draft = subscription.workflow.draft or await WorkflowDraft.get(workflow=subscription.workflow)
    spec = WorkflowSpec.model_validate(draft.spec)
    if payload.filters is not None:
        new_filters = dict(payload.filters or {})
        _validate_filters_payload(new_filters, definition)
        subscription.filters = new_filters
    if payload.bindings is not None:
        new_bindings = dict(payload.bindings or {})
        _validate_bindings_against_workflow(new_bindings, spec, definition.event_schema)
        subscription.bindings = new_bindings
    if payload.provider_connection_id is not None:
        subscription.provider_connection_id = payload.provider_connection_id
    if payload.provider_config is not None:
        subscription.provider_config = dict(payload.provider_config or {})
    if payload.enabled is not None:
        subscription.enabled = payload.enabled
    if _should_emit_webhook_url(subscription.trigger_key) and not subscription.secret_token:
        subscription.secret_token = _generate_subscription_secret()
    await subscription.save()
    return _serialize_subscription(subscription)


async def delete_trigger_subscription(user: User, subscription_id: int) -> None:
    subscription = await _get_trigger_subscription(user, subscription_id)
    subscription = await _get_trigger_subscription(user, subscription_id)
     # For Supabase webhook triggers, clean up the database trigger
    if subscription.trigger_key == "webhook.supabase.db_changes":
        try:
            await delete_database_webhook(subscription)
        except Exception as exc:
            # Log but don't block deletion - trigger cleanup is best-effort
            logger.warning(
                "Failed to delete Supabase webhook (non-fatal)",
                extra={"subscription_id": subscription_id, "error": str(exc)}
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
    subscription = await _get_trigger_subscription(user, subscription_id)
    await subscription.fetch_related("workflow")
    draft = subscription.workflow.draft or await WorkflowDraft.get(workflow=subscription.workflow)
    spec = WorkflowSpec.model_validate(draft.spec)
    definition = _load_trigger_definition(subscription.trigger_key)
    event_payload = payload.event or definition.sample_event
    if event_payload is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Event payload required",
            detail="Provide an event payload or configure a trigger sample.",
            status=400,
        )
    _validate_event_payload(event_payload, definition.event_schema)
    try:
        resolved = _evaluate_bindings(dict(subscription.bindings or {}), event_payload)
    except ValueError as exc:
        return api_models.TriggerSubscriptionTestResponse(inputs={}, errors=[str(exc)])
    errors = _validate_resolved_inputs(resolved, spec)
    return api_models.TriggerSubscriptionTestResponse(inputs=resolved, errors=errors)
