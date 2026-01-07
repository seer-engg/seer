"""Trigger subscription management and event binding validation."""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.workflows import models as api_models
from api.workflows.services._shared import (
    VALIDATION_PROBLEM,
    _raise_problem,
    _spec_to_dict,
)
from shared.database.base import async_session_maker
from shared.database.models import (
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

# ===== Helper Functions =====


def _load_trigger_definition(trigger_key: str):
    """Load trigger definition from registry, raise if not found."""
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
    """Validate trigger filters against schema."""
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
    """Check if value is an expression binding (${...})."""
    return isinstance(value, str) and value.strip().startswith("${") and value.strip().endswith("}")


def _extract_event_path(expression: str) -> List[str]:
    """Extract property path from event binding expression."""
    content = expression.strip()[2:-1].strip()
    if not content.startswith("event."):
        raise ValueError("Bindings must reference event.*")
    segments = [segment for segment in content.split(".") if segment]
    if len(segments) < 2:
        raise ValueError("Binding must reference at least one event property")
    return segments[1:]


def _resolve_schema_for_path(schema: Dict[str, Any], segments: List[str]) -> Dict[str, Any]:
    """Resolve schema fragment for a given property path."""
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
    """Check if schema fragment type is compatible with input definition."""
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
    """Check if literal value type is compatible with input definition."""
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
    trigger_definition=None,
) -> None:
    """
    Validate trigger bindings against workflow input schema.

    Checks that all bindings reference valid inputs and have compatible types.

    For workflows without inputs (spec.inputs is empty), bindings are allowed
    as "late-bound" - they will be validated at runtime instead of subscription
    creation time. This enables triggers to work with workflows that don't
    define input schemas.

    For auto-binding triggers (e.g., Form trigger), bindings should be empty
    and will be auto-generated from workflow inputs.
    """
    # Handle auto-binding mode (Form trigger)
    if trigger_definition and trigger_definition.binding_mode == "auto":
        if not spec.inputs:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Form trigger requires workflow inputs",
                detail="Workflows using form triggers must define inputs that match the form fields.",
                status=400,
            )
        if bindings:
            _raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Form trigger does not accept manual bindings",
                detail="Form triggers automatically map form fields to workflow inputs. Do not provide bindings.",
                status=400,
            )
        # Validation passes - bindings will be auto-generated
        return

    if not bindings:
        return

    # Allow late-bound inputs for workflows without input schema
    # These bindings will be validated at runtime when the trigger fires
    if not spec.inputs:
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
    """Generate secure random token for webhook subscriptions."""
    return secrets.token_urlsafe(24)


def _should_emit_webhook_url(trigger_key: str) -> bool:
    """Check if trigger type should include webhook URL in response."""
    return trigger_key.startswith("webhook.")


def _build_webhook_url(subscription_id: int, trigger_key: str) -> Optional[str]:
    """Build webhook URL for subscription if applicable."""
    if trigger_key == "webhook.generic":
        return f"/v1/webhooks/generic/{subscription_id}"
    return None


def _serialize_subscription(subscription: TriggerSubscription) -> api_models.TriggerSubscriptionResponse:
    """Serialize trigger subscription to API response."""
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
    """Resolve value from event payload using property path."""
    current = payload
    for segment in segments:
        if not isinstance(current, dict):
            raise ValueError(f"Cannot traverse into '{segment}' on non-object value")
        if segment not in current:
            raise ValueError(f"Event payload is missing '{segment}'")
        current = current[segment]
    return current


def _evaluate_bindings(bindings: Dict[str, Any], event_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate bindings to produce workflow inputs from event payload."""
    resolved: Dict[str, Any] = {}
    for key, value in (bindings or {}).items():
        if _is_expression(value):
            path = _extract_event_path(value)
            resolved[key] = _resolve_event_value(event_payload, path)
        else:
            resolved[key] = value
    return resolved


def _validate_event_payload(event_payload: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate event payload against trigger event schema."""
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
    """Validate resolved workflow inputs meet required constraints."""
    errors: List[str] = []
    for name, input_def in (spec.inputs or {}).items():
        if input_def.required and input_def.default is None and name not in resolved:
            errors.append(f"Missing required input '{name}'")
        elif name in resolved and not _literal_value_matches_input(resolved[name], input_def):
            errors.append(f"Input '{name}' has incompatible type")
    return errors


async def _get_trigger_subscription(user: User, subscription_id: int, session: AsyncSession) -> TriggerSubscription:
    """Get trigger subscription by ID for user."""
    try:
        pk = int(subscription_id)
    except (TypeError, ValueError):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid subscription id",
            detail="Subscription id must be an integer",
            status=400,
        )
    stmt = (
        select(TriggerSubscription)
        .where(TriggerSubscription.id == pk, TriggerSubscription.user_id == user.id)
        .options(selectinload(TriggerSubscription.workflow))
    )
    result = await session.execute(stmt)
    subscription = result.scalar_one_or_none()
    if subscription is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger subscription not found",
            detail=f"Subscription '{subscription_id}' not found",
            status=404,
        )
    return subscription


# ===== Public API Functions =====


async def list_trigger_subscriptions(
    user: User,
    *,
    workflow_id: Optional[str] = None,
) -> api_models.TriggerSubscriptionListResponse:
    """List trigger subscriptions for user, optionally filtered by workflow."""
    # Import here to avoid circular dependency
    from api.workflows.services.workflows import _get_workflow

    async with async_session_maker() as session:
        stmt = select(TriggerSubscription).where(TriggerSubscription.user_id == user.id)
        if workflow_id:
            workflow = await _get_workflow(user, workflow_id, session)
            stmt = stmt.where(TriggerSubscription.workflow_id == workflow.id)
        stmt = stmt.order_by(TriggerSubscription.created_at.desc())
        result = await session.execute(stmt)
        subscriptions = result.scalars().all()
        return api_models.TriggerSubscriptionListResponse(
            items=[_serialize_subscription(item) for item in subscriptions],
        )


async def create_trigger_subscription(
    user: User,
    payload: api_models.TriggerSubscriptionCreateRequest,
) -> api_models.TriggerSubscriptionResponse:
    """Create a new trigger subscription for a workflow."""
    # Import here to avoid circular dependency
    from api.workflows.services.workflows import _get_workflow

    async with async_session_maker() as session:
        workflow = await _get_workflow(user, payload.workflow_id, session)
        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()
        definition = _load_trigger_definition(payload.trigger_key)
        spec = WorkflowSpec.model_validate(draft.spec)
        filters = dict(payload.filters or {})
        bindings = dict(payload.bindings or {})
        provider_config = dict(payload.provider_config or {})
        _validate_filters_payload(filters, definition)
        _validate_bindings_against_workflow(bindings, spec, definition.event_schema, definition)

        # Auto-generate bindings for form triggers
        if definition.binding_mode == "auto":
            bindings = {name: f"${{event.data.{name}}}" for name in spec.inputs.keys()}

        secret = _generate_subscription_secret() if _should_emit_webhook_url(payload.trigger_key) else None
        subscription = TriggerSubscription(
            user_id=user.id,
            workflow_id=workflow.id,
            trigger_key=payload.trigger_key,
            provider_connection_id=payload.provider_connection_id,
            enabled=payload.enabled,
            filters=filters,
            bindings=bindings,
            provider_config=provider_config,
            secret_token=secret,
        )
        session.add(subscription)
        await session.commit()
        await session.refresh(subscription)
        return _serialize_subscription(subscription)


async def get_trigger_subscription(
    user: User,
    subscription_id: int,
) -> api_models.TriggerSubscriptionResponse:
    """Get a specific trigger subscription."""
    async with async_session_maker() as session:
        subscription = await _get_trigger_subscription(user, subscription_id, session)
        return _serialize_subscription(subscription)


async def update_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionUpdateRequest,
) -> api_models.TriggerSubscriptionResponse:
    """Update an existing trigger subscription."""
    async with async_session_maker() as session:
        subscription = await _get_trigger_subscription(user, subscription_id, session)
        definition = _load_trigger_definition(subscription.trigger_key)

        # Load workflow and draft
        stmt = select(Workflow).where(Workflow.id == subscription.workflow_id).options(selectinload(Workflow.draft))
        result = await session.execute(stmt)
        workflow = result.scalar_one()

        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()

        spec = WorkflowSpec.model_validate(draft.spec)
        if payload.filters is not None:
            new_filters = dict(payload.filters or {})
            _validate_filters_payload(new_filters, definition)
            subscription.filters = new_filters
        if payload.bindings is not None:
            new_bindings = dict(payload.bindings or {})
            _validate_bindings_against_workflow(new_bindings, spec, definition.event_schema, definition)
            subscription.bindings = new_bindings
        if payload.provider_connection_id is not None:
            subscription.provider_connection_id = payload.provider_connection_id
        if payload.provider_config is not None:
            subscription.provider_config = dict(payload.provider_config or {})
        if payload.enabled is not None:
            subscription.enabled = payload.enabled
        if _should_emit_webhook_url(subscription.trigger_key) and not subscription.secret_token:
            subscription.secret_token = _generate_subscription_secret()

        session.add(subscription)
        await session.commit()
        await session.refresh(subscription)
        return _serialize_subscription(subscription)


async def delete_trigger_subscription(user: User, subscription_id: int) -> None:
    """Delete a trigger subscription."""
    async with async_session_maker() as session:
        subscription = await _get_trigger_subscription(user, subscription_id, session)
        await session.delete(subscription)
        await session.commit()


async def test_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionTestRequest,
) -> api_models.TriggerSubscriptionTestResponse:
    """
    Test trigger subscription binding evaluation.

    Validates event payload and evaluates bindings to produce workflow inputs.
    """
    async with async_session_maker() as session:
        subscription = await _get_trigger_subscription(user, subscription_id, session)

        # Load workflow and draft
        stmt = select(Workflow).where(Workflow.id == subscription.workflow_id).options(selectinload(Workflow.draft))
        result = await session.execute(stmt)
        workflow = result.scalar_one()

        draft = workflow.draft
        if not draft:
            stmt = select(WorkflowDraft).where(WorkflowDraft.workflow_id == workflow.id)
            result = await session.execute(stmt)
            draft = result.scalar_one()

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
