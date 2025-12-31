from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fastapi import HTTPException
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from tortoise.exceptions import DoesNotExist

from shared.database.models import User
from shared.database.workflow_models import (
    TriggerSubscription,
    WorkflowRecord,
    WorkflowRun,
    WorkflowRunStatus,
    make_workflow_public_id,
    parse_run_public_id,
    parse_workflow_public_id,
)
from shared.tools.base import list_tools as registry_list_tools
from shared.config import config as shared_config
from workflow_compiler.errors import WorkflowCompilerError, ValidationPhaseError
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.registry.trigger_registry import trigger_registry
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    InputDef,
    InputType,
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)
from workflow_compiler.expr import parser as expr_parser
from workflow_compiler.expr.typecheck import Scope, TypeEnvironment, typecheck_reference

from api.workflows import models as api_models
import traceback

from api.agents.checkpointer import get_checkpointer

compiler = WorkflowCompilerSingleton.instance()
logger = logging.getLogger(__name__)

PROBLEM_BASE = "https://seer.errors/workflows"
VALIDATION_PROBLEM = f"{PROBLEM_BASE}/validation"
COMPILE_PROBLEM = f"{PROBLEM_BASE}/compile"
RUN_PROBLEM = f"{PROBLEM_BASE}/run"


NODE_TYPE_DESCRIPTORS = api_models.NodeTypeResponse(
    node_types=[
        api_models.NodeTypeDescriptor(
            type="tool",
            title="Tool",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="tool", kind="select", required=True, source="tools"),
                api_models.NodeFieldDescriptor(name="in", kind="json"),
                api_models.NodeFieldDescriptor(name="out", kind="string"),
                api_models.NodeFieldDescriptor(name="expect_output", kind="output_contract"),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="llm",
            title="LLM",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="model", kind="select", required=True, source="models"),
                api_models.NodeFieldDescriptor(name="prompt", kind="textarea", required=True),
                api_models.NodeFieldDescriptor(name="in", kind="json"),
                api_models.NodeFieldDescriptor(name="out", kind="string"),
                api_models.NodeFieldDescriptor(name="output", kind="output_contract", required=True),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="if",
            title="If/Else",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="condition", kind="expression", required=True),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="for_each",
            title="For Each",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="items", kind="expression", required=True),
                api_models.NodeFieldDescriptor(name="item_var", kind="string"),
                api_models.NodeFieldDescriptor(name="index_var", kind="string"),
                api_models.NodeFieldDescriptor(name="out", kind="string"),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="task",
            title="Task",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="kind", kind="select", required=True),
                api_models.NodeFieldDescriptor(name="value", kind="json"),
                api_models.NodeFieldDescriptor(name="out", kind="string"),
            ],
        ),
    ]
)

DEFAULT_MODEL_REGISTRY = [
    api_models.ModelDescriptor(id="gpt-4.1-mini", title="GPT-4.1 mini", supports_json_schema=True),
    api_models.ModelDescriptor(id="gpt-4o-mini", title="GPT-4o mini", supports_json_schema=True),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec_to_dict(spec: WorkflowSpec) -> Dict[str, Any]:
    return spec.model_dump(mode="json")


def _raise_problem(
    *,
    type_uri: str,
    title: str,
    detail: str,
    status: int,
    errors: Optional[Sequence[api_models.ProblemError]] = None,
) -> None:
    payload = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "errors": [error.model_dump() for error in errors] if errors else [],
    }
    raise HTTPException(status_code=status, detail=payload)


async def list_node_types() -> api_models.NodeTypeResponse:
    return NODE_TYPE_DESCRIPTORS


async def list_tools(include_schemas: bool = False) -> api_models.ToolRegistryResponse:
    tools: List[api_models.ToolDescriptor] = []
    for tool in registry_list_tools():
        definition = compiler.ensure_tool(tool.name)
        descriptor = api_models.ToolDescriptor(
            id=f"tools.{definition.name}@{definition.version}",
            name=definition.name,
            version=definition.version,
            title=getattr(tool, "title", definition.name.replace("_", " ").title()),
            input_schema=definition.input_schema if include_schemas else None,
            output_schema=definition.output_schema if include_schemas else None,
        )
        tools.append(descriptor)
    return api_models.ToolRegistryResponse(tools=tools)


async def list_triggers() -> api_models.TriggerCatalogResponse:
    triggers = [
        api_models.TriggerDescriptor(
            key=definition.key,
            title=definition.title,
            provider=definition.provider,
            mode=definition.mode,
            description=definition.description,
            event_schema=definition.event_schema,
            filter_schema=definition.filter_schema,
        )
        for definition in trigger_registry.all()
    ]
    return api_models.TriggerCatalogResponse(triggers=triggers)


async def list_models() -> api_models.ModelRegistryResponse:
    models = list(DEFAULT_MODEL_REGISTRY)
    default_id = shared_config.default_llm_model
    if default_id and not any(model.id == default_id for model in models):
        models.append(
            api_models.ModelDescriptor(
                id=default_id,
                title=default_id,
                supports_json_schema=True,
            )
        )
    return api_models.ModelRegistryResponse(models=models)


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
        workflow = await _get_workflow_record(user, workflow_id)
        query = query.filter(workflow=workflow)
    subscriptions = await query.order_by("-created_at")
    return api_models.TriggerSubscriptionListResponse(
        items=[_serialize_subscription(item) for item in subscriptions],
    )


async def create_trigger_subscription(
    user: User,
    payload: api_models.TriggerSubscriptionCreateRequest,
) -> api_models.TriggerSubscriptionResponse:
    workflow = await _get_workflow_record(user, payload.workflow_id)
    definition = _load_trigger_definition(payload.trigger_key)
    spec = WorkflowSpec.model_validate(workflow.spec)
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
    workflow = await WorkflowRecord.get(id=subscription.workflow_id, user=user)
    spec = WorkflowSpec.model_validate(workflow.spec)
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
    await subscription.delete()


async def test_trigger_subscription(
    user: User,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionTestRequest,
) -> api_models.TriggerSubscriptionTestResponse:
    subscription = await _get_trigger_subscription(user, subscription_id)
    workflow = await WorkflowRecord.get(id=subscription.workflow_id, user=user)
    spec = WorkflowSpec.model_validate(workflow.spec)
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


async def resolve_schema(schema_id: str) -> api_models.SchemaResponse:
    schema = compiler.schema_registry.get(schema_id)
    if schema is None:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Schema not found",
            detail=f"Schema '{schema_id}' is not registered",
            status=404,
        )
    return api_models.SchemaResponse(id=schema_id, json_schema=schema)


def _collect_warnings_from_nodes(nodes: Iterable[Node]) -> List[api_models.WorkflowWarning]:
    warnings: List[api_models.WorkflowWarning] = []
    for node in nodes:
        if isinstance(node, (ToolNode, LLMNode)) and not node.out:
            warnings.append(
                api_models.WorkflowWarning(
                    code="OUT_MISSING",
                    node_id=node.id,
                    message=f"Node '{node.id}' has no 'out'; downstream references may fail.",
                )
            )
        if isinstance(node, IfNode):
            warnings.extend(_collect_warnings_from_nodes(node.then))
            warnings.extend(_collect_warnings_from_nodes(node.else_))
        if isinstance(node, ForEachNode):
            warnings.extend(_collect_warnings_from_nodes(node.body))
    return warnings


def validate_spec(payload: api_models.ValidateRequest) -> api_models.ValidateResponse:
    spec = payload.spec
    warnings = _collect_warnings_from_nodes(spec.nodes)
    return api_models.ValidateResponse(warnings=warnings)


def _graph_preview(spec: WorkflowSpec) -> Dict[str, Any]:
    nodes = [{"id": node.id, "kind": node.type} for node in spec.nodes]
    edges = []
    for idx in range(len(spec.nodes) - 1):
        edges.append({"from": spec.nodes[idx].id, "to": spec.nodes[idx + 1].id})
    return {"nodes": nodes, "edges": edges}


async def compile_spec(user: User, payload: api_models.CompileRequest) -> api_models.CompileResponse:
    spec = payload.spec
    spec_dict = _spec_to_dict(spec)
    checkpointer = await get_checkpointer()
    try:
        compiled = await compiler.compile(user, spec_dict, checkpointer=checkpointer)
    except WorkflowCompilerError as exc:
        _raise_problem(
            type_uri=COMPILE_PROBLEM,
            title="Compilation failed",
            detail=str(exc),
            status=400,
        )

    warnings = _collect_warnings_from_nodes(spec.nodes)
    artifacts = api_models.CompileArtifacts()
    if payload.options.emit_type_env:
        artifacts.type_env = dict(compiled.workflow.type_env)
    if payload.options.emit_graph_preview:
        artifacts.graph_preview = _graph_preview(spec)

    return api_models.CompileResponse(warnings=warnings, artifacts=artifacts)


def _workflow_summary(record: WorkflowRecord) -> api_models.WorkflowSummary:
    return api_models.WorkflowSummary(
        workflow_id=record.workflow_id,
        name=record.name,
        description=record.description,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _workflow_response(record: WorkflowRecord) -> api_models.WorkflowResponse:
    spec = WorkflowSpec.model_validate(record.spec)
    return api_models.WorkflowResponse(
        workflow_id=record.workflow_id,
        name=record.name,
        description=record.description,
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        spec=spec,
        tags=list(record.tags or []),
        meta=api_models.WorkflowMeta(last_compile_ok=record.last_compile_ok),
    )


def _parse_workflow_cursor(cursor: Optional[str]) -> Optional[int]:
    if cursor is None:
        return None
    try:
        if cursor.startswith("wf_"):
            return parse_workflow_public_id(cursor)
        return int(cursor)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid cursor",
            detail="Cursor parameter is invalid",
            status=400,
        )


async def create_workflow(user: User, payload: api_models.WorkflowCreateRequest) -> api_models.WorkflowResponse:
    record = await WorkflowRecord.create(
        user=user,
        name=payload.name,
        description=payload.description,
        spec=_spec_to_dict(payload.spec),
        version=1,
        tags=list(payload.tags or []),
    )
    return _workflow_response(record)


async def list_workflows(
    user: User,
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> api_models.WorkflowListResponse:
    limit = max(1, min(limit, 100))
    cursor_pk = _parse_workflow_cursor(cursor)

    query = WorkflowRecord.filter(user=user)
    if cursor_pk:
        query = query.filter(id__lt=cursor_pk)

    records = await query.order_by("-id").limit(limit + 1)
    items = [_workflow_summary(record) for record in records[:limit]]
    next_cursor = items[-1].workflow_id if len(records) > limit and items else None
    return api_models.WorkflowListResponse(items=items, next_cursor=next_cursor)


async def get_workflow(user: User, workflow_id: str) -> api_models.WorkflowResponse:
    record = await _get_workflow_record(user, workflow_id)
    return _workflow_response(record)


async def _get_workflow_record(user: User, workflow_id: str) -> WorkflowRecord:
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow id",
            detail="Workflow id is invalid",
            status=400,
        )
    try:
        return await WorkflowRecord.get(id=pk, user=user)
    except DoesNotExist:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Workflow not found",
            detail=f"Workflow '{workflow_id}' not found",
            status=404,
        )


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
    try:
        return await TriggerSubscription.get(id=pk, user=user)
    except DoesNotExist:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger subscription not found",
            detail=f"Subscription '{subscription_id}' not found",
            status=404,
        )


async def update_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.WorkflowUpdateRequest,
) -> api_models.WorkflowResponse:
    record = await _get_workflow_record(user, workflow_id)
    if payload.name is not None:
        record.name = payload.name
    if payload.description is not None:
        record.description = payload.description
    if payload.tags is not None:
        record.tags = list(payload.tags)
    if payload.spec is not None:
        record.spec = _spec_to_dict(payload.spec)
        record.version += 1
    await record.save()
    return _workflow_response(record)


async def apply_workflow_from_spec(
    user: User,
    workflow_id: str,
    spec_payload: Dict[str, Any],
) -> api_models.WorkflowResponse:
    """
    Replace an existing workflow's spec with a validated WorkflowSpec payload.
    """
    record = await _get_workflow_record(user, workflow_id)
    try:
        spec = WorkflowSpec.model_validate(spec_payload)
    except Exception as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid workflow spec",
            detail=str(exc),
            status=400,
        )

    record.spec = _spec_to_dict(spec)
    record.version += 1
    await record.save()
    return _workflow_response(record)


async def delete_workflow(user: User, workflow_id: str) -> None:
    record = await _get_workflow_record(user, workflow_id)
    await record.delete()


def _type_env_from_compiled(compiled) -> TypeEnvironment:
    return compiled.workflow.runtime.services.type_env


async def _prepare_type_env(user: User, spec: WorkflowSpec) -> TypeEnvironment:
    checkpointer = await get_checkpointer()
    compiled = await compiler.compile(
        user,
        _spec_to_dict(spec),
        checkpointer=checkpointer,
    )
    return _type_env_from_compiled(compiled)


def suggest_expression(user: User, payload: api_models.ExpressionSuggestRequest) -> api_models.ExpressionSuggestResponse:
    type_env = _prepare_type_env(user, payload.spec)
    prefix = payload.cursor_context.prefix.strip()
    if not prefix.startswith("${"):
        return api_models.ExpressionSuggestResponse()
    content = prefix[2:]
    if content.endswith("}"):
        content = content[:-1]
    if "." in content:
        base, partial = content.rsplit(".", 1)
    else:
        base, partial = content, ""
    if not base:
        return api_models.ExpressionSuggestResponse()
    try:
        reference = expr_parser.parse_reference_string(base)
        schema = typecheck_reference(reference, Scope(env=type_env))
    except Exception:
        return api_models.ExpressionSuggestResponse()

    props = schema.get("properties", {})
    suggestions: List[api_models.ExpressionSuggestion] = []
    for key, value in props.items():
        if partial and not key.startswith(partial):
            continue
        type_name = value.get("type") if isinstance(value, dict) else None
        suggestions.append(
            api_models.ExpressionSuggestion(label=key, insert=key, type=type_name)
        )
    return api_models.ExpressionSuggestResponse(suggestions=suggestions)


def typecheck_expression(user: User, payload: api_models.ExpressionTypecheckRequest) -> api_models.ExpressionTypecheckResponse:
    expression = payload.expression.strip()
    if not (expression.startswith("${") and expression.endswith("}")):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid expression",
            detail="Expression must be a ${...} reference",
            status=400,
        )
    content = expression[2:-1]
    try:
        reference = expr_parser.parse_reference_string(content)
    except ValueError as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid expression",
            detail=str(exc),
            status=400,
        )

    type_env = _prepare_type_env(user, payload.spec)
    try:
        schema = typecheck_reference(reference, Scope(env=type_env))
    except ValidationPhaseError as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Expression validation failed",
            detail=str(exc),
            status=400,
        )
    except Exception as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Expression validation failed",
            detail=str(exc),
            status=400,
        )
    return api_models.ExpressionTypecheckResponse(type=schema)


async def _create_run_record(
    user: User,
    *,
    workflow: Optional[WorkflowRecord],
    spec: WorkflowSpec,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
) -> WorkflowRun:
    return await WorkflowRun.create(
        user=user,
        workflow=workflow,
        workflow_version=workflow.version if workflow else None,
        spec=_spec_to_dict(spec),
        inputs=inputs or {},
        config=config_payload or {},
        status=WorkflowRunStatus.QUEUED,
    )


def _serialize_run(run: WorkflowRun) -> api_models.RunResponse:
    return api_models.RunResponse(
        run_id=run.run_id,
        status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
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
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        inputs=dict(run.inputs or {}),
        output=run.output,
        error=run.error,
    )


def _build_run_config(run: WorkflowRun, config_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure LangGraph defaults (thread_id) are present so checkpoints can be recovered.
    """

    base_config = dict((config_payload or {}) or {})
    configurable = dict((base_config.get("configurable") or {}) or {})
    configurable.setdefault("thread_id", run.run_id)
    base_config["configurable"] = configurable
    return base_config


async def _execute_compiled_run(
    run: WorkflowRun,
    user: User,
    *,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    logger.debug(
        "Preparing workflow run '%s' (workflow_id=%s) inputs_keys=%s config_payload_keys=%s user_id=%s",
        run.run_id,
        getattr(run.workflow, "workflow_id", None),
        sorted((inputs or {}).keys()),
        sorted((config_payload or {}).keys()),
        getattr(user, "id", None),
    )
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.RUNNING,
        started_at=_now(),
    )
    checkpointer = await get_checkpointer()
    try:
        compiled = await compiler.compile(
            user,
            run.spec,
            checkpointer=checkpointer,
        )
    except WorkflowCompilerError as exc:
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        _raise_problem(
            type_uri=COMPILE_PROBLEM,
            title="Compilation failed",
            detail=str(exc),
            status=400,
        )
    try:
        run_config = dict(config_payload or {})
        logger.debug(
            "Invoking compiled workflow for run '%s' with config_keys=%s user_context_id=%s",
            run.run_id,
            sorted(run_config.keys()),
            getattr(user, "id", None),
        )
        effective_config = _build_run_config(run, run_config)
        result = await compiled.ainvoke(inputs or {}, config=effective_config)
    except WorkflowCompilerError as exc:
        print(f"{traceback.format_exc()}")
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run failed",
            detail=str(exc),
            status=400,
        )
    except Exception as exc:
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run failed",
            detail=str(exc),
            status=400,
        )
    return result


async def _complete_run(run: WorkflowRun, output: Dict[str, Any]) -> WorkflowRun:
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.SUCCEEDED,
        finished_at=_now(),
        output=output,
    )
    await run.refresh_from_db()
    return run


async def run_draft_workflow(user: User, payload: api_models.RunFromSpecRequest) -> api_models.RunResponse:
    run = await _create_run_record(
        user,
        workflow=None,
        spec=payload.spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )
    output = await _execute_compiled_run(run, user, inputs=payload.inputs, config_payload=payload.config)
    run = await _complete_run(run, output)
    return _serialize_run(run)


async def list_workflow_runs(
    user: User,
    workflow_id: str,
    *,
    limit: int = 50,
) -> api_models.WorkflowRunListResponse:
    record = await _get_workflow_record(user, workflow_id)
    limit = max(1, min(limit, 100))
    runs = (
        await WorkflowRun.filter(user=user, workflow=record)
        .order_by("-created_at")
        .limit(limit)
    )
    return api_models.WorkflowRunListResponse(
        workflow_id=record.workflow_id,
        runs=[_serialize_run_summary(run) for run in runs],
    )


def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key in (
        "checkpoint_id",
        "parent_checkpoint_id",
        "values",
        "next",
        "tasks",
        "metadata",
        "created_at",
        "config",
        "parent_config",
    ):
        if hasattr(snapshot, key):
            value = getattr(snapshot, key)
            if value is not None:
                serializable[key] = value
    return serializable


async def get_run_history(user: User, run_id: str) -> api_models.RunHistoryResponse:
    if not shared_config.DATABASE_URL:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer is not configured",
            status=503,
        )
    run = await _get_run(user, run_id)
    # TODO : do we even need to compile the workflow again?
    checkpointer = await get_checkpointer()
    if checkpointer is None:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer failed to initialize",
            status=503,
        )
    try:
        compiled = await compiler.compile(
            user,
            run.spec,
            checkpointer=checkpointer,
        )
    except WorkflowCompilerError as exc:
        _raise_problem(
            type_uri=COMPILE_PROBLEM,
            title="Failed to rebuild workflow for history",
            detail=str(exc),
            status=500,
        )
    graph = compiled.workflow.graph
    config = _build_run_config(run, run.config)
    try:
        history_iter = graph.aget_state_history(config)
        history = [_snapshot_to_dict(item) async for item in history_iter]
    except Exception as exc:  # pragma: no cover - bubble up as HTTP problem
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Failed to load run history",
            detail=str(exc),
            status=500,
        )
    if not history:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run history not found",
            detail=f"No checkpoints found for run '{run.run_id}'",
            status=404,
        )
    return api_models.RunHistoryResponse(run_id=run.run_id, history=history)


async def run_saved_workflow(
    user: User,
    workflow_id: str,
    payload: api_models.RunFromWorkflowRequest,
) -> api_models.RunResponse:
    record = await _get_workflow_record(user, workflow_id)
    if payload.version and payload.version != record.version:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Workflow version mismatch",
            detail="Requested version does not match saved workflow",
            status=409,
        )
    spec = WorkflowSpec.model_validate(record.spec)
    run = await _create_run_record(
        user,
        workflow=record,
        spec=spec,
        inputs=payload.inputs,
        config_payload=payload.config,
    )
    output = await _execute_compiled_run(run, user, inputs=payload.inputs, config_payload=payload.config)
    run = await _complete_run(run, output)
    return _serialize_run(run)


async def _get_run(user: User, run_id: str) -> WorkflowRun:
    try:
        pk = parse_run_public_id(run_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid run id",
            detail="Run id is invalid",
            status=400,
        )
    try:
        return await WorkflowRun.get(id=pk, user=user)
    except DoesNotExist:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run not found",
            detail=f"Run '{run_id}' not found",
            status=404,
        )


async def get_run_status(user: User, run_id: str) -> api_models.RunResponse:
    run = await _get_run(user, run_id)
    return _serialize_run(run)


async def get_run_result(
    user: User,
    run_id: str,
    *,
    include_state: bool = False,
) -> api_models.RunResultResponse:
    run = await _get_run(user, run_id)
    output = run.output or {}
    return api_models.RunResultResponse(
        run_id=run.run_id,
        status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        output=output,
        state=None,
        metrics=run.metrics,
    )


async def list_run_steps(*_: Any, **__: Any) -> None:
    _raise_problem(
        type_uri=RUN_PROBLEM,
        title="Not implemented",
        detail="Run step telemetry is not implemented",
        status=501,
    )


async def cancel_run(*_: Any, **__: Any) -> None:
    _raise_problem(
        type_uri=RUN_PROBLEM,
        title="Not implemented",
        detail="Run cancellation is not implemented",
        status=501,
    )


__all__ = [
    "list_node_types",
    "list_tools",
    "list_models",
    "resolve_schema",
    "validate_spec",
    "compile_spec",
    "create_workflow",
    "list_workflows",
    "get_workflow",
    "update_workflow",
    "delete_workflow",
    "suggest_expression",
    "typecheck_expression",
    "run_draft_workflow",
    "run_saved_workflow",
    "list_workflow_runs",
    "get_run_status",
    "get_run_result",
    "get_run_history",
    "list_run_steps",
    "cancel_run",
]


