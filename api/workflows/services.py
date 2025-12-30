from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fastapi import HTTPException
from tortoise.exceptions import DoesNotExist

from shared.database.models import User
from shared.database.workflow_models import (
    WorkflowRecord,
    WorkflowRun,
    WorkflowRunStatus,
    parse_run_public_id,
    parse_workflow_public_id,
)
from shared.tools.base import list_tools as registry_list_tools
from shared.config import config as shared_config
from workflow_compiler.errors import WorkflowCompilerError, ValidationPhaseError
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
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


