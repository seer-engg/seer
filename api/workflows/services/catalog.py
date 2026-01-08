"""Registry and catalog endpoints for node types, tools, models, and schemas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from api.workflows import models as api_models
from api.workflows.services.shared import (
    COMPILE_PROBLEM,
    VALIDATION_PROBLEM,
    _raise_problem,
    _spec_to_dict,
)
from api.agents.checkpointer import get_checkpointer
from shared.config import config as shared_config
from shared.database.models import User
from shared.tools.base import list_tools as registry_list_tools
from workflow_compiler.errors import WorkflowCompilerError
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.registry.trigger_registry import trigger_registry
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)

COMPILER = WorkflowCompilerSingleton.instance()


NODE_TYPE_DESCRIPTORS = api_models.NodeTypeResponse(
    node_types=[
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
            type="if_else",
            title="If/Else",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="condition", kind="expression", required=True),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="for_loop",
            title="For Each",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="items", kind="expression", required=True),
                api_models.NodeFieldDescriptor(name="item_var", kind="string"),
                api_models.NodeFieldDescriptor(name="index_var", kind="string"),
                api_models.NodeFieldDescriptor(name="out", kind="string"),
            ],
        ),
        # Note: 'task' node type is not supported in the frontend builder UI
        # It's only used internally in workflow specs
    ]
)

DEFAULT_MODEL_REGISTRY = [
    api_models.ModelDescriptor(id="gpt-4.1-mini", title="GPT-4.1 mini", supports_json_schema=True),
    api_models.ModelDescriptor(id="gpt-4o-mini", title="GPT-4o mini", supports_json_schema=True),
]



async def list_node_types() -> api_models.NodeTypeResponse:
    return NODE_TYPE_DESCRIPTORS


async def list_tools(include_schemas: bool = False) -> api_models.ToolRegistryResponse:
    tools: List[api_models.ToolDescriptor] = []
    for tool in registry_list_tools():
        definition = COMPILER.ensure_tool(tool.name)
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



async def resolve_schema(schema_id: str) -> api_models.SchemaResponse:
    schema = COMPILER.schema_registry.get(schema_id)
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



def _graph_preview(spec: WorkflowSpec) -> Dict[str, Any]:
    nodes = [{"id": node.id, "kind": node.type} for node in spec.nodes]
    edges = []
    for idx in range(len(spec.nodes) - 1):
        edges.append({"from": spec.nodes[idx].id, "to": spec.nodes[idx + 1].id})
    return {"nodes": nodes, "edges": edges}


def validate_spec(payload: api_models.ValidateRequest) -> api_models.ValidateResponse:
    spec = payload.spec
    warnings = _collect_warnings_from_nodes(spec.nodes)
    return api_models.ValidateResponse(warnings=warnings)

async def compile_spec(user: User, payload: api_models.CompileRequest) -> api_models.CompileResponse:
    spec = payload.spec
    spec_dict = _spec_to_dict(spec)
    checkpointer = await get_checkpointer()
    try:
        compiled = await COMPILER.compile(user, spec_dict, checkpointer=checkpointer)
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
