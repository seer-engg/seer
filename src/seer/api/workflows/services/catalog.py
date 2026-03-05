# pylint: disable=import-outside-toplevel,broad-exception-caught
# Reason: Catalog service uses lazy imports for LLM generation; broad exception handling for catalog operations
"""Registry and catalog endpoints for node types, tools, models, and schemas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.workflows import models as api_models
from seer.api.workflows.services.shared import (
    _spec_to_dict,
    raise_compiler_error,
)
from seer.api.core.errors import VALIDATION_PROBLEM, raise_problem
from seer.config import config as shared_config
from seer.database import User
from seer.tools.base import list_tools as registry_list_tools
from seer.core.errors import WorkflowCompilerError
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton
from seer.core.schema.models import (
    Node,
    ToolNode,
    WorkflowSpec,
)

COMPILER = WorkflowCompilerSingleton.instance()


NODE_TYPE_DESCRIPTORS = api_models.NodeTypeResponse(
    node_types=[
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
        api_models.NodeTypeDescriptor(
            type="mcp",
            title="MCP Tool",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="server", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="server_type", kind="select", required=True, source="mcp_server_types"),
                api_models.NodeFieldDescriptor(name="tool", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="auth", kind="json"),
                api_models.NodeFieldDescriptor(name="inputs", kind="json"),
                api_models.NodeFieldDescriptor(name="expect_outputs", kind="output_contract"),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="hitl",
            title="Human Input",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="title", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="description", kind="textarea"),
                api_models.NodeFieldDescriptor(name="display", kind="hitl_display"),
                api_models.NodeFieldDescriptor(name="inputs", kind="hitl_inputs"),
                api_models.NodeFieldDescriptor(name="timeout_seconds", kind="number"),
            ],
        ),
        api_models.NodeTypeDescriptor(
            type="agent",
            title="AI Agent",
            fields=[
                api_models.NodeFieldDescriptor(name="id", kind="string", required=True),
                api_models.NodeFieldDescriptor(name="model", kind="model_select", required=True),
                api_models.NodeFieldDescriptor(name="prompt", kind="textarea", required=True),
                api_models.NodeFieldDescriptor(name="tools", kind="tools_list"),
                api_models.NodeFieldDescriptor(name="max_iterations", kind="number"),
                api_models.NodeFieldDescriptor(name="outputs", kind="output_contract"),
            ],
        ),
        # Note: 'task' node type is not supported in the frontend builder UI
        # It's only used internally in workflow specs
    ]
)

DEFAULT_BROWSER_MODEL_REGISTRY = [
    api_models.ModelDescriptor(id="qwen/qwen3-vl-8b-thinking", title="Qwen3 VL 8B", supports_json_schema=False),
    api_models.ModelDescriptor(id="google/gemini-3.1-flash-lite-preview", title="Gemini 3.1 Flash Lite", supports_json_schema=False),
]

DEFAULT_MODEL_REGISTRY = [
    # OpenAI models
    api_models.ModelDescriptor(id="gpt-5-mini", title="GPT-5 Mini", supports_json_schema=True),
    api_models.ModelDescriptor(id="gpt-5", title="GPT-5", supports_json_schema=True),
    api_models.ModelDescriptor(id="z-ai/glm-5", title="GLM 5", supports_json_schema=True),
    api_models.ModelDescriptor(id="moonshotai/kimi-k2.5", title="Kimi K2.5", supports_json_schema=True),
    api_models.ModelDescriptor(id="minimax/minimax-m2.5", title="MiniMax M2.5", supports_json_schema=True),
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


async def list_triggers(user: User) -> api_models.TriggerCatalogResponse:
    from seer.database.models_oauth import OAuthConnection
    from seer.services.integrations.auth.oauth import get_oauth_provider
    from seer.services.integrations.auth.helpers import has_required_scopes

    # Get all connections WITH scopes (not just provider names)
    connections = await OAuthConnection.filter(
        user=user,
        status="active"
    ).all()

    provider_to_scopes = {
        conn.provider: conn.scopes or ""
        for conn in connections
    }

    triggers = []
    for definition in trigger_registry.all():
        # Determine connection status
        if not definition.meta.requires_connection:
            is_connected = True  # No OAuth needed
        else:
            oauth_provider = get_oauth_provider(definition.provider)

            if oauth_provider not in provider_to_scopes:
                is_connected = False
            else:
                # Check if connection has required scopes
                granted_scopes = provider_to_scopes[oauth_provider]
                required_scopes = definition.meta.required_scopes or []

                if required_scopes:
                    is_connected = has_required_scopes(granted_scopes, required_scopes)
                else:
                    is_connected = True  # No scope requirements

        triggers.append(
            api_models.TriggerDescriptor(
                key=definition.key,
                title=definition.title,
                provider=definition.provider,
                mode=definition.mode,
                description=definition.description,
                event_schema=definition.schemas.event,
                filter_schema=definition.schemas.filter,
                config_schema=definition.schemas.config,
                is_connected=is_connected,
            )
        )
    return api_models.TriggerCatalogResponse(triggers=triggers)


async def get_trigger_accounts(user: User, trigger_key: str) -> api_models.TriggerAccountsResponse:
    """
    Get available OAuth accounts for a specific trigger.

    Used by frontend to let users select which account to use
    when they have multiple accounts for the same provider.

    Similar to get_tool_accounts but for triggers.
    """
    from seer.database.models_oauth import OAuthConnection
    from seer.services.integrations.auth.helpers import (
        get_connection_display_name,
        has_required_scopes,
        parse_scopes,
    )
    from seer.services.integrations.auth.oauth import get_oauth_provider

    definition = trigger_registry.get(trigger_key)
    if definition is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Trigger not found",
            detail=f"Trigger '{trigger_key}' not found in registry",
            status=404,
        )

    # If trigger doesn't require OAuth, return empty response
    if not definition.meta.requires_connection:
        return api_models.TriggerAccountsResponse(
            trigger_key=trigger_key,
            provider=definition.provider,
            accounts=[],
            requires_selection=False,
        )

    oauth_provider = get_oauth_provider(definition.provider)
    required_scopes = definition.meta.required_scopes or []

    # Get all connections for this provider
    connections = await OAuthConnection.filter(
        user=user,
        provider=oauth_provider,
        status="active"
    ).all()

    accounts = []
    for conn in connections:
        # Check scope coverage
        missing = []
        if conn.scopes:
            conn_has_scopes = has_required_scopes(conn.scopes, required_scopes)
            if not conn_has_scopes:
                granted_set = parse_scopes(conn.scopes)
                for scope in required_scopes:
                    if scope not in granted_set:
                        missing.append(scope)
        else:
            conn_has_scopes = not required_scopes  # True if no scopes required
            missing = list(required_scopes)

        accounts.append(api_models.TriggerAccountInfo(
            id=conn.id,
            provider_account_id=conn.provider_account_id or f"ID:{conn.id}",
            display_name=get_connection_display_name(conn),
            has_required_scopes=conn_has_scopes,
            missing_scopes=missing,
        ))

    return api_models.TriggerAccountsResponse(
        trigger_key=trigger_key,
        provider=definition.provider,
        accounts=accounts,
        requires_selection=len(accounts) > 1,
    )


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


async def list_browser_models() -> api_models.ModelRegistryResponse:
    return api_models.ModelRegistryResponse(models=list(DEFAULT_BROWSER_MODEL_REGISTRY))


async def resolve_schema(schema_id: str) -> api_models.SchemaResponse:
    schema = COMPILER.schema_registry.get(schema_id)
    if schema is None:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Schema not found",
            detail=f"Schema '{schema_id}' is not registered",
            status=404,
        )
    return api_models.SchemaResponse(id=schema_id, json_schema=schema)


async def generate_schema_metadata(
    payload: api_models.SchemaMetadataGenerateRequest
) -> api_models.SchemaMetadataGenerateResponse:
    """
    Generate schema title and description using LLM.

    Analyzes JSON Schema structure to produce:
    - Concise PascalCase title (2-4 words)
    - Clear description (1-2 sentences)

    Args:
        payload: Schema to analyze

    Returns:
        Generated title and description
    """
    import json

    from langchain_core.messages import HumanMessage, SystemMessage

    from seer.llm import get_llm

    # Extract field information
    schema = payload.json_schema
    if not schema.get("properties"):
        return api_models.SchemaMetadataGenerateResponse(
            title="OutputSchema",
            description="Structured output schema"
        )

    # Build field summary
    fields_info = []
    for field_name, field_def in schema.get("properties", {}).items():
        fields_info.append({
            "name": field_name,
            "type": field_def.get("type", "any"),
            "description": field_def.get("description", "")
        })

    # System prompt
    system_prompt = """You are a technical writer helping create clear schema metadata.

Given a JSON Schema's field definitions, generate:
1. A concise title in PascalCase (2-4 words) that captures the schema's purpose
2. A clear description (1-2 sentences) explaining what this schema represents

Guidelines:
- Title should be specific and descriptive (e.g., "UserProfile", "PaymentDetails", "SearchResults")
- Description should explain the purpose and key information the schema contains
- Keep both professional and concise
- Don't include phrases like "This schema" or "A schema for" in the description

Respond with JSON in this exact format:
{
  "title": "YourTitle",
  "description": "Your description here."
}"""

    user_prompt = f"""Schema fields:
{json.dumps(fields_info, indent=2)}

Generate appropriate title and description:"""

    try:
        llm = get_llm(model="moonshotai/kimi-k2.5", temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = await llm.ainvoke(messages)
        result = json.loads(response.content)

        return api_models.SchemaMetadataGenerateResponse(
            title=result.get("title", "OutputSchema"),
            description=result.get("description", "Structured output schema")
        )

    except Exception as e:
        from seer.logger import get_logger
        logger = get_logger("api.workflows.schema_generation")
        logger.error("Schema metadata generation failed: %s", str(e), exc_info=True)

        return api_models.SchemaMetadataGenerateResponse(
            title="OutputSchema",
            description="Structured output schema"
        )


async def list_mcp_tools(payload: api_models.McpToolsRequest) -> api_models.McpToolsResponse:
    """Fetch available tools from an MCP server for the config panel UI."""
    from seer.core.registry.mcp_client_registry import MCPServerConfig

    try:
        server_config = MCPServerConfig(
            server=payload.server,
            server_type=payload.server_type,
            auth=payload.auth,
        )
        tools = await COMPILER.mcp_client_registry.list_tools(server_config)
        return api_models.McpToolsResponse(
            tools=[
                api_models.McpToolDescriptor(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema,
                )
                for t in tools
            ]
        )
    except (OSError, RuntimeError, ConnectionError) as exc:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="MCP server connection failed",
            detail=str(exc),
            status=422,
        )
    except Exception as exc:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Failed to fetch MCP tools",
            detail=str(exc),
            status=422,
        )


def _collect_warnings_from_nodes(nodes: Iterable[Node]) -> List[api_models.WorkflowWarning]:
    warnings: List[api_models.WorkflowWarning] = []
    for node in nodes:
        if isinstance(node, ToolNode) and not node.out:
            warnings.append(
                api_models.WorkflowWarning(
                    code="OUT_MISSING",
                    node_id=node.id,
                    message=f"Node '{node.id}' has no 'out'; downstream references may fail.",
                )
            )
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
        raise_compiler_error(exc, "Compilation failed")

    warnings = _collect_warnings_from_nodes(spec.nodes)
    artifacts = api_models.CompileArtifacts()
    if payload.options.emit_type_env:
        artifacts.type_env = dict(compiled.workflow.type_env)
    if payload.options.emit_graph_preview:
        artifacts.graph_preview = _graph_preview(spec)

    return api_models.CompileResponse(warnings=warnings, artifacts=artifacts)
