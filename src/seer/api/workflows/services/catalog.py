"""Registry and catalog endpoints for node types, tools, models, and schemas."""

from __future__ import annotations

from collections import defaultdict

from seer.api.workflows import models as api_models
from seer.api.core.errors import VALIDATION_PROBLEM, raise_problem
from seer.config import config as shared_config
from seer.database import User
from seer.database.models_oauth import OAuthConnection
from seer.core.registry.default_models import DEFAULT_AGENT_MODELS
from seer.core.registry.mcp_client_registry import MCPServerConfig
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton
from seer.services.integrations.auth.helpers import (
    get_connection_display_name,
    has_required_scopes,
    parse_scopes,
)
from seer.services.integrations.auth.oauth import get_oauth_provider

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
                api_models.NodeFieldDescriptor(name="memory_bank_id", kind="memory_bank_select"),
                api_models.NodeFieldDescriptor(name="tools", kind="tools_list"),
                api_models.NodeFieldDescriptor(name="max_iterations", kind="number"),
                api_models.NodeFieldDescriptor(name="enable_artifacts", kind="boolean"),
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
    api_models.ModelDescriptor(
        id=model.id,
        title=model.title,
        supports_json_schema=model.supports_json_schema,
        category=model.category,
    )
    for model in DEFAULT_AGENT_MODELS
]


async def list_node_types() -> api_models.NodeTypeResponse:
    return NODE_TYPE_DESCRIPTORS


async def list_triggers(user: User) -> api_models.TriggerCatalogResponse:
    # Get all connections WITH scopes (not just provider names)
    connections = await OAuthConnection.filter(
        user=user,
        status="active"
    ).all()

    # RCA(wf_70): Group connections by provider to check if ANY connection has
    # the required scopes. The old dict-comprehension only kept the last connection
    # per provider, so is_connected depended on DB query order when a user had
    # multiple connections for the same provider (e.g. two Google accounts).
    provider_to_connections: dict[str, list] = defaultdict(list)
    for conn in connections:
        provider_to_connections[conn.provider].append(conn)

    triggers = []
    for definition in trigger_registry.all():
        # Determine connection status
        if not definition.meta.requires_connection:
            is_connected = True  # No OAuth needed
        else:
            oauth_provider = get_oauth_provider(definition.provider)
            provider_connections = provider_to_connections.get(oauth_provider, [])

            if not provider_connections:
                is_connected = False
            else:
                required_scopes = definition.meta.required_scopes or []
                if required_scopes:
                    # Check if ANY connection for this provider has the required scopes
                    is_connected = any(
                        has_required_scopes(conn.scopes or "", required_scopes)
                        for conn in provider_connections
                    )
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


async def list_mcp_tools(
    payload: api_models.McpToolsRequest,
    *,
    user: User | None = None,
) -> api_models.McpToolsResponse:
    """Fetch available tools from an MCP server for the config panel UI.

    Supports both inline server config and saved config via mcp_server_config_id.
    """
    try:
        server = payload.server
        server_type = payload.server_type
        auth = payload.auth

        # Resolve saved config if mcp_server_config_id is provided
        if payload.mcp_server_config_id and not server:
            if user is None:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="User context required",
                    detail="User context is required when using mcp_server_config_id",
                    status=400,
                )
            from seer.api.integrations.services import resolve_mcp_server_config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
            resolved = await resolve_mcp_server_config(user, payload.mcp_server_config_id)
            server = resolved["server"]
            server_type = resolved.get("server_type", "http")
            auth = resolved.get("auth")

        if not server:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="No server specified",
                detail="Either 'server' or 'mcp_server_config_id' must be provided",
                status=400,
            )

        server_config = MCPServerConfig(
            server=server,
            server_type=server_type,
            auth=auth,
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
    except Exception as exc:  # pylint: disable=broad-exception-caught  # MCP servers can raise arbitrary exceptions
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Failed to fetch MCP tools",
            detail=str(exc),
            status=422,
        )
