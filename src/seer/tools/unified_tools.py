# pylint: disable=too-many-lines  # Reason: unified surface for all tool implementations, splitting would break the single-source pattern
"""
Unified tool implementations for both Nexus (LangGraph) and MCP (FastMCP) surfaces.

Each tool has ONE canonical async implementation used by both surfaces. Shared
parameters like 'reasoning' have defaults so MCP callers can ignore them while
Nexus agents can populate them for tracing.

All tools are registered via register_unified_tools() which is idempotent
and safe to call from both MCP and Nexus startup paths.
"""
# pylint: disable=duplicate-code  # Reason: Canonical implementations intentionally consolidate MCP + Nexus formatting

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

from seer.tools.tool_factory import ToolDefinition, ToolSurface, unified_registry
from seer.logger import get_logger

if TYPE_CHECKING:
    from seer.database import User

logger = get_logger(__name__)

_REGISTERED = False


# ---------------------------------------------------------------------------
# User context resolution (supports both MCP and Nexus contexts)
# ---------------------------------------------------------------------------


async def _get_unified_user() -> Optional["User"]:
    """
    Get the current user from either MCP or Nexus context.

    Order of resolution:
    1. MCP authenticated user (from MCPAuthMiddleware context variable)
    2. Nexus thread context (from _current_thread_id context variable)
    3. MCP system user fallback (for stdio transport)

    Returns:
        User object if found, None if no context available
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database import User, init_db
    from tortoise import Tortoise

    # Ensure DB is initialized
    # pylint: disable=protected-access # Reason: Tortoise doesn't expose public init check
    if not Tortoise._inited:
        await init_db()

    # Try MCP authenticated user first
    try:
        from seer.mcp.auth import get_mcp_authenticated_user
        verified_token = get_mcp_authenticated_user()
        if verified_token:
            user, _ = await User.get_or_create(
                user_id=verified_token.user_id,
                defaults={
                    "email": verified_token.email,
                    "first_name": verified_token.first_name,
                    "last_name": verified_token.last_name,
                    "claims": verified_token.claims,
                }
            )
            return user
    except ImportError:
        pass  # MCP auth module not available

    # Try Nexus thread context
    try:
        from seer.agents.nexus.context import _current_thread_id, get_user_for_thread
        thread_id = _current_thread_id.get()
        if thread_id:
            user = await get_user_for_thread(thread_id)
            if user:
                return user
    except ImportError:
        pass  # Nexus context module not available

    # Fallback to system user (for MCP stdio transport without auth)
    system_user = await User.get_or_none(id=1)
    return system_user


# ---------------------------------------------------------------------------
# Canonical implementations (plain async, no decorators)
# ---------------------------------------------------------------------------


async def search_tools_impl(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Search for available workflow tools using natural language intent.

    Use this to discover tools for workflow automation based on what you want to accomplish.
    For example: "send email", "create draft", "insert database row", "search files".

    Args:
        query: Natural language description of what you want to do (e.g., "send email", "create draft")
        reasoning: Why you need this tool (helps with context and tracing)
        integration_filter: Optional integration to prioritize (e.g., "gmail", "slack", "supabase")
        top_k: Maximum number of results to return (default: 5)

    Returns:
        JSON with top_match (highest confidence tool) and alternatives
    """
    from seer.tools.discovery_shared import (  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        async_search_tools_intent,
        get_available_integrations,
    )

    try:
        results = await async_search_tools_intent(
            query=query,
            integration_filter=integration_filter,
            top_k=top_k,
        )

        if not results:
            return json.dumps({
                "query": query,
                "reasoning": reasoning,
                "top_match": None,
                "alternatives": [],
                "message": f"No tools found for: {query}",
                "available_integrations": get_available_integrations(),
                "suggestion": "Try rephrasing with action verbs (create, send, list, search, etc.)"
            })

        # Format top match with rich details including resource_pickers
        top_tool = results[0]
        top_match = {
            "tool": top_tool.get("name"),
            "integration": top_tool.get("integration_type", "").title(),
            "confidence": top_tool.get("confidence_score", 0),
            "description": top_tool.get("description", ""),
            "parameters": top_tool.get("parameters", {}),
            "resource_pickers": top_tool.get("resource_pickers", {}),
        }

        # Format alternatives
        alternatives = []
        for alt_tool in results[1:]:
            desc = alt_tool.get("description", "")
            alternatives.append({
                "tool": alt_tool.get("name"),
                "integration": alt_tool.get("integration_type", "").title(),
                "confidence": alt_tool.get("confidence_score", 0),
                "description": (desc[:150] + "...") if len(desc) > 150 else desc,
            })

        return json.dumps({
            "query": query,
            "reasoning": reasoning,
            "top_match": top_match,
            "alternatives": alternatives,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error searching tools: %s", e)
        return json.dumps({
            "query": query,
            "top_match": None,
            "alternatives": [],
            "error": str(e),
            "message": "Tool search failed. Try using list_tools()."
        })


async def list_tools_impl(integration_type: Optional[str] = None) -> str:
    """
    List all available workflow tools from the registry.

    Use this to see what tools are available for workflow automation.
    You can filter by integration type to see tools for a specific service.

    Args:
        integration_type: Optional integration type to filter by (e.g., "gmail", "github", "slack", "supabase")

    Returns:
        JSON with list of all available tools and their metadata
    """
    from seer.tools.discovery_shared import list_all_tools  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    try:
        result = list_all_tools(integration_type=integration_type)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing tools: %s", e)
        return json.dumps({
            "tools": [],
            "error": str(e)
        })


async def search_triggers_impl(
    query: str,
    reasoning: str = "",
    provider_filter: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """
    Search for available workflow triggers using keyword matching.

    Use this to discover what events can start a workflow automatically.
    For example: "new email", "database insert", "schedule", "webhook".

    Args:
        query: Search query describing the trigger event (e.g., "gmail new email", "supabase insert", "cron schedule")
        reasoning: Why you need this trigger (helps with context and tracing)
        provider_filter: Optional provider name to restrict search (e.g., "gmail", "supabase", "schedule")

    Returns:
        JSON with list of matching triggers, their keys, descriptions, and config schemas
    """
    from seer.tools.discovery_shared import (  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        async_search_triggers_intent,
        get_available_providers,
    )

    try:
        results = await async_search_triggers_intent(
            query=query,
            provider_filter=provider_filter,
            top_k=top_k,
        )

        if not results:
            return json.dumps({
                "triggers": [],
                "reasoning": reasoning,
                "message": f"No triggers found matching query: {query}",
                "available_providers": get_available_providers(),
                "suggestion": "Try list_triggers() to see all triggers, or search with provider-specific terms"
            })

        return json.dumps({
            "triggers": results,
            "reasoning": reasoning,
            "query": query,
            "count": len(results),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error searching triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e),
            "message": "Trigger search failed. Try list_triggers() to see all triggers."
        })


async def list_triggers_impl(provider: Optional[str] = None) -> str:
    """
    List all available workflow triggers from the registry.

    Use this to see what triggers are available to start workflows automatically.
    Triggers include webhooks, polling events, schedules, and forms.

    Args:
        provider: Optional provider name to filter by (e.g., "gmail", "supabase", "schedule", "form")

    Returns:
        JSON with list of all available triggers grouped by provider
    """
    from seer.tools.discovery_shared import list_all_triggers  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    try:
        result = list_all_triggers(provider_filter=provider)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })


async def get_workflow_template_impl(query: str) -> str:
    """
    Retrieve workflow templates by name or tags to use as starting points.

    Use this when you need a pre-built workflow pattern that can be customized.
    Templates include common patterns like "supabase signup to email", "slack notification", etc.

    Args:
        query: Template name or tag to search for (e.g., "supabase gmail", "welcome", "slack notification")

    Returns:
        JSON with matching template(s) including full spec that can be customized
    """
    from seer.tools.template_shared import search_templates  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    try:
        result = await search_templates(query)
        return json.dumps(result, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error retrieving template: %s", e)
        return json.dumps({
            "query": query,
            "matches": [],
            "error": str(e)
        })


async def list_workflow_templates_impl() -> str:
    """
    List all available workflow templates.

    Use this to see all pre-built workflow patterns that can be used as starting points.

    Returns:
        JSON with list of all templates including names, descriptions, and tags
    """
    from seer.tools.template_shared import list_all_templates  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    try:
        result = await list_all_templates()
        return json.dumps(result, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing templates: %s", e)
        return json.dumps({
            "templates": [],
            "error": str(e)
        })


async def get_workflow_schema_impl(focus: str = "basic") -> str:
    """
    Get workflow schema with node examples and edge types.

    Returns compact reference (~2KB) with auto-generated node examples and edge types.
    Use focus="basic" (default) for tool/for_each/hitl/if nodes (covers 80% of workflows).
    Use focus="full" for all node types including agent/browser/image_gen/mcp.

    Args:
        focus: "basic" for common nodes only, "full" for all node types

    Returns:
        Node examples and edge type reference for building valid workflows
    """
    from seer.agents.nexus.schema_context import (  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        generate_node_type_reference,
        generate_edge_reference,
    )

    try:
        full_ref = generate_node_type_reference()
        edge_ref = generate_edge_reference()

        if focus == "basic":
            # Filter to only basic node types
            basic_types = {"### tool", "### for_each", "### hitl", "### if"}
            lines = full_ref.split("\n")
            filtered = []
            include = False
            for line in lines:
                if line.startswith("### "):
                    include = line.lower() in basic_types
                if include:
                    filtered.append(line)
            node_ref = "\n".join(filtered)
        else:
            node_ref = full_ref

        return f"**Node Types**\n{node_ref}\n\n{edge_ref}"

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting workflow schema: %s", e)
        return json.dumps({
            "error": str(e),
            "message": "Failed to retrieve workflow schema"
        })


async def get_tool_accounts_impl(
    tool_name: str,
    reasoning: str = "",
) -> str:
    """
    Get available OAuth accounts for a tool.

    Call this BEFORE building workflow specs with OAuth-based tools
    (gmail_send_email, google_sheets_read, slack_send_message, etc.) to check
    if the user has connected accounts and if account selection is required.

    Args:
        tool_name: The tool name (e.g., "gmail_send_email", "google_sheets_read")
        reasoning: Why you need to check accounts (helps with tracing)

    Returns:
        JSON with:
        - tool_name: The tool name queried
        - provider: The OAuth provider name (e.g., "google", "slack")
        - accounts: List of available accounts with id, display_name, scope status
        - requires_selection: True if user must choose (multiple accounts)

    Usage:
        1. If accounts=[] → Tell user to connect their account first
        2. If requires_selection=false and len(accounts)==1 → Use that account's id as connection_id
        3. If requires_selection=true → Use ask_clarification_questions to let user pick
        4. Include connection_id in tool node ONLY when user selected from multiple accounts
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.tools.base import get_tool
    from seer.services.integrations.auth.helpers import list_connections
    from seer.tools.account_helpers import (
        build_account_entry,
        make_error_response,
        make_no_oauth_response,
    )

    try:
        user = await _get_unified_user()
        if not user:
            return json.dumps(make_error_response("tool_name", tool_name, "User context not available"))

        tool = get_tool(tool_name)
        if tool is None:
            return json.dumps(make_error_response("tool_name", tool_name, f"Tool '{tool_name}' not found"))

        if not tool.required_scopes:
            return json.dumps(make_no_oauth_response(
                "tool_name", tool_name, tool.provider, "This tool does not require OAuth authentication"
            ))

        provider = tool.provider
        if not provider:
            return json.dumps(make_no_oauth_response(
                "tool_name", tool_name, None, "This tool does not have a configured OAuth provider"
            ))

        connections = await list_connections(user)
        provider_connections = [c for c in connections if c.provider == provider]
        accounts = [build_account_entry(conn, tool.required_scopes) for conn in provider_connections]

        return json.dumps({
            "tool_name": tool_name,
            "provider": provider,
            "accounts": accounts,
            "requires_selection": len(accounts) > 1,
            "reasoning": reasoning,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting tool accounts: %s", e)
        return json.dumps(make_error_response("tool_name", tool_name, str(e)))


async def get_trigger_accounts_impl(
    trigger_key: str,
    reasoning: str = "",
) -> str:
    """
    Get available OAuth accounts for a trigger.

    Call this BEFORE building workflow specs with OAuth-based triggers
    (poll.gmail.email_received, poll.googlesheets.row_added, etc.) to check
    if the user has connected accounts and if account selection is required.

    Args:
        trigger_key: The trigger key (e.g., "poll.gmail.email_received")
        reasoning: Why you need to check accounts (helps with tracing)

    Returns:
        JSON with:
        - trigger_key: The trigger key queried
        - provider: The OAuth provider name (e.g., "google", "slack")
        - accounts: List of available accounts with id, display_name, scope status
        - requires_selection: True if user must choose (multiple accounts)

    Usage:
        1. If accounts=[] → Tell user to connect their account first
        2. If requires_selection=false and len(accounts)==1 → System auto-selects (omit provider_connection_id)
        3. If requires_selection=true → Use ask_clarification_questions to let user pick
        4. Include provider_connection_id in trigger spec ONLY when user selected from multiple accounts
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.core.registry.trigger_registry import trigger_registry
    from seer.services.integrations.auth.oauth import get_oauth_provider
    from seer.database.models_oauth import OAuthConnection
    from seer.tools.account_helpers import (
        build_account_entry,
        make_error_response,
        make_no_oauth_response,
    )

    try:
        user = await _get_unified_user()
        if not user:
            return json.dumps(make_error_response("trigger_key", trigger_key, "User context not available"))

        definition = trigger_registry.get(trigger_key)
        if definition is None:
            return json.dumps(make_error_response("trigger_key", trigger_key, f"Trigger '{trigger_key}' not found in registry"))

        if not definition.meta.requires_connection:
            return json.dumps(make_no_oauth_response(
                "trigger_key", trigger_key, definition.provider, "This trigger does not require OAuth authentication"
            ))

        oauth_provider = get_oauth_provider(definition.provider)
        required_scopes = definition.meta.required_scopes or []

        connections = await OAuthConnection.filter(user=user, provider=oauth_provider, status="active").all()
        accounts = [build_account_entry(conn, required_scopes) for conn in connections]

        return json.dumps({
            "trigger_key": trigger_key,
            "provider": definition.provider,
            "accounts": accounts,
            "requires_selection": len(accounts) > 1,
            "reasoning": reasoning,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error getting trigger accounts: %s", e)
        return json.dumps(make_error_response("trigger_key", trigger_key, str(e)))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_unified_tools() -> None:
    """
    Register all unified tool definitions. Idempotent — safe to call multiple times.

    Called from both MCP server startup (_register_tools) and Nexus agent startup
    (get_workflow_tools). The first call registers; subsequent calls are no-ops.
    """
    global _REGISTERED  # pylint: disable=global-statement # Reason: Idempotent guard
    if _REGISTERED:
        return
    _REGISTERED = True

    unified_registry.register(ToolDefinition(
        name="search_tools",
        description=search_tools_impl.__doc__ or "",
        implementation=search_tools_impl,
        surface=ToolSurface.BOTH,
        nexus_name="search_tools",
        mcp_tracking_name="search_tools",
    ))

    unified_registry.register(ToolDefinition(
        name="list_tools",
        description=list_tools_impl.__doc__ or "",
        implementation=list_tools_impl,
        surface=ToolSurface.BOTH,
        nexus_name="list_available_tools",
        mcp_tracking_name="list_tools",
    ))

    unified_registry.register(ToolDefinition(
        name="search_triggers",
        description=search_triggers_impl.__doc__ or "",
        implementation=search_triggers_impl,
        surface=ToolSurface.BOTH,
        nexus_name="search_triggers",
        mcp_tracking_name="search_triggers",
    ))

    unified_registry.register(ToolDefinition(
        name="list_triggers",
        description=list_triggers_impl.__doc__ or "",
        implementation=list_triggers_impl,
        surface=ToolSurface.BOTH,
        nexus_name="list_available_triggers",
        mcp_tracking_name="list_triggers",
    ))

    unified_registry.register(ToolDefinition(
        name="get_workflow_template",
        description=get_workflow_template_impl.__doc__ or "",
        implementation=get_workflow_template_impl,
        surface=ToolSurface.BOTH,
        mcp_tracking_name="get_workflow_template",
    ))

    unified_registry.register(ToolDefinition(
        name="list_workflow_templates",
        description=list_workflow_templates_impl.__doc__ or "",
        implementation=list_workflow_templates_impl,
        surface=ToolSurface.BOTH,
        mcp_tracking_name="list_workflow_templates",
    ))

    unified_registry.register(ToolDefinition(
        name="get_workflow_schema",
        description=get_workflow_schema_impl.__doc__ or "",
        implementation=get_workflow_schema_impl,
        surface=ToolSurface.BOTH,
        mcp_tracking_name="get_workflow_schema",
    ))

    # OAuth account discovery tools
    unified_registry.register(ToolDefinition(
        name="get_tool_accounts",
        description=get_tool_accounts_impl.__doc__ or "",
        implementation=get_tool_accounts_impl,
        surface=ToolSurface.BOTH,
        nexus_name="get_tool_accounts",
        mcp_tracking_name="get_tool_accounts",
    ))

    unified_registry.register(ToolDefinition(
        name="get_trigger_accounts",
        description=get_trigger_accounts_impl.__doc__ or "",
        implementation=get_trigger_accounts_impl,
        surface=ToolSurface.BOTH,
        nexus_name="get_trigger_accounts",
        mcp_tracking_name="get_trigger_accounts",
    ))
