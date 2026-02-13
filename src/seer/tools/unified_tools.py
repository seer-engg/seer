"""
Unified tool implementations for both Nexus (LangGraph) and MCP (FastMCP) surfaces.

Each tool has ONE canonical async implementation used by both surfaces. Shared
parameters like 'reasoning' have defaults so MCP callers can ignore them while
Nexus agents can populate them for tracing.

All 6 tools are registered via register_unified_tools() which is idempotent
and safe to call from both MCP and Nexus startup paths.
"""
# pylint: disable=duplicate-code  # Reason: Canonical implementations intentionally consolidate MCP + Nexus formatting

from __future__ import annotations

import json
from typing import Optional

from seer.tools.tool_factory import ToolDefinition, ToolSurface, unified_registry
from seer.logger import get_logger

logger = get_logger(__name__)

_REGISTERED = False


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
        search_tools_intent,
        get_available_integrations,
    )

    try:
        results = search_tools_intent(
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
        search_triggers_intent,
        get_available_providers,
    )

    try:
        results = search_triggers_intent(
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
        result = search_templates(query)
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
        result = list_all_templates()
        return json.dumps(result, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing templates: %s", e)
        return json.dumps({
            "templates": [],
            "error": str(e)
        })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_unified_tools() -> None:
    """
    Register all 6 unified tool definitions. Idempotent — safe to call multiple times.

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
