"""
MCP discovery tools for searching and listing tools and triggers.

Uses shared discovery logic from seer.tools.discovery_shared.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
from typing import Optional

from seer.mcp.server import mcp
from seer.tools.discovery_shared import (
    search_tools_intent,
    search_triggers_intent,
    list_all_tools,
    list_all_triggers,
    get_available_integrations,
    get_available_providers,
)
from seer.logger import get_logger

logger = get_logger(__name__)


@mcp.tool()
async def search_tools(
    query: str,
    integration_filter: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    Search for available workflow tools using natural language intent.

    Use this to discover tools for workflow automation based on what you want to accomplish.
    For example: "send email", "create draft", "insert database row", "search files".

    Args:
        query: Natural language description of what you want to do (e.g., "send email", "create draft")
        integration_filter: Optional integration to prioritize (e.g., "gmail", "slack", "supabase")
        top_k: Maximum number of results to return (default: 5)

    Returns:
        JSON with top_match (highest confidence tool) and alternatives
    """
    try:
        results = search_tools_intent(
            query=query,
            integration_filter=integration_filter,
            top_k=top_k
        )

        if not results:
            return json.dumps({
                "query": query,
                "top_match": None,
                "alternatives": [],
                "message": f"No tools found for: {query}",
                "available_integrations": get_available_integrations(),
                "suggestion": "Try rephrasing with action verbs (create, send, list, search, etc.)"
            })

        # Format top match
        top_tool = results[0]
        top_match = {
            "tool": top_tool.get("name"),
            "integration": top_tool.get("integration_type", ""),
            "confidence": top_tool.get("confidence_score", 0),
            "description": top_tool.get("description", ""),
            "parameters": top_tool.get("parameters", {}),
        }

        # Format alternatives
        alternatives = []
        for alt_tool in results[1:]:
            desc = alt_tool.get("description", "")
            alternatives.append({
                "tool": alt_tool.get("name"),
                "integration": alt_tool.get("integration_type", ""),
                "confidence": alt_tool.get("confidence_score", 0),
                "description": (desc[:150] + "...") if len(desc) > 150 else desc
            })

        return json.dumps({
            "query": query,
            "top_match": top_match,
            "alternatives": alternatives,
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error searching tools: %s", e)
        return json.dumps({
            "query": query,
            "top_match": None,
            "alternatives": [],
            "error": str(e)
        })


@mcp.tool()
async def list_tools(integration_type: Optional[str] = None) -> str:
    """
    List all available workflow tools from the registry.

    Use this to see what tools are available for workflow automation.
    You can filter by integration type to see tools for a specific service.

    Args:
        integration_type: Optional integration type to filter by (e.g., "gmail", "github", "slack", "supabase")

    Returns:
        JSON with list of all available tools and their metadata
    """
    try:
        result = list_all_tools(integration_type=integration_type)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing tools: %s", e)
        return json.dumps({
            "tools": [],
            "error": str(e)
        })


@mcp.tool()
async def search_triggers(
    query: str,
    provider_filter: Optional[str] = None
) -> str:
    """
    Search for available workflow triggers using keyword matching.

    Use this to discover what events can start a workflow automatically.
    For example: "new email", "database insert", "schedule", "webhook".

    Args:
        query: Search query describing the trigger event (e.g., "gmail new email", "supabase insert", "cron schedule")
        provider_filter: Optional provider name to restrict search (e.g., "gmail", "supabase", "schedule")

    Returns:
        JSON with list of matching triggers, their keys, descriptions, and config schemas
    """
    try:
        results = search_triggers_intent(
            query=query,
            provider_filter=provider_filter,
            top_k=10
        )

        if not results:
            return json.dumps({
                "triggers": [],
                "message": f"No triggers found matching query: {query}",
                "available_providers": get_available_providers(),
                "suggestion": "Try list_triggers() to see all triggers, or search with provider-specific terms"
            })

        return json.dumps({
            "triggers": results,
            "query": query,
            "count": len(results)
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error searching triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })


@mcp.tool()
async def list_triggers(provider: Optional[str] = None) -> str:
    """
    List all available workflow triggers from the registry.

    Use this to see what triggers are available to start workflows automatically.
    Triggers include webhooks, polling events, schedules, and forms.

    Args:
        provider: Optional provider name to filter by (e.g., "gmail", "supabase", "schedule", "form")

    Returns:
        JSON with list of all available triggers grouped by provider
    """
    try:
        result = list_all_triggers(provider_filter=provider)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })
