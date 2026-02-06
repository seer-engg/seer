"""
Nexus agent discovery tools for searching and listing tools and triggers.

Uses shared discovery logic from seer.tools.discovery_shared.
These tools are decorated with LangChain's @tool for use in LangGraph agents.
"""

from typing import Optional, List
import json
from langchain_core.tools import tool

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


@tool
async def search_tools(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Discover tools based on natural language intent.

    The agent should use this to find tools WITHOUT asking the user for tool names.
    User says "create a draft" -> this finds gmail_create_draft automatically.

    **IMPORTANT**: Never ask users for tool names - discover them transparently!

    **QUERY GUIDELINES:**
    - Use natural language describing what needs to be done
    - GOOD: "create draft", "send email", "insert row", "list messages"
    - AVOID: tool names like "gmail_create_draft"

    Args:
        query: Natural language action (e.g., "create draft", "send message")
        reasoning: Why you need this tool (helps with context)
        integration_filter: Optional list to prioritize specific integrations (e.g., ["gmail"])

    Returns:
        JSON with top_match (highest confidence tool) and alternatives
    """
    try:
        # Use single integration filter if provided
        integration = integration_filter[0] if integration_filter and len(integration_filter) > 0 else None

        results = search_tools_intent(
            query=query,
            integration_filter=integration,
            top_k=5
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

        # Format top match with rich details including resource_pickers for UI
        top_tool = results[0]
        top_match = {
            "tool": top_tool.get("name"),
            "integration": top_tool.get("integration_type", "").title(),
            "confidence": top_tool.get("confidence_score", 0),
            "description": top_tool.get("description", ""),
            "parameters": top_tool.get("parameters", {}),
            "resource_pickers": top_tool.get("resource_pickers", {})  # Include resource pickers for UI
        }

        # Format alternatives (tools 2-5)
        alternatives = []
        for alt_tool in results[1:5]:
            desc = alt_tool.get("description", "")
            alternatives.append({
                "tool": alt_tool.get("name"),
                "integration": alt_tool.get("integration_type", "").title(),
                "confidence": alt_tool.get("confidence_score", 0),
                "description": (desc[:100] + "...") if len(desc) > 100 else desc
            })

        return json.dumps({
            "query": query,
            "top_match": top_match,
            "alternatives": alternatives,
            "reasoning": reasoning or "Discovering tools for user request"
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error searching tools: %s", e)
        return json.dumps({
            "query": query,
            "top_match": None,
            "alternatives": [],
            "error": str(e),
            "message": "Tool search failed. Try using list_available_tools()."
        })


@tool
async def list_available_tools(integration_type: Optional[str] = None) -> str:
    """
    List all available tools from the registry.

    Use this tool when you need to see what tools are available, especially when search_tools
    doesn't return what you need. You can filter by integration type (e.g., "gmail", "github").

    Args:
        integration_type: Optional integration type to filter by (e.g., "gmail", "github", "asana")

    Returns:
        JSON string with list of all available tools and their metadata
    """
    try:
        result = list_all_tools(integration_type=integration_type)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error listing tools: %s", e)
        return json.dumps({
            "tools": [],
            "error": str(e)
        })


@tool
async def search_triggers(
    query: str,
    reasoning: str = "",
    provider_filter: Optional[str] = None
) -> str:
    """
    Search for available workflow triggers using keyword matching.

    Use this tool when you need to discover what triggers are available for workflow automation.
    For example, if the user wants to trigger on "new Supabase row" or "Gmail email received",
    use this tool to find the appropriate trigger configuration.

    **QUERY GUIDELINES:**
    - Search for trigger events/conditions, not specific data values
    - Use specific, event-oriented queries
    - GOOD: "supabase new row", "gmail new email", "schedule cron", "webhook"
    - BAD: "supabase", "gmail" (too generic)

    Args:
        query: Search query describing the trigger event needed (e.g., "supabase insert", "new email")
        reasoning: Optional explanation of why you need this trigger
        provider_filter: Optional provider name to restrict search (e.g., "gmail", "supabase", "schedule")

    Returns:
        JSON string with list of matching triggers, their keys, descriptions, and config schemas
    """
    try:
        results = search_triggers_intent(
            query=query,
            provider_filter=provider_filter,
            top_k=5
        )

        if not results:
            return json.dumps({
                "triggers": [],
                "message": f"No triggers found matching query: {query}",
                "available_providers": get_available_providers(),
                "suggestion": (
                    "Try list_available_triggers() to see all triggers, or search with "
                    "provider-specific terms like 'supabase', 'gmail', 'schedule'"
                )
            })

        return json.dumps({
            "triggers": results,
            "query": query,
            "reasoning": reasoning or "Searching for triggers to fulfill user request",
            "count": len(results)
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error searching triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e),
            "message": "Trigger search failed. Try list_available_triggers() to see all triggers."
        })


@tool
async def list_available_triggers(provider: Optional[str] = None) -> str:
    """
    List all available workflow triggers from the registry.

    Use this tool when you need to see what triggers are available, especially when
    search_triggers doesn't return what you need. You can filter by provider (e.g., "gmail", "supabase").

    Args:
        provider: Optional provider name to filter by (e.g., "gmail", "supabase", "schedule", "form")

    Returns:
        JSON string with list of all available triggers and their metadata
    """
    try:
        result = list_all_triggers(provider_filter=provider)
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error listing triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })
