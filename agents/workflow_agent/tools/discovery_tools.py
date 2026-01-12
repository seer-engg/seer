from typing import Optional, List
import json
from langchain_core.tools import tool
from shared.tools.registry import get_tools_by_integration
from shared.tool_hub.singleton import get_toolhub_instance
from shared.logger import get_logger
from workflow_compiler.registry.trigger_registry import trigger_registry

logger = get_logger(__name__)


async def _search_tools_local(
    query: str,
    integration_name: Optional[List[str]] = None,
    top_k: int = 5
) -> List[dict]:
    """
    Search tools from local Chroma vector store using semantic search.

    Args:
        query: Search query string
        integration_name: Optional list of integration names to restrict search (e.g., ["github", "asana"])
        top_k: Number of results to return

    Returns:
        List of tool dictionaries
    """
    try:

        toolhub = get_toolhub_instance()
        if toolhub is None:
            raise ValueError("LocalToolHub not available")

        results = await toolhub.query(
            query=query,
            integration_name=integration_name,
            top_k=top_k
        )
        return results
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Fallback on any toolhub failure
        logger.warning("Local tool search not available: %s", e)
        # Enhanced fallback: try integration-based search first, then keyword matching
        matching_tools = []

        # If integration_name is provided, filter by integration first
        if integration_name and len(integration_name) > 0:
            for integration in integration_name:
                tools_for_integration = get_tools_by_integration(integration_type=integration)
                matching_tools.extend(tools_for_integration)

        # If no integration filter or integration search returned nothing, do keyword matching
        if not matching_tools:
            all_tools = get_tools_by_integration()
            query_lower = query.lower()
            for tool_meta in all_tools:
                tool_name = tool_meta.get("name", "").lower()
                tool_desc = tool_meta.get("description", "").lower()
                integration_type = tool_meta.get("integration_type", "").lower()

                # Match against name, description, or integration type
                if (query_lower in tool_name or
                    query_lower in tool_desc or
                    query_lower in integration_type):
                    matching_tools.append(tool_meta)

        return matching_tools[:top_k]


@tool
async def search_tools(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Search for available tools/actions using semantic search.

    Use this tool when you need to discover what tools are available for a specific capability.
    For example, if the user wants to "search emails" or "find Gmail messages", use this tool
    to discover the relevant Gmail tools.

    **QUERY GUIDELINES:**
    - Search for CAPABILITIES, not specific data values
    - Use specific, action-oriented queries
    - GOOD: "search emails", "find Gmail messages", "create Asana task", "list GitHub pull requests"
    - BAD: "Gmail", "GitHub", "search emails with subject 'test'" (includes actual data)

    Args:
        query: Search query describing the capability/action needed (e.g., "search emails", "create task")
        reasoning: Optional explanation of why you need this tool and what you're trying to accomplish
        integration_filter: Optional list of integration names to restrict search (e.g., ["gmail", "github"])

    Returns:
        JSON string with list of matching tools, their descriptions, and parameters
    """
    try:
        results = await _search_tools_local(
            query=query,
            integration_name=integration_filter,
            top_k=3
        )

        if not results:
            # Get list of available integrations for suggestions
            all_tools = get_tools_by_integration()
            available_integrations = sorted(set(t.get("integration_type", "") for t in all_tools if t.get("integration_type")))

            return json.dumps({
                "tools": [],
                "message": f"No tools found matching query: {query}",
                "available_integrations": available_integrations,
                "suggestions": [
                    "Try list_available_tools() to see all tools",
                    "Try list_available_tools(integration_type='<integration>') to filter by integration",
                    "Use more specific search terms describing the action (e.g., 'create draft', 'search messages')"
                ]
            })

        # Format results
        tools_list = []
        for tool_data in results:
            tools_list.append({
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                "parameters": tool_data.get("parameters", {}),
                "integration": tool_data.get("service", tool_data.get("integration_type", ""))
            })

        return json.dumps({
            "tools": tools_list,
            "query": query,
            "reasoning": reasoning or "Searching for tools to fulfill user request"
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error searching tools: %s", e)
        return json.dumps({
            "tools": [],
            "error": str(e),
            "message": "Tool search failed. Try using list_available_tools to see all available tools."
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
        tools = get_tools_by_integration(integration_type=integration_type)

        tools_list = []
        for tool_meta in tools:
            tools_list.append({
                "name": tool_meta.get("name", ""),
                "description": tool_meta.get("description", ""),
                "parameters": tool_meta.get("parameters", {}),
                "integration_type": tool_meta.get("integration_type", ""),
                "required_scopes": tool_meta.get("required_scopes", [])
            })

        return json.dumps({
            "tools": tools_list,
            "total": len(tools_list),
            "integration_filter": integration_type or "all"
        }, indent=2)

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
        all_triggers = trigger_registry.all()

        # Filter by provider if specified
        if provider_filter:
            provider_lower = provider_filter.lower()
            all_triggers = [t for t in all_triggers if provider_lower in t.provider.lower()]

        # Keyword matching against trigger metadata
        query_lower = query.lower()
        matching_triggers = []

        for trigger in all_triggers:
            # Search in: key, title, description, provider
            searchable_text = " ".join([
                trigger.key,
                trigger.title,
                trigger.description or "",
                trigger.provider,
                trigger.mode
            ]).lower()

            if query_lower in searchable_text:
                matching_triggers.append(trigger)

        if not matching_triggers:
            available_providers = sorted(set(t.provider for t in trigger_registry.all()))
            return json.dumps({
                "triggers": [],
                "message": f"No triggers found matching query: {query}",
                "available_providers": available_providers,
                "suggestion": (
                    "Try list_available_triggers() to see all triggers, or search with "
                    "provider-specific terms like 'supabase', 'gmail', 'schedule'"
                )
            })

        # Format results with key metadata
        triggers_list = []
        for trigger in matching_triggers[:5]:  # Limit to top 5
            trigger_data = {
                "key": trigger.key,
                "title": trigger.title,
                "provider": trigger.provider,
                "mode": trigger.mode,
                "description": trigger.description or f"{trigger.title} trigger",
                "config_schema": trigger.schemas.config if trigger.schemas.config else None,
                "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None,
                "requires_connection": trigger.meta.requires_connection
            }
            triggers_list.append(trigger_data)

        return json.dumps({
            "triggers": triggers_list,
            "query": query,
            "reasoning": reasoning or "Searching for triggers to fulfill user request",
            "count": len(triggers_list)
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
        all_triggers = trigger_registry.all()

        # Filter by provider if specified
        if provider:
            provider_lower = provider.lower()
            all_triggers = [t for t in all_triggers if provider_lower in t.provider.lower()]

        triggers_list = []
        for trigger in all_triggers:
            trigger_data = {
                "key": trigger.key,
                "title": trigger.title,
                "provider": trigger.provider,
                "mode": trigger.mode,
                "description": trigger.description or f"{trigger.title} trigger",
                "requires_connection": trigger.meta.requires_connection,
                "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None
            }
            triggers_list.append(trigger_data)

        # Group by provider for easier reading
        by_provider = {}
        for trigger_data in triggers_list:
            prov = trigger_data["provider"]
            if prov not in by_provider:
                by_provider[prov] = []
            by_provider[prov].append(trigger_data)

        return json.dumps({
            "triggers": triggers_list,
            "by_provider": by_provider,
            "total": len(triggers_list),
            "provider_filter": provider or "all"
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error listing triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })
