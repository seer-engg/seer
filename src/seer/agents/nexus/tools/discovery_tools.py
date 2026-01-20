from typing import Optional, List, Dict, Any, Set
import json
import re
from langchain_core.tools import tool
from seer.tools.registry import get_tools_by_integration
from seer.logger import get_logger
from seer.core.registry.trigger_registry import trigger_registry

logger = get_logger(__name__)


def _tokenize(text: str) -> Set[str]:
    """
    Tokenize text into normalized keywords.

    Handles underscores, camelCase, and common word variations.
    Example: "gmail_create_draft" → {"gmail", "create", "draft"}
    """
    if not text:
        return set()

    # Split on underscores, hyphens, and spaces
    text = re.sub(r'[_\-\s]+', ' ', text.lower())

    # Split camelCase: "createDraft" → "create Draft"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\w+', text.lower())

    return set(words)


def _build_tool_catalog() -> List[Dict[str, Any]]:
    """
    Build comprehensive tool catalog with searchable keywords.

    Returns enriched tool metadata with:
    - keywords: extracted from name/description
    - capabilities: action verbs (create, send, list, etc.)
    - integration: normalized integration type
    """
    all_tools = get_tools_by_integration()
    catalog = []

    for tool_meta in all_tools:
        name = tool_meta.get("name", "")
        description = tool_meta.get("description", "")
        integration = tool_meta.get("integration_type", "")

        # Extract keywords from name and description
        name_tokens = _tokenize(name)
        desc_tokens = _tokenize(description)
        integration_tokens = _tokenize(integration)

        # Identify capability keywords (action verbs)
        action_verbs = {"create", "send", "list", "get", "update", "delete", "search",
                       "find", "read", "write", "insert", "query", "fetch", "post",
                       "draft", "compose", "manage", "add", "remove", "modify"}
        capabilities = name_tokens.intersection(action_verbs)

        catalog.append({
            **tool_meta,  # Keep all original metadata
            "keywords": name_tokens | desc_tokens | integration_tokens,
            "capabilities": capabilities,
            "integration": integration.lower() if integration else ""
        })

    return catalog


def _score_tool_match(tool_data: Dict[str, Any], query_tokens: Set[str], integration_filter: Optional[str] = None) -> int:
    """
    Score how well a tool matches the query.

    Scoring:
    - Exact keyword match in name: 100 points
    - Keyword match in keywords: 50 points
    - Capability match: 75 points
    - Integration match (if specified): 25 points
    - Description substring: 10 points per token

    Returns: Total score
    """
    score = 0
    tool_name_tokens = _tokenize(tool_data.get("name", ""))
    tool_keywords = tool_data.get("keywords", set())
    tool_capabilities = tool_data.get("capabilities", set())
    tool_desc = tool_data.get("description", "").lower()

    # Exact name token matches (highest priority)
    name_matches = query_tokens.intersection(tool_name_tokens)
    score += len(name_matches) * 100

    # Capability matches (action verbs)
    capability_matches = query_tokens.intersection(tool_capabilities)
    score += len(capability_matches) * 75

    # Keyword matches
    keyword_matches = query_tokens.intersection(tool_keywords)
    score += len(keyword_matches) * 50

    # Description substring matches
    for token in query_tokens:
        if token in tool_desc:
            score += 10

    # Integration filter bonus
    if integration_filter and tool_data.get("integration", "").lower() == integration_filter.lower():
        score += 25

    return score


def _search_tools_intent(
    query: str,
    integration_filter: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search tools using unified intent-based matching.

    Single algorithm: tokenize query → score tools → return top matches.
    No fallbacks, no vector search, just direct matching.

    Args:
        query: Natural language query (e.g., "create draft", "send email")
        integration_filter: Optional integration to prioritize (e.g., "gmail")
        top_k: Number of results to return

    Returns:
        List of tools sorted by relevance score
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Build catalog (could be cached at startup for performance)
    catalog = _build_tool_catalog()

    # Score all tools
    scored_tools = []
    for tool_entry in catalog:
        score = _score_tool_match(tool_entry, query_tokens, integration_filter)
        if score > 0:  # Only include tools with some match
            scored_tools.append({
                **tool_entry,
                "confidence_score": score
            })

    # Sort by score descending
    scored_tools.sort(key=lambda t: t["confidence_score"], reverse=True)

    return scored_tools[:top_k]


@tool
async def search_tools(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Discover tools based on natural language intent.

    The agent should use this to find tools WITHOUT asking the user for tool names.
    User says "create a draft" → this finds gmail_create_draft automatically.

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

        results = _search_tools_intent(
            query=query,
            integration_filter=integration,
            top_k=5
        )

        if not results:
            # Get available integrations for suggestions
            all_tools = get_tools_by_integration()
            available_integrations = sorted(set(
                t.get("integration_type", "") for t in all_tools if t.get("integration_type")
            ))

            return json.dumps({
                "query": query,
                "top_match": None,
                "alternatives": [],
                "message": f"No tools found for: {query}",
                "available_integrations": available_integrations,
                "suggestion": "Try rephrasing with action verbs (create, send, list, search, etc.)"
            })

        # Format top match with rich details
        top_tool = results[0]
        top_match = {
            "tool": top_tool.get("name"),
            "integration": top_tool.get("integration_type", "").title(),
            "confidence": top_tool.get("confidence_score", 0),
            "description": top_tool.get("description", ""),
            "parameters": top_tool.get("parameters", {})
        }

        # Format alternatives (tools 2-5)
        alternatives = []
        for alt_tool in results[1:5]:
            alternatives.append({
                "tool": alt_tool.get("name"),
                "integration": alt_tool.get("integration_type", "").title(),
                "confidence": alt_tool.get("confidence_score", 0),
                "description": alt_tool.get("description", "")[:100] + "..."  # Truncate for brevity
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
