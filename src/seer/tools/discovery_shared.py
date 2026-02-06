"""
Shared discovery logic for tool and trigger search.

Used by both Nexus agent tools (@tool decorator) and MCP tools (@mcp.tool decorator).
This module provides the core algorithms while the consumer modules handle
decorator-specific formatting.
"""

import re
from typing import Any, Dict, List, Optional, Set

from seer.tools.registry import get_tools_by_integration
from seer.core.registry.trigger_registry import trigger_registry
from seer.logger import get_logger

logger = get_logger(__name__)


# Common action verbs used for capability detection
ACTION_VERBS = {
    "create", "send", "list", "get", "update", "delete", "search",
    "find", "read", "write", "insert", "query", "fetch", "post",
    "draft", "compose", "manage", "add", "remove", "modify"
}


def tokenize(text: str) -> Set[str]:
    """
    Tokenize text into normalized keywords.

    Handles underscores, camelCase, hyphens, and spaces.
    Example: "gmail_create_draft" -> {"gmail", "create", "draft"}
    Example: "createDraft" -> {"create", "draft"}

    Args:
        text: Input text to tokenize

    Returns:
        Set of lowercase word tokens
    """
    if not text:
        return set()

    # Split camelCase BEFORE lowercasing: "createDraft" -> "create Draft"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Split on underscores, hyphens, and spaces
    text = re.sub(r'[_\-\s]+', ' ', text.lower())

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\w+', text)

    return set(words)


def build_tool_catalog() -> List[Dict[str, Any]]:
    """
    Build comprehensive tool catalog with searchable keywords.

    Returns enriched tool metadata with:
    - keywords: extracted from name/description/integration
    - capabilities: action verbs found in name
    - integration: normalized integration type

    Returns:
        List of tool entries with enriched metadata
    """
    all_tools = get_tools_by_integration()
    catalog = []

    for tool_meta in all_tools:
        name = tool_meta.get("name", "")
        description = tool_meta.get("description", "")
        integration = tool_meta.get("integration_type", "")

        # Extract keywords from name and description
        name_tokens = tokenize(name)
        desc_tokens = tokenize(description)
        integration_tokens = tokenize(integration)

        # Identify capability keywords (action verbs)
        capabilities = name_tokens.intersection(ACTION_VERBS)

        catalog.append({
            **tool_meta,  # Keep all original metadata
            "keywords": name_tokens | desc_tokens | integration_tokens,
            "capabilities": capabilities,
            "integration": integration.lower() if integration else ""
        })

    return catalog


def score_tool_match(
    tool_data: Dict[str, Any],
    query_tokens: Set[str],
    integration_filter: Optional[str] = None
) -> int:
    """
    Score how well a tool matches the query.

    Scoring weights:
    - Exact keyword match in name: 100 points per token
    - Capability match (action verb): 75 points per token
    - Keyword match: 50 points per token
    - Description substring: 10 points per token
    - Integration match (if filtered): 25 points bonus

    Args:
        tool_data: Tool entry from catalog
        query_tokens: Tokenized query
        integration_filter: Optional integration to prioritize

    Returns:
        Total relevance score
    """
    score = 0
    tool_name_tokens = tokenize(tool_data.get("name", ""))
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


def search_tools_intent(
    query: str,
    integration_filter: Optional[str] = None,
    top_k: int = 5,
    include_internal_fields: bool = False
) -> List[Dict[str, Any]]:
    """
    Search tools using intent-based matching.

    Single algorithm: tokenize query -> score tools -> return top matches.

    Args:
        query: Natural language query (e.g., "create draft", "send email")
        integration_filter: Optional integration to prioritize (e.g., "gmail")
        top_k: Number of results to return
        include_internal_fields: If True, keep keywords/capabilities in results

    Returns:
        List of tools sorted by relevance score, each with confidence_score field
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    catalog = build_tool_catalog()

    # Score all tools
    scored_tools = []
    for tool_entry in catalog:
        score = score_tool_match(tool_entry, query_tokens, integration_filter)
        if score > 0:  # Only include tools with some match
            result = {**tool_entry}
            result["confidence_score"] = score

            # Remove non-serializable fields unless explicitly requested
            if not include_internal_fields:
                result.pop("keywords", None)
                result.pop("capabilities", None)

            scored_tools.append(result)

    # Sort by score descending
    scored_tools.sort(key=lambda t: t["confidence_score"], reverse=True)

    return scored_tools[:top_k]


def search_triggers_intent(
    query: str,
    provider_filter: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search triggers by keyword matching.

    Args:
        query: Search query (e.g., "supabase insert", "new email")
        provider_filter: Optional provider to filter by
        top_k: Maximum results to return

    Returns:
        List of matching triggers with metadata
    """
    all_triggers = trigger_registry.all()

    # Filter by provider if specified
    if provider_filter:
        provider_lower = provider_filter.lower()
        all_triggers = [t for t in all_triggers if provider_lower in t.provider.lower()]

    # Keyword matching against trigger metadata
    query_lower = query.lower()
    matching_triggers = []

    for trigger in all_triggers:
        # Search in: key, title, description, provider, mode
        searchable_text = " ".join([
            trigger.key,
            trigger.title,
            trigger.description or "",
            trigger.provider,
            trigger.mode
        ]).lower()

        if query_lower in searchable_text:
            trigger_data = {
                "key": trigger.key,
                "title": trigger.title,
                "provider": trigger.provider,
                "mode": trigger.mode,
                "description": trigger.description or f"{trigger.title} trigger",
                "config_schema": trigger.schemas.config if trigger.schemas.config else None,
                "event_schema": trigger.schemas.event if trigger.schemas.event else None,
                "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None,
                "requires_connection": trigger.meta.requires_connection
            }
            matching_triggers.append(trigger_data)

    return matching_triggers[:top_k]


def list_all_tools(integration_type: Optional[str] = None) -> Dict[str, Any]:
    """
    List all available tools, optionally filtered by integration.

    Args:
        integration_type: Optional integration to filter by

    Returns:
        Dict with tools list, total count, and available integrations
    """
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

    # Get unique integrations for reference
    all_tools = get_tools_by_integration()
    all_integrations = sorted(set(
        t.get("integration_type", "") for t in all_tools if t.get("integration_type")
    ))

    return {
        "tools": tools_list,
        "total": len(tools_list),
        "integration_filter": integration_type or "all",
        "available_integrations": all_integrations
    }


def list_all_triggers(provider_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    List all available triggers, optionally filtered by provider.

    Args:
        provider_filter: Optional provider to filter by

    Returns:
        Dict with triggers list, grouped by provider, and total count
    """
    all_triggers = trigger_registry.all()

    # Filter by provider if specified
    if provider_filter:
        provider_lower = provider_filter.lower()
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
            "event_schema": trigger.schemas.event if trigger.schemas.event else None,
            "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None
        }
        triggers_list.append(trigger_data)

    # Group by provider for easier reading
    by_provider: Dict[str, List[Dict[str, Any]]] = {}
    for trigger_data in triggers_list:
        prov = trigger_data["provider"]
        if prov not in by_provider:
            by_provider[prov] = []
        by_provider[prov].append(trigger_data)

    return {
        "triggers": triggers_list,
        "by_provider": by_provider,
        "total": len(triggers_list),
        "provider_filter": provider_filter or "all"
    }


def get_available_integrations() -> List[str]:
    """Get sorted list of all available integration types."""
    all_tools = get_tools_by_integration()
    return sorted(set(
        t.get("integration_type", "") for t in all_tools if t.get("integration_type")
    ))


def get_available_providers() -> List[str]:
    """Get sorted list of all available trigger providers."""
    return sorted(set(t.provider for t in trigger_registry.all()))
