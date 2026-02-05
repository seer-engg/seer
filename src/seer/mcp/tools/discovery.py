"""
MCP discovery tools for searching and listing tools and triggers.

Reuses existing discovery logic from seer.agents.nexus.tools.discovery_tools
and seer.tools.registry.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from seer.mcp.server import mcp
from seer.tools.registry import get_tools_by_integration
from seer.core.registry.trigger_registry import trigger_registry
from seer.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> set[str]:
    """
    Tokenize text into normalized keywords.

    Handles underscores, camelCase, and common word variations.
    Example: "gmail_create_draft" -> {"gmail", "create", "draft"}
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


def _build_tool_catalog() -> List[Dict[str, Any]]:
    """Build comprehensive tool catalog with searchable keywords."""
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
            **tool_meta,
            "keywords": name_tokens | desc_tokens | integration_tokens,
            "capabilities": capabilities,
            "integration": integration.lower() if integration else ""
        })

    return catalog


def _score_tool_match(
    tool_data: Dict[str, Any],
    query_tokens: set[str],
    integration_filter: Optional[str] = None
) -> int:
    """Score how well a tool matches the query."""
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
    """Search tools using unified intent-based matching."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    catalog = _build_tool_catalog()

    # Score all tools
    scored_tools = []
    for tool_entry in catalog:
        score = _score_tool_match(tool_entry, query_tokens, integration_filter)
        if score > 0:
            # Remove non-serializable set fields before returning
            result = {k: v for k, v in tool_entry.items() if k not in ("keywords", "capabilities")}
            result["confidence_score"] = score
            scored_tools.append(result)

    # Sort by score descending
    scored_tools.sort(key=lambda t: t["confidence_score"], reverse=True)

    return scored_tools[:top_k]


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
        results = _search_tools_intent(
            query=query,
            integration_filter=integration_filter,
            top_k=top_k
        )

        if not results:
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
            alternatives.append({
                "tool": alt_tool.get("name"),
                "integration": alt_tool.get("integration_type", ""),
                "confidence": alt_tool.get("confidence_score", 0),
                "description": (alt_tool.get("description", "")[:150] + "..."
                               if len(alt_tool.get("description", "")) > 150
                               else alt_tool.get("description", ""))
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
        all_integrations = sorted(set(t["integration_type"] for t in tools_list if t["integration_type"]))

        return json.dumps({
            "tools": tools_list,
            "total": len(tools_list),
            "integration_filter": integration_type or "all",
            "available_integrations": all_integrations
        }, indent=2)

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
                "suggestion": "Try list_triggers() to see all triggers, or search with provider-specific terms"
            })

        # Format results
        triggers_list = []
        for trigger in matching_triggers[:10]:
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
            "count": len(triggers_list)
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

        # Group by provider
        by_provider: Dict[str, List[Dict[str, Any]]] = {}
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

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing triggers: %s", e)
        return json.dumps({
            "triggers": [],
            "error": str(e)
        })
