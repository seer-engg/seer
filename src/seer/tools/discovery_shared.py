"""
Shared discovery logic for tool and trigger search.

Uses TF-IDF weighted scoring with substring matching for semantic-ish
tool discovery. No hardcoded word lists, no external APIs.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from seer.tools.registry import get_tools_by_integration
from seer.core.registry.trigger_registry import trigger_registry
from seer.logger import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into normalized words.

    Handles underscores, camelCase, hyphens, and spaces.
    Returns list (not set) to preserve frequency for TF-IDF.
    """
    if not text:
        return []
    # Split camelCase before lowercasing
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[_\-\s]+', ' ', text.lower())
    return [w for w in re.findall(r'\w+', text) if len(w) > 1]


class _ToolIndex:
    """Pre-computed TF-IDF index over tool name + description corpus."""

    def __init__(self, catalog: List[Dict[str, Any]]) -> None:
        self.catalog = catalog
        self.docs: List[Set[str]] = []
        self.idf: Dict[str, float] = {}
        self._build(catalog)

    def _build(self, catalog: List[Dict[str, Any]]) -> None:
        num_tools = len(catalog)
        if num_tools == 0:
            return

        # Tokenize each tool's name + description
        for entry in catalog:
            tokens = set(tokenize(f"{entry.get('name', '')} {entry.get('description', '')}"))
            self.docs.append(tokens)

        # Compute IDF: log(N / doc_frequency)
        df: Counter[str] = Counter()
        for doc in self.docs:
            for word in doc:
                df[word] += 1
        self.idf = {w: math.log(num_tools / c) + 1.0 for w, c in df.items()}

    def search(self, query: str, top_k: int = 10) -> List[int]:
        """Return indices of top-k matching tools."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.docs:
            return []

        scores: List[float] = []
        for doc_tokens in self.docs:
            score = 0.0
            for qt in query_tokens:
                # Exact match: 2× IDF weight
                if qt in doc_tokens:
                    score += self.idf.get(qt, 1.0) * 2.0
                else:
                    # Substring match: qt is part of a doc token or vice versa
                    for dt in doc_tokens:
                        if qt in dt or dt in qt:
                            score += self.idf.get(dt, 0.5)
                            break
            scores.append(score)

        # Return top-k indices sorted by score descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:top_k] if scores[i] > 0]


def _build_catalog() -> List[Dict[str, Any]]:
    all_tools = get_tools_by_integration()
    return [
        {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "integration_type": t.get("integration_type", ""),
            "parameters": t.get("parameters", {}),
            "required_scopes": t.get("required_scopes", []),
            "resource_pickers": t.get("resource_pickers", {}),
        }
        for t in all_tools
    ]


# Keep for backward compat
def build_tool_catalog() -> List[Dict[str, Any]]:
    """Build tool catalog from the registry."""
    return _build_catalog()


def search_tools_intent(
    query: str,
    integration_filter: Optional[str] = None,
    top_k: int = 10,
    include_internal_fields: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search tools using TF-IDF + substring matching (sync fallback).

    Prefer async_search_tools_intent() which uses semantic embeddings.
    """
    _ = include_internal_fields
    catalog = _build_catalog()

    if integration_filter:
        filt = integration_filter.lower()
        catalog = [t for t in catalog if t.get("integration_type", "").lower() == filt]

    index = _ToolIndex(catalog)
    indices = index.search(query, top_k=top_k)

    results = []
    for i in indices:
        entry = {**catalog[i]}
        entry.pop("parameters", None)
        entry.pop("required_scopes", None)
        entry.pop("resource_pickers", None)
        entry["confidence_score"] = top_k - len(results)
        results.append(entry)

    return results


async def async_search_tools_intent(
    query: str,
    integration_filter: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search tools using semantic embeddings (OpenAI text-embedding-3-small).
    Falls back to TF-IDF if embeddings unavailable.

    Args:
        query: Natural language description of desired capability
        integration_filter: Optional integration to filter by
        top_k: Maximum results to return

    Returns:
        List of matching tools sorted by semantic relevance
    """
    try:
        from seer.tools.semantic_index import get_semantic_index  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

        index = await get_semantic_index()
        results = await index.search(query, top_k=top_k, item_type="tool")

        if integration_filter:
            filt = integration_filter.lower()
            results = [r for r in results if r.get("integration_type", "").lower() == filt]

        # Add confidence_score for backward compat
        for i, r in enumerate(results):
            r["confidence_score"] = len(results) - i

        if results:
            return results

        # Semantic returned nothing — fall through to TF-IDF
        logger.warning("Semantic search returned no results for '%s', falling back to TF-IDF", query)
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: semantic search is non-critical
        logger.exception("Semantic search failed, falling back to TF-IDF")

    return search_tools_intent(query, integration_filter=integration_filter, top_k=top_k)


def search_triggers_intent(
    query: str,
    provider_filter: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Search triggers using TF-IDF + substring matching."""
    all_triggers = trigger_registry.all()

    if provider_filter:
        provider_lower = provider_filter.lower()
        all_triggers = [t for t in all_triggers if provider_lower in t.provider.lower()]

    # Build searchable entries
    entries = []
    for trigger in all_triggers:
        entries.append({
            "key": trigger.key,
            "title": trigger.title,
            "provider": trigger.provider,
            "mode": trigger.mode,
            "description": trigger.description or f"{trigger.title} trigger",
            "config_schema": trigger.schemas.config if trigger.schemas.config else None,
            "event_schema": trigger.schemas.event if trigger.schemas.event else None,
            "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None,
            "requires_connection": trigger.meta.requires_connection,
        })

    # Build index over trigger key + title + description
    trigger_catalog = [
        {"name": e["key"], "description": f"{e['title']} {e['description']}"}
        for e in entries
    ]
    index = _ToolIndex(trigger_catalog)
    indices = index.search(query, top_k=top_k)

    return [entries[i] for i in indices]


def list_all_tools(integration_type: Optional[str] = None) -> Dict[str, Any]:
    """List all available tools, optionally filtered by integration."""
    tools = get_tools_by_integration(integration_type=integration_type)

    tools_list = [
        {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
            "integration_type": t.get("integration_type", ""),
            "required_scopes": t.get("required_scopes", []),
        }
        for t in tools
    ]

    all_tools = get_tools_by_integration()
    all_integrations = sorted(set(
        t.get("integration_type", "") for t in all_tools if t.get("integration_type")
    ))

    return {
        "tools": tools_list,
        "total": len(tools_list),
        "integration_filter": integration_type or "all",
        "available_integrations": all_integrations,
    }


def list_all_triggers(provider_filter: Optional[str] = None) -> Dict[str, Any]:
    """List all available triggers, optionally filtered by provider."""
    all_triggers = trigger_registry.all()

    if provider_filter:
        provider_lower = provider_filter.lower()
        all_triggers = [t for t in all_triggers if provider_lower in t.provider.lower()]

    triggers_list = []
    by_provider: Dict[str, List[Dict[str, Any]]] = {}
    for trigger in all_triggers:
        data = {
            "key": trigger.key,
            "title": trigger.title,
            "provider": trigger.provider,
            "mode": trigger.mode,
            "description": trigger.description or f"{trigger.title} trigger",
            "requires_connection": trigger.meta.requires_connection,
            "event_schema": trigger.schemas.event if trigger.schemas.event else None,
            "sample_event": trigger.meta.sample_event if trigger.meta.sample_event else None,
        }
        triggers_list.append(data)
        by_provider.setdefault(data["provider"], []).append(data)

    return {
        "triggers": triggers_list,
        "by_provider": by_provider,
        "total": len(triggers_list),
        "provider_filter": provider_filter or "all",
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
