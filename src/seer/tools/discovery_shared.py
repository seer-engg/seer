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

import numpy as np

from seer.tools.registry import get_tools_by_integration
from seer.core.registry.trigger_registry import trigger_registry
from seer.services.knowledge.embedding_service import get_embedding_service
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
    """Embedding-based tool search via _EmbeddingToolIndex."""
    return await _embedding_index.search_tools(query, top_k=top_k, integration_filter=integration_filter)


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


# ---------------------------------------------------------------------------
# Embedding-based search (replaces keyword matching for production callers)
# ---------------------------------------------------------------------------


class _EmbeddingToolIndex:
    """Lazy-init embedding index for tool and trigger search.

    Uses hybrid scoring: alpha * embedding_similarity + (1-alpha) * tfidf_score.
    TF-IDF catches exact keyword matches that pure embeddings miss.
    """

    # Weight for embeddings vs TF-IDF (0.7 = 70% embeddings, 30% TF-IDF)
    ALPHA = 0.7

    def __init__(self):
        self._tool_embeddings: np.ndarray | None = None
        self._tool_catalog: List[Dict[str, Any]] = []
        self._tool_tfidf: _ToolIndex | None = None
        self._trigger_embeddings: np.ndarray | None = None
        self._trigger_data: List[Dict[str, Any]] = []
        self._trigger_tfidf: _ToolIndex | None = None

    async def _ensure_tools(self):
        if self._tool_embeddings is not None:
            return
        self._tool_catalog = build_tool_catalog()
        texts = [f"{t['name']}: {t.get('description', '')}" for t in self._tool_catalog]
        svc = get_embedding_service()
        vecs = await svc.embed_texts(texts)
        self._tool_embeddings = np.array(vecs)
        # Build TF-IDF index from same catalog
        self._tool_tfidf = _ToolIndex(self._tool_catalog)

    async def _ensure_triggers(self):
        if self._trigger_embeddings is not None:
            return
        all_triggers = trigger_registry.all()
        self._trigger_data = []
        texts = []
        trigger_catalog = []
        for t in all_triggers:
            text = f"{t.key} {t.title}: {t.description or ''}"
            texts.append(text)
            self._trigger_data.append({
                "key": t.key, "title": t.title, "provider": t.provider,
                "mode": t.mode, "description": t.description or f"{t.title} trigger",
                "config_schema": t.schemas.config if t.schemas.config else None,
                "event_schema": t.schemas.event if t.schemas.event else None,
                "sample_event": t.meta.sample_event if t.meta.sample_event else None,
                "requires_connection": t.meta.requires_connection,
            })
            trigger_catalog.append({"name": t.key, "description": f"{t.title} {t.description or ''}"})
        svc = get_embedding_service()
        vecs = await svc.embed_texts(texts)
        self._trigger_embeddings = np.array(vecs)
        self._trigger_tfidf = _ToolIndex(trigger_catalog)

    @staticmethod
    def _merge_scores(embedding_sims: np.ndarray, tfidf_indices: List[int], alpha: float) -> np.ndarray:
        """Merge embedding similarity with TF-IDF scores using weighted combination."""
        n = len(embedding_sims)
        tfidf_scores = np.zeros(n)
        if tfidf_indices:
            # TF-IDF returns ranked indices — assign descending scores
            max_rank = len(tfidf_indices)
            for rank, idx in enumerate(tfidf_indices):
                tfidf_scores[idx] = (max_rank - rank) / max_rank
        # Normalize embedding sims to 0-1
        e_min, e_max = embedding_sims.min(), embedding_sims.max()
        if e_max > e_min:
            normed_emb = (embedding_sims - e_min) / (e_max - e_min)
        else:
            normed_emb = np.zeros_like(embedding_sims)
        return alpha * normed_emb + (1 - alpha) * tfidf_scores

    async def search_tools(self, query: str, top_k: int = 10, integration_filter: str | None = None) -> List[Dict]:
        """Search tools by hybrid embedding + TF-IDF similarity."""
        await self._ensure_tools()
        svc = get_embedding_service()
        q_vec = np.array(await svc.embed_query(query))
        norms = np.linalg.norm(self._tool_embeddings, axis=1) * np.linalg.norm(q_vec)
        emb_sims = (self._tool_embeddings @ q_vec) / np.where(norms == 0, 1, norms)

        # Get TF-IDF scores (request more than top_k to have full ranking)
        tfidf_indices = self._tool_tfidf.search(query, top_k=len(self._tool_catalog)) if self._tool_tfidf else []
        scores = self._merge_scores(emb_sims, tfidf_indices, self.ALPHA)

        indices = np.argsort(scores)[::-1]
        results = []
        for idx in indices:
            tool = {**self._tool_catalog[int(idx)]}
            if integration_filter and tool.get("integration", "") != integration_filter.lower():
                continue
            tool["confidence_score"] = float(scores[idx])
            tool.pop("keywords", None)
            tool.pop("capabilities", None)
            tool.pop("parameters", None)
            tool.pop("resource_pickers", None)
            results.append(tool)
            if len(results) >= top_k:
                break
        return results

    async def search_triggers(self, query: str, top_k: int = 10, provider_filter: str | None = None) -> List[Dict]:
        """Search triggers by hybrid embedding + TF-IDF similarity."""
        await self._ensure_triggers()
        svc = get_embedding_service()
        q_vec = np.array(await svc.embed_query(query))
        norms = np.linalg.norm(self._trigger_embeddings, axis=1) * np.linalg.norm(q_vec)
        emb_sims = (self._trigger_embeddings @ q_vec) / np.where(norms == 0, 1, norms)

        # Get TF-IDF scores
        tfidf_indices = self._trigger_tfidf.search(query, top_k=len(self._trigger_data)) if self._trigger_tfidf else []
        scores = self._merge_scores(emb_sims, tfidf_indices, self.ALPHA)

        indices = np.argsort(scores)[::-1]
        results = []
        for idx in indices:
            trig = {**self._trigger_data[int(idx)]}
            if provider_filter and provider_filter.lower() not in trig["provider"].lower():
                continue
            trig["confidence_score"] = float(scores[idx])
            results.append(trig)
            if len(results) >= top_k:
                break
        return results

    def invalidate(self):
        """Reset cache (for testing or after tool registry changes)."""
        self._tool_embeddings = None
        self._tool_tfidf = None
        self._trigger_embeddings = None
        self._trigger_tfidf = None


_embedding_index = _EmbeddingToolIndex()


async def warm_embedding_index() -> None:
    """Pre-compute tool and trigger embeddings. Call at server startup to avoid first-request latency."""
    # pylint: disable=protected-access # Reason: warming index requires accessing private members
    await _embedding_index._ensure_tools()
    await _embedding_index._ensure_triggers()
    logger.info("Embedding index warmed: %d tools, %d triggers",
                len(_embedding_index._tool_catalog),
                len(_embedding_index._trigger_data))


async def async_search_triggers_intent(
    query: str,
    provider_filter: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Embedding-based trigger search. Async replacement for search_triggers_intent."""
    return await _embedding_index.search_triggers(query, top_k=top_k, provider_filter=provider_filter)
