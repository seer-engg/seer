"""Utilities for loading and prioritizing MCP tools."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph

from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs
import asyncio
import traceback
from graph_db import NEO4J_GRAPH, TOOL_NODE_LABEL, TOOL_EMBED_PROP, TOOL_VECTOR_INDEX


logger = get_logger("shared.tool_catalog")


DEFAULT_MCP_SERVICES: Sequence[str] = ("asana", "github", "langchain_docs")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_mcp_services(requested_services: Iterable[str]) -> List[str]:
    """Normalize and optionally augment requested services."""

    normalized: List[str] = []
    for service in requested_services or []:
        if not service:
            continue
        normalized_name = service.strip().lower()
        if normalized_name and normalized_name not in normalized:
            normalized.append(normalized_name)

    if not _env_flag("EVAL_AGENT_LOAD_DEFAULT_MCPS", default=True):
        return normalized

    combined: List[str] = list(DEFAULT_MCP_SERVICES)
    for service in normalized:
        if service not in combined:
            combined.append(service)
    return combined


@dataclass
class ToolEntry:
    name: str
    description: str
    service: str


# --- Neo4j + Embeddings setup for semantic tool catalog ---
# We keep this local to avoid coupling shared/ -> agents/*

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


_EMBED_DIMS = 1536  # OpenAI text-embedding-3-small default

_embeddings: OpenAIEmbeddings | None = None



def _get_embeddings() -> OpenAIEmbeddings | None:
    """Lazy-init OpenAIEmbeddings or return None if not configured."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not _OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY missing; falling back to keyword tool selection.")
        return None
    _embeddings = OpenAIEmbeddings(openai_api_key=_OPENAI_API_KEY)
    return _embeddings



def _build_tool_id(entry: "ToolEntry") -> str:
    """Stable identifier for a tool node."""
    # Tools typically include service prefix (e.g., 'asana.create_task'), but be safe.
    return f"{entry.service}::{entry.name}".lower()


async def _sync_tools_to_vector_index(entries: Dict[str, "ToolEntry"]) -> None:
    """
    Upsert MCP tools as nodes with embeddings, ensuring the vector index exists.
    No-op if Neo4j or embeddings are not configured.
    """
    if not entries:
        return

    embedder = _get_embeddings()

    # Prepare documents for embeddings
    # Use name + description to represent the tool semantically.
    ordered_entries: List[ToolEntry] = list(entries.values())
    documents: List[str] = [
        f"{e.name}\n{e.description or ''}".strip() for e in ordered_entries
    ]
    vectors: List[List[float]] = await asyncio.to_thread(embedder.embed_documents, documents)

    rows: List[Dict[str, object]] = []
    for entry, vec in zip(ordered_entries, vectors):
        rows.append(
            {
                "tool_id": _build_tool_id(entry),
                "name": entry.name,
                "service": entry.service,
                "description": entry.description,
                "embedding": vec,
            }
        )

    await asyncio.to_thread(
        NEO4J_GRAPH.query,
        f"""
        UNWIND $rows AS row
        MERGE (t:{TOOL_NODE_LABEL} {{tool_id: row.tool_id}})
        SET t.name = row.name,
            t.service = row.service,
            t.description = row.description,
            t.{TOOL_EMBED_PROP} = row.embedding
        RETURN count(*) AS upserts
        """,
        params={"rows": rows},
    )
    logger.info("Upserted %d tools into Neo4j vector index.", len(rows))


async def _semantic_select_tools(
    entries: Dict[str, "ToolEntry"],
    context: str,
    *,
    max_total: int,
    max_per_service: int,
) -> List[str]:
    """Semantic selection using Neo4j vector index; assumes entries are synced."""
    embedder = _get_embeddings()
    if not embedder:
        return []
    if not context:
        return []

    query_vec: List[float] = await asyncio.to_thread(embedder.embed_query, context)

    # Query top N, then enforce per-service max in Python
    k = max_total * 3  # fetch extra to allow per-service filtering
    results = await asyncio.to_thread(
        NEO4J_GRAPH.query,
        f"""
        CALL db.index.vector.queryNodes("{TOOL_VECTOR_INDEX}", $k, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.service AS service, score
        ORDER BY score DESC
        LIMIT $k
        """,
        params={"embedding": query_vec, "k": k},
    )

    by_service: Dict[str, int] = {}
    prioritized: List[str] = []
    for row in results or []:
        name = row.get("name")
        service = row.get("service") or "misc"
        if not name:
            continue
        # Only consider names present in the current entries payload
        if name.lower() not in entries:
            # entries dict keyed by lower(), preserve original names check
            # We also allow exact match if keys differ in case
            if name not in (e.name for e in entries.values()):
                continue
        used = by_service.get(service, 0)
        if used >= max_per_service:
            continue
        prioritized.append(name)
        by_service[service] = used + 1
        if len(prioritized) >= max_total:
            break
    return prioritized


async def load_tool_entries(service_names: Sequence[str]) -> Dict[str, ToolEntry]:
    """Return lightweight metadata for the requested MCP tools keyed by name."""

    if not service_names:
        return {}

    mcp_client, _ = await get_mcp_client_and_configs(list(service_names))
    tools: List[BaseTool] = await mcp_client.get_tools()
    entries: Dict[str, ToolEntry] = {}
    for tool in tools:
        service = tool.name.split(".", 1)[0] if "." in tool.name else "misc"
        entry = ToolEntry(
            name=tool.name,
            description=getattr(tool, "description", "") or "",
            service=service,
        )
        entries[tool.name.lower()] = entry
    logger.info(
        "Loaded %d MCP tool entries for services: %s",
        len(entries),
        ", ".join(service_names),
    )
    return entries


async def select_relevant_tools(
    entries: Dict[str, ToolEntry],
    context: str,
    *,
    max_total: int = 20,
    max_per_service: int = 5,
) -> List[str]:
    """
    Return tool names most relevant to the context.
    Primary: semantic search via Neo4j vector index.
    Fallback: keyword ranking (previous behavior).
    """

    if not entries:
        return []

    # 1) Try semantic selection
    try:
        # Ensure graph is up to date with current entries
        await _sync_tools_to_vector_index(entries)
        semantic = await _semantic_select_tools(
            entries,
            context,
            max_total=max_total,
            max_per_service=max_per_service,
        )
        if semantic:
            return semantic[:max_total]
    except Exception as exc:
        logger.error(f"Semantic tool selection failed: {traceback.format_exc()}")
        logger.warning("Semantic tool selection failed, falling back to keywords: %s", exc)

    # 2) Fallback to keyword-based selection (original behavior)
    keywords = set(re.findall(r"[a-z0-9_]+", (context or "").lower()))
    service_buckets: Dict[str, List[tuple[int, str]]] = defaultdict(list)
    for entry in entries.values():
        haystack = f"{entry.name} {entry.description}".lower()
        score = sum(1 for kw in keywords if kw and kw in haystack)
        service_buckets[entry.service].append((score, entry.name))

    prioritized: List[str] = []
    for items in service_buckets.values():
        items.sort(key=lambda pair: (-pair[0], pair[1]))
        limited = [name for score, name in items[:max_per_service]]
        prioritized.extend(limited)
        if len(prioritized) >= max_total:
            break
    return prioritized[:max_total]


def canonicalize_tool_name(raw_tool: str, service_hint: str | None = None) -> str:
    """Normalize various tool name spellings into lookup-friendly keys."""

    if not raw_tool:
        return raw_tool

    normalized = raw_tool.strip()
    if not normalized:
        return normalized

    lowered = normalized.lower()
    if lowered.startswith("system."):
        return lowered

    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")
    normalized = normalized.strip("_")
    normalized = normalized.lower()

    if "_" in normalized:
        return normalized

    if service_hint:
        prefix = service_hint.strip().lower()
        if prefix:
            return f"{prefix}_{normalized}"

    return normalized


def build_tool_name_set(entries: Dict[str, ToolEntry]) -> Dict[str, str]:
    """Return canonical tool keys mapped to their original names."""

    return {
        canonicalize_tool_name(entry.name): entry.name
        for entry in entries.values()
    }
