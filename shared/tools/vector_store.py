"""
Neo4j vector store operations for tool embeddings.
Handles syncing and querying tool embeddings in Neo4j.
"""
import asyncio
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings

from shared.logger import get_logger
from shared.config import config
from shared.tools.registry import ToolEntry
from shared.tools.normalizer import canonicalize_tool_name

logger = get_logger("shared.tools.vector_store")

# Import graph_db with fallback
from graph_db import NEO4J_GRAPH
GRAPH_AVAILABLE = True


_embeddings: OpenAIEmbeddings | None = None


def _get_embeddings() -> OpenAIEmbeddings | None:
    """Lazy-init OpenAIEmbeddings or return None if not configured."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not config.openai_api_key:
        logger.info("OPENAI_API_KEY missing; vector search unavailable.")
        return None
    _embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    return _embeddings


def _build_tool_id(entry: ToolEntry) -> str:
    """Stable identifier for a tool node."""
    return f"{entry.service}::{entry.name}".lower()


async def sync_tools_to_vector_index(entries: Dict[str, ToolEntry]) -> None:
    """
    Upsert MCP tools as nodes with embeddings in Neo4j.
    No-op if Neo4j or embeddings are not configured.
    
    Args:
        entries: Dict of tool entries to sync
    """
    if not entries or not GRAPH_AVAILABLE or NEO4J_GRAPH is None:
        return

    embedder = _get_embeddings()
    if not embedder:
        return

    # Prepare documents for embeddings
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
        MERGE (t:{config.tool_node_label} {{tool_id: row.tool_id}})
        SET t.name = row.name,
            t.service = row.service,
            t.description = row.description,
            t.{config.tool_embed_prop} = row.embedding
        RETURN count(*) AS upserts
        """,
        params={"rows": rows},
    )
    logger.info("Upserted %d tools into Neo4j vector index.", len(rows))


async def semantic_select_tools(
    entries: Dict[str, ToolEntry],
    context: str,
    *,
    max_total: int,
    max_per_service: int,
) -> List[str]:
    """
    Semantic selection using Neo4j vector index.
    
    Args:
        entries: Available tool entries
        context: Context string for semantic search
        max_total: Maximum total tools to return
        max_per_service: Maximum tools per service
        
    Returns:
        List of tool names prioritized by semantic relevance
    """
    if not GRAPH_AVAILABLE or NEO4J_GRAPH is None:
        return []
    
    embedder = _get_embeddings()
    if not embedder or not context:
        return []

    query_vec: List[float] = await asyncio.to_thread(embedder.embed_query, context)

    # Query top N, then enforce per-service max in Python
    k = max_total * 3  # fetch extra to allow per-service filtering
    results = await asyncio.to_thread(
        NEO4J_GRAPH.query,
        f"""
        CALL db.index.vector.queryNodes("{config.tool_vector_index}", $k, $embedding)
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
            if name not in (e.name for e in entries.values()):
                continue
        used = by_service.get(service, 0)
        if used >= max_per_service:
            continue
        prioritized.append(canonicalize_tool_name(name, service_hint=service))
        by_service[service] = used + 1
        if len(prioritized) >= max_total:
            break
    return prioritized

