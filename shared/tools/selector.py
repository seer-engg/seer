"""
Tool selection utilities.
Provides semantic and keyword-based tool selection.
"""
import re
import traceback
from collections import defaultdict
from typing import Dict, List

from shared.logger import get_logger
from shared.tools.registry import ToolEntry
from shared.tools.vector_store import sync_tools_to_vector_index, semantic_select_tools
from shared.tools.normalizer import canonicalize_tool_name

logger = get_logger("shared.tools.selector")


async def select_relevant_tools(
    entries: Dict[str, ToolEntry],
    context: str,
    *,
    max_total: int = 20,
    max_per_service: int = 10,
) -> List[str]:
    """
    Return tool names most relevant to the context.
    Primary: semantic search via Neo4j vector index.
    Fallback: keyword ranking.
    
    Args:
        entries: Available tool entries
        context: Context string for selection
        max_total: Maximum total tools to return
        max_per_service: Maximum tools per service
        
    Returns:
        List of tool names prioritized by relevance
    """
    if not entries:
        return []

    # 1) Try semantic selection
    try:
        # Ensure graph is up to date with current entries
        await sync_tools_to_vector_index(entries)
        semantic = await semantic_select_tools(
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

    # 2) Fallback to keyword-based selection
    keywords = set(re.findall(r"[a-z0-9_]+", (context or "").lower()))
    service_buckets: Dict[str, List[tuple[int, str]]] = defaultdict(list)
    for entry in entries.values():
        haystack = f"{entry.name} {entry.description}".lower()
        score = sum(1 for kw in keywords if kw and kw in haystack)
        service_buckets[entry.service].append((score, entry.name))

    prioritized: List[str] = []
    for service, items in service_buckets.items():
        items.sort(key=lambda pair: (-pair[0], pair[1]))
        limited = [
            canonicalize_tool_name(name, service_hint=service)
            for score, name in items[:max_per_service]
        ]
        prioritized.extend(limited)
        if len(prioritized) >= max_total:
            break
    return prioritized[:max_total]

