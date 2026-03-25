"""
Brave Search API client.

Shared helper for web search tools that use the Brave Search REST API.
"""
from __future__ import annotations

from typing import Any, Dict, List

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger("tools.websearch.brave")

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """Execute a Brave Search API query and return normalized results.

    Args:
        query: Search query string.
        max_results: Number of results (1-10).
        search_depth: "basic" or "advanced" (advanced enables summary).
        include_answer: Whether to request a summarizer answer.
        include_raw_content: Whether to include extra snippets.

    Returns:
        Dict with keys: query, search_depth, answer (optional), results, result_count.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from Brave.
    """
    api_key = config.brave_search_api_key
    if not api_key:
        raise ValueError("Brave Search API key not configured. Set BRAVE_SEARCH_API_KEY environment variable.")

    clamped = max(1, min(max_results, 10))

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params: Dict[str, Any] = {
        "q": query,
        "count": clamped,
    }
    if search_depth == "advanced" or include_answer:
        params["summary"] = 1

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    result: Dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
    }

    # Extract summarizer answer if available
    if include_answer:
        summarizer = data.get("summarizer", {})
        results_list = summarizer.get("results", [])
        if results_list:
            result["answer"] = results_list[0].get("summary", "")

    # Format web results
    results: List[Dict[str, Any]] = []
    for item in data.get("web", {}).get("results", []):
        formatted: Dict[str, Any] = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("description", ""),
        }
        if include_raw_content:
            extra = item.get("extra_snippets", [])
            if extra:
                formatted["raw_content"] = "\n\n".join(extra)
        results.append(formatted)

    result["results"] = results
    result["result_count"] = len(results)

    logger.debug("Brave search completed: query=%s, results=%d", query, len(results))
    return result
