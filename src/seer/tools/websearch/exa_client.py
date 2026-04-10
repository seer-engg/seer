"""
Exa Search API client.

Shared helper for web search tools that use the Exa Search REST API.
"""
from __future__ import annotations

from typing import Any, Dict, List

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger("tools.websearch.exa")

EXA_SEARCH_URL = "https://api.exa.ai/search"


def _build_exa_answer(data: Dict[str, Any]) -> str | None:
    """Extract answer from Exa highlights."""
    highlights: List[str] = []
    for item in data.get("results", []):
        item_hl = item.get("highlights", [])
        if item_hl:
            highlights.extend(item_hl)
    return "\n".join(highlights[:3]) if highlights else None


def _format_exa_results(data: Dict[str, Any], include_raw: bool) -> List[Dict[str, Any]]:
    """Normalize Exa results into standard output format."""
    results: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        entry: Dict[str, Any] = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": (
                item.get("highlights", [""])[0]
                if item.get("highlights")
                else item.get("text", "")[:500]
            ),
        }
        if include_raw:
            text = item.get("text")
            if text:
                entry["raw_content"] = text
        results.append(entry)
    return results


async def exa_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """Execute an Exa Search API query and return normalized results.

    Args:
        query: Search query string.
        max_results: Number of results (1-10).
        search_depth: "basic" (type=auto) or "advanced" (type=neural).
        include_answer: Whether to include highlighted snippets.
        include_raw_content: Whether to include full text content.

    Returns:
        Dict with keys: query, search_depth, answer (optional), results, result_count.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from Exa.
    """
    api_key = config.exa_api_key
    if not api_key:
        raise ValueError("Exa API key not configured. Set EXA_API_KEY environment variable.")

    headers = {"Content-Type": "application/json", "x-api-key": api_key}
    search_type = "neural" if search_depth == "advanced" else "auto"

    body: Dict[str, Any] = {
        "query": query,
        "type": search_type,
        "numResults": max(1, min(max_results, 10)),
    }

    contents: Dict[str, Any] = {}
    if include_answer:
        contents["highlights"] = {"maxCharacters": 3000}
    if include_raw_content:
        contents["text"] = {"maxCharacters": 5000}
    if contents:
        body["contents"] = contents

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(EXA_SEARCH_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    result: Dict[str, Any] = {"query": query, "search_depth": search_depth}

    if include_answer:
        answer = _build_exa_answer(data)
        if answer:
            result["answer"] = answer

    results = _format_exa_results(data, include_raw_content)
    result["results"] = results
    result["result_count"] = len(results)

    logger.debug("Exa search completed: query=%s, results=%d", query, len(results))
    return result
