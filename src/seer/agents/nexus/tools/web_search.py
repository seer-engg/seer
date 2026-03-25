"""
Web search tool for Nexus agent using Brave Search API.

Provides real-time web search capability to fetch current information
that may not be available in the LLM's training data.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


@tool
async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
) -> str:
    """
    Search the web using Brave Search API for current information.

    Use this tool when you need up-to-date information that may not be in your training data,
    such as recent events, current documentation, or real-time data.

    Args:
        query: The search query describing what information you need.
        max_results: Maximum number of search results to return (1-10, default: 5).
        search_depth: Search depth - "basic" for quick results, "advanced" for more comprehensive search.
        include_answer: Whether to include an AI-generated summary answer (default: True).
        include_raw_content: Whether to include full page content in results (default: False).

    Returns:
        JSON string containing search results with titles, URLs, content snippets,
        and optionally an AI-generated answer summarizing the findings.

    Examples:
        - "latest workflow automation trends 2024"
        - "how to use Supabase edge functions"
        - "Gmail API rate limits"
    """
    if not config.brave_search_api_key:
        return json.dumps({
            "error": "Brave Search API key not configured",
            "query": query,
            "suggestion": "Set BRAVE_SEARCH_API_KEY environment variable to enable web search",
        })

    try:
        from seer.tools.websearch.brave_client import brave_search  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

        result = await brave_search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )
        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error to agent
        logger.exception("Web search failed: %s", e)
        return json.dumps({
            "error": str(e),
            "query": query,
            "suggestion": "Check your Brave Search API key and try again",
        })
