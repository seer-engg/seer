"""
Web search tool for Nexus agent using Tavily API.

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
    Search the web using Tavily API for current information.

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
    if not config.tavily_api_key:
        return json.dumps({
            "error": "Tavily API key not configured",
            "query": query,
            "suggestion": "Set TAVILY_API_KEY environment variable to enable web search",
        })

    try:
        # Import here to avoid startup dependency if Tavily is not used
        from tavily import TavilyClient  # pylint: disable=import-outside-toplevel # Reason: Optional dependency

        client = TavilyClient(api_key=config.tavily_api_key)

        # Clamp max_results to valid range
        clamped_max_results = max(1, min(max_results, 10))

        # Execute search
        response = client.search(
            query=query,
            max_results=clamped_max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )

        # Format response for agent consumption
        result = {
            "query": query,
            "search_depth": search_depth,
        }

        # Include AI-generated answer if available
        if include_answer and response.get("answer"):
            result["answer"] = response["answer"]

        # Format search results
        results = []
        for item in response.get("results", []):
            formatted_result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            }
            if include_raw_content and item.get("raw_content"):
                formatted_result["raw_content"] = item["raw_content"]
            results.append(formatted_result)

        result["results"] = results
        result["result_count"] = len(results)

        logger.debug("Web search completed: query=%s, results=%d", query, len(results))
        return json.dumps(result, indent=2)

    except ImportError:
        logger.error("Tavily package not installed")
        return json.dumps({
            "error": "Tavily package not installed",
            "query": query,
            "suggestion": "Install tavily-python package: uv add tavily-python",
        })
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error to agent
        logger.exception("Web search failed: %s", e)
        return json.dumps({
            "error": str(e),
            "query": query,
            "suggestion": "Check your Tavily API key and try again",
        })
