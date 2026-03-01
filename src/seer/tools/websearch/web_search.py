"""
Web search tool using Tavily API.

Provides real-time web search capability for agent nodes to fetch
current information that may not be available in the LLM's training data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from fastapi import HTTPException

from seer.config import config
from seer.logger import get_logger
from seer.tools.base import BaseTool

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.websearch")


class WebSearchTool(BaseTool):
    """Search the web using Tavily API for current information."""

    name = "web_search"
    description = (
        "Search the web for current information using Tavily API. "
        "Use this when you need up-to-date information that may not be in your training data, "
        "such as recent events, current documentation, or real-time data."
    )
    integration_type = "websearch"
    required_scopes: List[str] = []  # No OAuth required - uses API key from config

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what information you need.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (1-10).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": 'Search depth - "basic" for quick results, "advanced" for more comprehensive search.',
                    "default": "basic",
                },
                "include_answer": {
                    "type": "boolean",
                    "description": "Whether to include an AI-generated summary answer.",
                    "default": True,
                },
                "include_raw_content": {
                    "type": "boolean",
                    "description": "Whether to include full page content in results.",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The original search query"},
                "search_depth": {"type": "string", "description": "The search depth used"},
                "answer": {"type": "string", "description": "AI-generated summary answer (if requested)"},
                "results": {
                    "type": "array",
                    "description": "Search results",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Result title"},
                            "url": {"type": "string", "description": "Result URL"},
                            "content": {"type": "string", "description": "Content snippet"},
                            "score": {"type": "number", "description": "Relevance score"},
                            "raw_content": {"type": "string", "description": "Full page content (if requested)"},
                        },
                    },
                },
                "result_count": {"type": "integer", "description": "Number of results returned"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        # access_token, credentials, context unused - this tool uses API key from config
        _ = access_token, credentials, context

        query: str = arguments["query"]
        max_results: int = arguments.get("max_results", 5)
        search_depth: Literal["basic", "advanced"] = arguments.get("search_depth", "basic")
        include_answer: bool = arguments.get("include_answer", True)
        include_raw_content: bool = arguments.get("include_raw_content", False)

        # Validate API key is configured
        if not config.tavily_api_key:
            raise HTTPException(
                status_code=503,
                detail="Web search is not available: Tavily API key not configured. "
                "Set TAVILY_API_KEY environment variable to enable web search.",
            )

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

            # Format response
            result: Dict[str, Any] = {
                "query": query,
                "search_depth": search_depth,
            }

            # Include AI-generated answer if available
            if include_answer and response.get("answer"):
                result["answer"] = response["answer"]

            # Format search results
            results = []
            for item in response.get("results", []):
                formatted_result: Dict[str, Any] = {
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
            return result

        except ImportError as e:
            logger.error("Tavily package not installed")
            raise HTTPException(
                status_code=503,
                detail="Web search is not available: Tavily package not installed. "
                "Install with: uv add tavily-python",
            ) from e
        except Exception as e:
            logger.exception("Web search failed: %s", e)
            raise HTTPException(
                status_code=502,
                detail=f"Web search failed: {e}. Check your Tavily API key and try again.",
            ) from e


__all__ = ["WebSearchTool"]
