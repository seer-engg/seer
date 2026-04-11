"""
Web search tool using Exa Search API.

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
    """Search the web using Exa Search API for current information."""

    name = "web_search"
    description = (
        "Search the web for current information using Exa Search API. "
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

        if not config.exa_api_key:
            raise HTTPException(
                status_code=503,
                detail="Web search is not available: Exa API key not configured. "
                "Set EXA_API_KEY environment variable to enable web search.",
            )

        try:
            from seer.tools.websearch.exa_client import exa_search  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

            return await exa_search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )
        except Exception as e:
            logger.exception("Web search failed: %s", e)
            raise HTTPException(
                status_code=502,
                detail=f"Web search failed: {e}. Check your Exa API key and try again.",
            ) from e


__all__ = ["WebSearchTool"]
