"""
Web search tools package.

Provides web search capability using Exa Search API.
"""
from seer.tools.base import register_tool
from seer.tools.websearch.web_search import WebSearchTool


def register_websearch_tools() -> None:
    """Register all websearch tools."""
    register_tool(WebSearchTool())


__all__ = ["WebSearchTool", "register_websearch_tools"]
