"""
Shared tools module - MCP tool management utilities.

This module provides a clean API for working with MCP tools:
- Loading tools from MCP services
- Selecting relevant tools based on context
- Normalizing tool names
- Managing tool metadata
- General-purpose tools (web_search, think)

Public API:
-----------
- load_tool_entries: Load tool metadata from MCP services
- canonicalize_tool_name: Normalize tool names
- resolve_mcp_services: Resolve service names with defaults
- ToolEntry: Tool metadata dataclass
- web_search: Tavily-based web search tool
- think: Thinking/reasoning tool
- LANGCHAIN_MCP_TOOLS: Pre-loaded LangChain docs tools

Example:
--------

    
    # Load tools
    entries = await load_tool_entries(["asana", "github"])
    
    # Select relevant tools for context
"""
import asyncio
from langchain.tools import tool
from tavily import TavilyClient
from shared.logger import get_logger
from shared.config import config
from .composio import ComposioMCPClient

# Public API exports
from shared.tools.loader import (
    resolve_mcp_services,
    DEFAULT_MCP_SERVICES,
)
from shared.tools.registry import ToolEntry
from .mcp_client import LANGCHAIN_DOCS_TOOLS, CONTEXT7_TOOLS
from .general import search_composio_documentation

logger = get_logger("shared.tools")

# ============================================================================
# General-Purpose Tools
# ============================================================================

@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using Tavily API.
    Use this when you need to find current information, documentation, or answers to questions.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string containing search results with titles, snippets, and links
    
    Example:
        result = await web_search("Python asyncio best practices")
    
    Note:
        Requires TAVILY_API_KEY environment variable to be set.
    """
    if not config.tavily_api_key:
        return "Error: TAVILY_API_KEY not configured"
    
    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        # Offload the blocking Tavily client call to a thread
        response = await asyncio.to_thread(client.search, query, max_results=max_results)
        results = response.get("results", [])
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format Tavily results
        formatted_results = f"Search results for: {query}\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No description")
            url = result.get("url", "No link")
            
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   {content}\n"
            formatted_results += f"   URL: {url}\n\n"
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return f"Error performing web search: {e}"


@tool
def think(thought: str) -> str:
    """Use the tool to think about something.
        This is perfect to start your workflow.
        It will not obtain new information or take any actions, but just append the thought to the log and return the result.
        Use it when complex reasoning or some cache memory or a scratchpad is needed.
    
    Args:
        thought: Intermediate reasoning to log
        
    Returns:
        the thought
    """
    return thought


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Loader
    "resolve_mcp_services",
    "DEFAULT_MCP_SERVICES",
    
    # Registry
    "ToolEntry",
    
    # General-purpose tools
    "web_search",
    "think",
    "LANGCHAIN_DOCS_TOOLS",
    "CONTEXT7_TOOLS",
    "search_composio_documentation",
    "ComposioMCPClient",
]


