from typing import Optional, List
import json
from langchain_core.tools import tool
from shared.tools.registry import get_tools_by_integration
from shared.tool_hub.singleton import get_toolhub_instance
from shared.logger import get_logger

logger = get_logger(__name__)

async def _search_tools_local(
    query: str,
    integration_name: Optional[List[str]] = None,
    top_k: int = 5
) -> List[dict]:
    """
    Search tools from local Chroma vector store using semantic search.
    
    Args:
        query: Search query string
        integration_name: Optional list of integration names to restrict search (e.g., ["github", "asana"])
        top_k: Number of results to return
    
    Returns:
        List of tool dictionaries
    """
    try:
        
        toolhub = get_toolhub_instance()
        if toolhub is None:
            raise ValueError("LocalToolHub not available")
        
        results = await toolhub.query(
            query=query,
            integration_name=integration_name,
            top_k=top_k
        )
        return results
    except Exception as e:
        logger.warning(f"Local tool search not available: {e}")
        # Fallback to registry-based search
        all_tools = get_tools_by_integration()
        # Simple keyword matching fallback
        query_lower = query.lower()
        matching_tools = []
        for tool_meta in all_tools:
            tool_name = tool_meta.get("name", "").lower()
            tool_desc = tool_meta.get("description", "").lower()
            if query_lower in tool_name or query_lower in tool_desc:
                matching_tools.append(tool_meta)
                if len(matching_tools) >= top_k:
                    break
        return matching_tools[:top_k]


@tool
async def search_tools(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Search for available tools/actions using semantic search.
    
    Use this tool when you need to discover what tools are available for a specific capability.
    For example, if the user wants to "search emails" or "find Gmail messages", use this tool
    to discover the relevant Gmail tools.
    
    **QUERY GUIDELINES:**
    - Search for CAPABILITIES, not specific data values
    - Use specific, action-oriented queries
    - GOOD: "search emails", "find Gmail messages", "create Asana task", "list GitHub pull requests"
    - BAD: "Gmail", "GitHub", "search emails with subject 'test'" (includes actual data)
    
    Args:
        query: Search query describing the capability/action needed (e.g., "search emails", "create task")
        reasoning: Optional explanation of why you need this tool and what you're trying to accomplish
        integration_filter: Optional list of integration names to restrict search (e.g., ["gmail", "github"])
    
    Returns:
        JSON string with list of matching tools, their descriptions, and parameters
    """
    try:
        results = await _search_tools_local(
            query=query,
            integration_name=integration_filter,
            top_k=3
        )
        
        if not results:
            return json.dumps({
                "tools": [],
                "message": f"No tools found matching query: {query}. Try a different search term or use list_available_tools to see all tools."
            })
        
        # Format results
        tools_list = []
        for tool_data in results:
            tools_list.append({
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                "parameters": tool_data.get("parameters", {}),
                "integration": tool_data.get("service", tool_data.get("integration_type", ""))
            })
        
        return json.dumps({
            "tools": tools_list,
            "query": query,
            "reasoning": reasoning or "Searching for tools to fulfill user request"
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error searching tools: {e}")
        return json.dumps({
            "tools": [],
            "error": str(e),
            "message": "Tool search failed. Try using list_available_tools to see all available tools."
        })


@tool
async def list_available_tools(integration_type: Optional[str] = None) -> str:
    """
    List all available tools from the registry.
    
    Use this tool when you need to see what tools are available, especially when search_tools
    doesn't return what you need. You can filter by integration type (e.g., "gmail", "github").
    
    Args:
        integration_type: Optional integration type to filter by (e.g., "gmail", "github", "asana")
    
    Returns:
        JSON string with list of all available tools and their metadata
    """
    try:
        tools = get_tools_by_integration(integration_type=integration_type)
        
        tools_list = []
        for tool_meta in tools:
            tools_list.append({
                "name": tool_meta.get("name", ""),
                "description": tool_meta.get("description", ""),
                "parameters": tool_meta.get("parameters", {}),
                "integration_type": tool_meta.get("integration_type", ""),
                "required_scopes": tool_meta.get("required_scopes", [])
            })
        
        return json.dumps({
            "tools": tools_list,
            "total": len(tools_list),
            "integration_filter": integration_type or "all"
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error listing tools: {e}")
        return json.dumps({
            "tools": [],
            "error": str(e)
        })
