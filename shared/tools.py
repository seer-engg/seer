from langchain.tools import tool
from shared.logger import get_logger
import os
import asyncio
from tavily import TavilyClient

logger = get_logger("shared.tools")

tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=tavily_api_key)



@tool
async def web_search(query: str, max_results: int ) -> str:
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
    if max_results is None:
        max_results = 5
    try:
        # Try Tavily first (preferred for production)
        
        # Offload the blocking Tavily client call to a thread to avoid blocking the event loop
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
