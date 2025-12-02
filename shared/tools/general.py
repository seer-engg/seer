


from langchain.tools import tool
from .mcp_client import CONTEXT7_LIBRARY_TOOL

@tool
async def search_composio_documentation(query: str) -> str:
    """
    Search the Composio documentation for the given query. Use this tool to search for any documentation in composio libraries. e.g 'asana create project', 'github create a pull request'
    Args:
        query: The query to search for.
    Returns:
        The result of the search.
    """
    OWNER = 'websites'
    REPO = 'composio_dev'

    result = await CONTEXT7_LIBRARY_TOOL.ainvoke({
        'context7CompatibleLibraryID':f"{OWNER}/{REPO}",
        'mode': 'code',
        'topic': query
        })
    return result