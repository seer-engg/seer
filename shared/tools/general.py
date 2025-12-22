from langchain.tools import tool
from .mcp_client import CONTEXT7_LIBRARY_TOOL


async def get_context7_results(owner: str, repo: str, mode: str = 'code', topic: str = None) -> str:
    """
    Tool to get the results from the Context7 library for the given owner, repo, mode, and topic.
    Args:
        owner: The owner of the repository.
        repo: The name of the repository.
        mode: The mode of the search.
        topic: The topic of the search.
    """

    result = await CONTEXT7_LIBRARY_TOOL.ainvoke({
        'context7CompatibleLibraryID':f"{owner}/{repo}",
        'mode': mode,
        'topic': topic
        })
    return result

@tool
async def search_langchain_documentation(keywords: list[str]) -> str:
    """
    Tool to search the Langchain documentation for the given keywords. Use When you need to search for any documentation in langchain/langgraph libraries.
    e.g keywords = ['react', 'agent',]
    Args:
        keywords: The keywords to search for.
    Returns:
        The result of the search.
    """
    OWNER = 'websites'
    REPO = 'langchain'

    query = ' '.join(keywords)
    if 'python' not in query:
        query = f'python {query}'

    result = await get_context7_results(OWNER, REPO, 'code', query)
    return result