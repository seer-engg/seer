
from langchain_mcp_adapters.client import MultiServerMCPClient  
import asyncio


docs_client = MultiServerMCPClient(  
    {
        "langchain_docs": {
            "transport": "streamable_http",
            "url": "https://docs.langchain.com/mcp",
        },
        "langchain_docs": {
            "transport": "streamable_http",
            "url": "https://docs.composio.dev/_mcp/server",
        }
    }
)

def get_docs_tools():
    return asyncio.run(docs_client.get_tools())

docs_tools = get_docs_tools()