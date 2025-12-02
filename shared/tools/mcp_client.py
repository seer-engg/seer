# shared/tools/mcp_client.py

from langchain_mcp_adapters.client import MultiServerMCPClient  
from shared.config import config
import asyncio
from typing import Optional, Any

client = MultiServerMCPClient(
    {
        "context7": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp",
            "headers": {
                "CONTEXT7_API_KEY": config.CONTEXT7_API_KEY,
            },
        },
        "langchain": {
            "transport": "streamable_http",
            "url": "https://docs.langchain.com/mcp",
        }
    }
)

# In Jupyter notebook, we need to comment out the asyncio.run() to avoid blocking the event loop
LANGCHAIN_DOCS_TOOLS = asyncio.run(client.get_tools(server_name="langchain"))
CONTEXT7_TOOLS = asyncio.run(client.get_tools(server_name="context7"))

CONTEXT7_LIBRARY_TOOL = [tool for tool in CONTEXT7_TOOLS if tool.name == "get-library-docs"][0]