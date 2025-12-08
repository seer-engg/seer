# shared/tools/mcp_client.py

from langchain_mcp_adapters.client import MultiServerMCPClient  
from shared.config import config
import asyncio
from typing import Optional, Any

# Build headers dict, filtering out None values
context7_headers = {}
if config.CONTEXT7_API_KEY:
    context7_headers["CONTEXT7_API_KEY"] = config.CONTEXT7_API_KEY

client = MultiServerMCPClient(
    {
        "context7": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp",
            "headers": context7_headers if context7_headers else None,
        },
        "langchain": {
            "transport": "streamable_http",
            "url": "https://docs.langchain.com/mcp",
        }
    }
)

# In Jupyter notebook, we need to comment out the asyncio.run() to avoid blocking the event loop
# Load tools with error handling - allow server to start even if some tools fail
try:
    LANGCHAIN_DOCS_TOOLS = asyncio.run(client.get_tools(server_name="langchain"))
except Exception as e:
    import logging
    logging.warning(f"Failed to load LangChain docs tools: {e}")
    LANGCHAIN_DOCS_TOOLS = []

try:
    if config.CONTEXT7_API_KEY:
        CONTEXT7_TOOLS = asyncio.run(client.get_tools(server_name="context7"))
    else:
        CONTEXT7_TOOLS = []
except Exception as e:
    import logging
    logging.warning(f"Failed to load Context7 tools: {e}")
    CONTEXT7_TOOLS = []

CONTEXT7_LIBRARY_TOOL = None
if CONTEXT7_TOOLS:
    library_tools = [tool for tool in CONTEXT7_TOOLS if tool.name == "get-library-docs"]
    if library_tools:
        CONTEXT7_LIBRARY_TOOL = library_tools[0]