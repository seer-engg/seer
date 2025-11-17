"""Action execution core logic for test runner."""
from typing import List, Dict

from shared.tool_service import get_tool_service
from shared.logger import get_logger
from langchain_core.tools import BaseTool



logger = get_logger("test_runner.action_executor")


async def load_mcp_tools(mcp_services: List[str]) -> Dict[str, BaseTool]:
    """Load MCP tools using ToolService singleton."""
    tool_service = get_tool_service()
    await tool_service.initialize(mcp_services)
    tools_dict = tool_service.get_tools()

    if not tools_dict and mcp_services:
        logger.error(
            f"MCP services {mcp_services} were requested, but no tools were loaded."
        )
        logger.error(
            "This usually means the local MCP servers (ports 8004, 8005) are not running."
        )
    else:
        logger.info(
            f"Loaded {len(tools_dict)} MCP tools from ToolService"
        )
    return tools_dict

