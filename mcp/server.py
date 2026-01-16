"""Seer MCP Server - Main entry point."""

import asyncio
import sys
from typing import Any, Callable, Dict

from mcp.server import Server  # pylint: disable=import-self,no-name-in-module # Reason: external MCP SDK, not our local package
from mcp.server.stdio import stdio_server  # pylint: disable=no-name-in-module # Reason: external MCP SDK
from mcp.types import TextContent  # pylint: disable=no-name-in-module # Reason: external MCP SDK

from shared.database import User, init_db
from shared.logger import get_logger

from .config import get_config, MCPMode
from .tools import get_workflow_tools, get_execution_tools, get_integration_tools
from .tools.workflows import (
    seer_create_workflow,
    seer_list_workflows,
    seer_get_workflow,
    seer_update_workflow,
    seer_delete_workflow,
)
from .tools.executions import (
    seer_run_workflow,
    seer_get_execution,
    seer_get_execution_history,
    seer_list_executions,
)
from .tools.integrations import (
    seer_list_integrations,
    seer_configure_integration_auth,
)

logger = get_logger("mcp.server")

# Create MCP server
app = Server("seer")


async def get_current_user() -> User:
    """Get the current authenticated user.

    TODO: Implement proper OAuth authentication flow.
    For now, return a mock user for testing.
    """
    users = await User.all().limit(1)
    if not users:
        raise RuntimeError("No users found in database. Please create a user first.")
    return users[0]


# Tool dispatch registry - maps tool names to handler functions
TOOL_HANDLERS: Dict[str, Callable] = {
    # Workflow tools
    "seer_create_workflow": seer_create_workflow,
    "seer_list_workflows": seer_list_workflows,
    "seer_get_workflow": seer_get_workflow,
    "seer_update_workflow": seer_update_workflow,
    "seer_delete_workflow": seer_delete_workflow,
    # Execution tools
    "seer_run_workflow": seer_run_workflow,
    "seer_get_execution": seer_get_execution,
    "seer_get_execution_history": seer_get_execution_history,
    "seer_list_executions": seer_list_executions,
    # Integration tools
    "seer_list_integrations": seer_list_integrations,
    "seer_configure_integration_auth": seer_configure_integration_auth,
}

# Tools that don't require user context
NO_USER_TOOLS = {"seer_configure_integration_auth"}


@app.list_tools()
async def list_tools():
    """List all available MCP tools."""
    workflow_tools = get_workflow_tools()
    execution_tools = get_execution_tools()
    integration_tools = get_integration_tools()

    return workflow_tools + execution_tools + integration_tools


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    """Handle tool calls from MCP client."""
    try:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]

        # Some tools don't require user context
        if name in NO_USER_TOOLS:
            return await handler(**arguments)

        # Get user and call handler
        user = await get_current_user()
        return await handler(user, **arguments)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Top-level error handling for MCP
        logger.error("Error calling tool %s: %s", name, str(e), exc_info=True)
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def main():
    """Main entry point for MCP server."""
    config = get_config()

    logger.info("Starting Seer MCP server in %s mode", config.mode.value)

    if config.mode == MCPMode.LOCAL:
        # Initialize database connection for local mode
        logger.info("Initializing database connection")
        try:
            await init_db()
            logger.info("Database connection established")
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Startup error handling
            logger.error("Failed to initialize database: %s", str(e))
            sys.exit(1)
    else:
        # Cloud mode - use API client
        logger.info("Using cloud API mode: %s", config.api_url)
        # TODO: Implement cloud API client

    # Run the MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
