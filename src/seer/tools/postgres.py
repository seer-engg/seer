"""
Async-native PostgreSQL connector with LangChain-compatible tools.

This is the public API module that re-exports components from:
- postgres_client: PostgresClient class
- postgres_tools: Tool factory functions
- postgres_provider: PostgresProvider class

Usage:
    from seer.tools.postgres import PostgresClient, get_postgres_tools

    # Get tools for an agent
    client = PostgresClient(connection_string="postgresql://...")
    tools = client.get_tools()

    # Or use standalone
    tools = get_postgres_tools(connection_string="postgresql://...")
"""
import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

# Re-exports from submodules
from seer.tools.postgres_client import PostgresClient, _get_asyncpg
from seer.tools.postgres_provider import PostgresProvider

__all__ = [
    "PostgresClient",
    "PostgresProvider",
    "get_default_client",
    "close_default_client",
    "get_postgres_tools",
]

# Global client instance for simple usage
_default_client: Optional[PostgresClient] = None
_default_client_lock = asyncio.Lock()


async def get_default_client(connection_string: Optional[str] = None) -> PostgresClient:
    """
    Get or create a default PostgresClient instance.

    Args:
        connection_string: PostgreSQL connection URI. If not provided,
            uses the DATABASE_URL from seer.config.

    Returns:
        PostgresClient instance
    """
    global _default_client

    async with _default_client_lock:
        if _default_client is None:
            if connection_string is None:
                from seer.config import config
                connection_string = config.DATABASE_URL

            if not connection_string:
                raise ValueError(
                    "No PostgreSQL connection string provided. "
                    "Set DATABASE_URL environment variable or pass connection_string."
                )

            _default_client = PostgresClient(connection_string)
            await _default_client.connect()

        return _default_client


async def close_default_client() -> None:
    """Close the default PostgresClient if it exists."""
    global _default_client

    async with _default_client_lock:
        if _default_client is not None:
            await _default_client.close()
            _default_client = None


def get_postgres_tools(
    connection_string: Optional[str] = None,
    client: Optional[PostgresClient] = None,
) -> List[BaseTool]:
    """
    Get PostgreSQL LangChain tools.

    Convenience function to get PostgreSQL tools either from a provided client
    or by creating a new one with the given connection string.

    Args:
        connection_string: PostgreSQL connection URI
        client: Existing PostgresClient instance (takes precedence)

    Returns:
        List of LangChain BaseTool instances

    Example:
        # With connection string
        tools = get_postgres_tools("postgresql://user:pass@localhost/db")

        # With existing client
        client = PostgresClient("postgresql://...")
        tools = get_postgres_tools(client=client)
    """
    if client is not None:
        return client.get_tools()

    if connection_string is None:
        from seer.config import config
        connection_string = config.DATABASE_URL

    if not connection_string:
        raise ValueError(
            "No PostgreSQL connection string provided. "
            "Set DATABASE_URL environment variable or pass connection_string."
        )

    new_client = PostgresClient(connection_string)
    return new_client.get_tools()
