"""
Registry for managing MCP (Model Context Protocol) client connections.

Provides connection pooling, TTL-based cleanup, and compile-time tool validation
for both HTTP and stdio MCP servers.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

try:
    from langchain_mcp_adapters.client import StreamableHttpConnection, create_session
except ImportError:
    StreamableHttpConnection = None  # type: ignore[assignment,misc]
    create_session = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    server: str
    server_type: Literal["http", "stdio"]
    auth: Optional[Dict[str, Any]] = None

    def cache_key(self) -> str:
        """
        Generate cache key excluding auth for compilation.

        Auth is intentionally excluded because:
        1. At compile-time, auth expressions aren't resolved yet (e.g., ${secrets.api_key})
        2. We want to reuse connections across requests with same server URL
        3. Auth is applied at runtime when invoking tools
        """
        return hashlib.sha256(f"{self.server}:{self.server_type}".encode()).hexdigest()[
            :16
        ]


@dataclass
class MCPClientEntry:
    """Entry in the client registry with lifecycle tracking."""

    session: ClientSession
    config: MCPServerConfig
    created_at: datetime
    transport: Any = None  # holds the stdio transport CM for proper cleanup
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


class MCPClientRegistry:
    """
    Registry for managing MCP client connections with TTL-based cleanup.

    Features:
    - Thread-safe async initialization
    - Connection pooling by server URL
    - Auto-cleanup of idle connections (5 min TTL)
    - Compile-time tool validation
    """

    CONNECTION_TTL = timedelta(minutes=5)
    CLEANUP_INTERVAL = 60  # seconds

    def __init__(self) -> None:
        self._clients: Dict[str, MCPClientEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task and close all connections."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.close_all()

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup stale connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except (OSError, RuntimeError) as exc:
                logger.exception("Error in MCP cleanup loop: %s", exc)

    async def _cleanup_stale_connections(self) -> None:
        """Close connections idle for longer than TTL."""
        now = datetime.now(timezone.utc)
        async with self._lock:
            stale_keys = [
                key
                for key, entry in self._clients.items()
                if (now - entry.last_accessed) > self.CONNECTION_TTL
            ]
            for key in stale_keys:
                entry = self._clients.pop(key)
                try:
                    await entry.session.close()
                    if entry.transport is not None:
                        await entry.transport.__aexit__(None, None, None)  # pylint: disable=unnecessary-dunder-call
                    logger.debug(
                        "Closed stale MCP connection: %s", entry.config.server
                    )
                except OSError as exc:
                    logger.warning("Error closing stale connection: %s", exc)

    async def _create_client(self, config: MCPServerConfig) -> tuple[ClientSession, Any]:
        """Create a new MCP client session. Returns (session, transport) for cleanup."""
        if config.server_type != "stdio":
            raise ValueError(
                f"Unsupported MCP server type: {config.server_type}. "
                "HTTP servers must use _http_session() context manager."
            )

        # Parse stdio command from server URL
        # Expected format: "command arg1 arg2" or just "command"
        parts = config.server.split()
        if not parts:
            raise ValueError("stdio server command cannot be empty")

        server_params = StdioServerParameters(
            command=parts[0], args=parts[1:] if len(parts) > 1 else []
        )

        # Apply env vars from auth if provided
        if config.auth and "env" in config.auth:
            server_params.env = config.auth["env"]

        # Enter the stdio transport context manager and keep it alive for pooling.
        # Manual __aenter__ is required because the session must outlive this method.
        stdio_transport = stdio_client(server_params)
        read, write = await stdio_transport.__aenter__()  # pylint: disable=no-member
        session = ClientSession(read, write)
        await session.__aenter__()  # pylint: disable=unnecessary-dunder-call
        return session, stdio_transport

    async def _detect_http_transport(self, config: MCPServerConfig, headers: Optional[Dict[str, str]]) -> str:
        """Probe whether Streamable HTTP works; fall back to SSE."""
        if create_session is None:
            raise ImportError(
                "langchain-mcp-adapters is required for HTTP MCP support. "
                "Install with: uv pip install langchain-mcp-adapters"
            )

        streamable_conn: StreamableHttpConnection = {"transport": "streamable_http", "url": config.server}
        if headers:
            streamable_conn["headers"] = headers

        try:
            async with create_session(streamable_conn) as session:
                await session.initialize()
            return "streamable_http"
        except OSError:
            logger.debug("Streamable HTTP failed for %s, trying SSE transport", config.server)
            return "sse"

    @asynccontextmanager
    async def _http_session(self, config: MCPServerConfig) -> AsyncIterator[ClientSession]:
        """Create a short-lived HTTP MCP session, auto-detecting SSE vs Streamable HTTP transport."""
        if create_session is None:
            raise ImportError(
                "langchain-mcp-adapters is required for HTTP MCP support. "
                "Install with: uv pip install langchain-mcp-adapters"
            )

        headers = None
        if config.auth and "headers" in config.auth:
            headers = config.auth["headers"]

        transport = await self._detect_http_transport(config, headers)

        conn: Dict[str, Any] = {"transport": transport, "url": config.server}
        if headers:
            conn["headers"] = headers

        async with create_session(conn) as session:  # pylint: disable=contextmanager-generator-missing-cleanup
            yield session

    async def get_client(self, config: MCPServerConfig) -> ClientSession:
        """Get or create MCP client session, updating last_accessed time."""
        cache_key = config.cache_key()

        # Fast path: client exists and update access time
        if cache_key in self._clients:
            entry = self._clients[cache_key]
            entry.last_accessed = datetime.now(timezone.utc)
            entry.access_count += 1
            return entry.session

        # Slow path: create new client
        async with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._clients:
                entry = self._clients[cache_key]
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                return entry.session

            logger.info("Creating new MCP connection: %s", config.server)
            session, transport = await self._create_client(config)
            now = datetime.now(timezone.utc)
            self._clients[cache_key] = MCPClientEntry(
                session=session,
                config=config,
                created_at=now,
                transport=transport,
                last_accessed=now,
                access_count=1,
            )
            return session

    async def list_tools(self, config: MCPServerConfig) -> list[Tool]:
        """
        List all available tools from an MCP server.

        Used for compile-time validation and resource picker integration.
        """
        if config.server_type == "http":
            async with self._http_session(config) as session:
                result = await session.list_tools()
                return result.tools

        session = await self.get_client(config)
        result = await session.list_tools()
        return result.tools

    async def validate_tool(
        self, config: MCPServerConfig, tool_name: str
    ) -> Dict[str, Any]:
        """
        Validate tool existence and fetch schema (compile-time).

        Returns:
            {
                "name": str,
                "description": str,
                "input_schema": dict,
            }

        Raises:
            ConnectionError: If server unreachable
            ValueError: If tool not found
        """
        try:
            tools = await self.list_tools(config)
        except (OSError, RuntimeError) as exc:
            raise ConnectionError(
                f"Failed to connect to MCP server '{config.server}': {exc}"
            ) from exc

        tool_dict = {t.name: t for t in tools}
        if tool_name not in tool_dict:
            available = ", ".join(sorted(tool_dict.keys()))
            raise ValueError(
                f"Tool '{tool_name}' not found on MCP server '{config.server}'. "
                f"Available tools: {available or '(none)'}"
            )

        tool = tool_dict[tool_name]
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema or {},
        }

    async def invoke_tool(
        self, config: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Invoke a tool on an MCP server.

        Used at runtime to execute MCP tools with resolved auth.
        """
        if config.server_type == "http":
            async with self._http_session(config) as session:
                result = await session.call_tool(tool_name, arguments)
                return self._parse_tool_result(result)

        session = await self.get_client(config)
        result = await session.call_tool(tool_name, arguments)
        return self._parse_tool_result(result)

    @staticmethod
    def _parse_tool_result(result: CallToolResult) -> Any:
        """Extract content from a CallToolResult."""
        if len(result.content) == 1 and hasattr(result.content[0], "text"):
            return result.content[0].text
        return [
            {"type": item.type, "text": getattr(item, "text", None)}
            for item in result.content
        ]

    async def close_all(self) -> None:
        """Close all active MCP connections."""
        async with self._lock:
            for _, entry in list(self._clients.items()):
                try:
                    await entry.session.close()
                    if entry.transport is not None:
                        await entry.transport.__aexit__(None, None, None)  # pylint: disable=unnecessary-dunder-call
                    logger.debug("Closed MCP connection: %s", entry.config.server)
                except OSError as exc:
                    logger.warning("Error closing connection: %s", exc)
            self._clients.clear()
