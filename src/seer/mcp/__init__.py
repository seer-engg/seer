"""
Seer MCP Server - Exposes workflow management capabilities via Model Context Protocol.

This module provides an MCP server that allows external agents (Claude Code, Cursor, etc.)
to programmatically discover tools/triggers and create/manage workflows.
"""

from seer.mcp.server import mcp, main

__all__ = ["mcp", "main"]
