"""
Adapter implementing McpServerConfigResolver protocol for runtime MCP config resolution.

Bridges the core/ protocol with the services/ layer that has DB access.
"""
from __future__ import annotations

from typing import Any, Dict

from seer.database import User


class McpServerConfigResolverImpl:
    """Resolves saved MCP server configs from IntegrationResource + IntegrationSecret."""

    def __init__(self, user: User) -> None:
        self._user = user

    async def resolve(self, resource_id: int) -> Dict[str, Any]:
        from seer.api.integrations.services import resolve_mcp_server_config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

        return await resolve_mcp_server_config(self._user, resource_id)
