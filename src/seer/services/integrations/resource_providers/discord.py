"""Discord resource provider for browsing guilds and channels."""
# pylint: disable=too-many-arguments
# Reason: Resource provider list_resources has many filter parameters
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.database import IntegrationResource
from seer.services.integrations.providers.discord import DiscordProvider
from seer.services.integrations.resource_providers.base import ResourceContext, ResourceProvider
from seer.services.integrations.resource_providers.utils import parse_offset
from seer.logger import get_logger

logger = get_logger(__name__)


class DiscordResourceProvider(ResourceProvider):
    """
    Resource provider for Discord.

    Supports:
    - guild: Database-backed list of Discord servers the user has connected
    - channel: API-backed list of channels in a specific guild (requires guild_id dependency)
    """

    provider = "discord"
    resource_configs: Dict[str, Dict[str, Any]] = {
        "guild": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "source": "database",  # Fetched from local database
        },
        "channel": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "depends_on": "guild_id",
            "source": "api",  # Fetched from Discord API
        },
    }

    async def list_resources(
        self,
        *,
        access_token: Optional[str] = None,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
        context: Optional[ResourceContext] = None,
    ) -> Dict[str, Any]:
        """
        List Discord resources (guilds or channels).

        For guilds:
        - Queries local database (IntegrationResource table)
        - Returns list of Discord servers the user has connected

        For channels:
        - Requires guild_id in depends_on_values
        - Fetches channels from Discord API using bot token
        - Filters to text/news/forum channels only

        Args:
            access_token: Not used (Discord uses bot token or database)
            resource_type: "guild" or "channel"
            query: Optional search query
            parent_id: Not used
            page_token: Pagination token (offset-based)
            page_size: Number of results per page
            filter_params: Not used
            depends_on_values: For channels, must contain "guild_id"
            context: ResourceContext with user information

        Returns:
            Standard resource response with items, next_page_token, metadata

        Raises:
            HTTPException: If resource type is unsupported or required params missing
        """
        if not context or not context.user:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Missing user context",
                detail="Discord resource provider requires user context",
                status=400
            )
        assert context is not None  # For type checker
        assert context.user is not None  # For type checker

        if resource_type == "guild":
            return await self._list_guilds(
                user=context.user,
                query=query,
                page_token=page_token,
                page_size=page_size,
            )
        if resource_type == "channel":
            # Extract guild_id from depends_on_values
            guild_id = (depends_on_values or {}).get("guild_id")
            if not guild_id:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing required parameter",
                    detail="guild_id is required to list Discord channels. Please select a guild first.",
                    status=400
                )
            return await self._list_channels(
                user=context.user,
                guild_id=str(guild_id),
                query=query,
                page_token=page_token,
                page_size=page_size,
            )

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Discord resource type '{resource_type}'"
        )

    async def _list_guilds(
        self,
        user: Any,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """
        List Discord guilds from database.

        Args:
            user: User object
            query: Optional search query
            page_token: Pagination token
            page_size: Number of results per page

        Returns:
            Standard resource response
        """
        from seer.api.integrations.services import list_integration_resources  # pylint: disable=import-outside-toplevel  # Avoid circular import

        # List guilds from IntegrationResource records
        resources = await list_integration_resources(
            user,
            provider="discord",
            resource_type="guild",
        )

        # Apply search filter if provided
        filtered_resources = resources
        if query:
            q_lower = query.lower()
            filtered_resources = [
                r for r in resources
                if q_lower in (r.name or "").lower() or q_lower in (r.resource_id or "").lower()
            ]

        # Pagination
        offset = parse_offset(page_token)
        paged_resources = filtered_resources[offset:offset + page_size]

        items = [
            {
                "id": r.resource_id,  # guild_id
                "name": r.name or f"Discord Server {r.resource_id}",
                "display_name": r.name or f"Discord Server {r.resource_id}",
                "type": "guild",
                "metadata": r.resource_metadata or {},
            }
            for r in paged_resources
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_resources) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }

    async def _list_channels(
        self,
        user: Any,
        guild_id: str,
        *,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """
        List Discord channels for a specific guild using Discord API.

        Args:
            user: User object
            guild_id: Discord guild ID
            query: Optional search query
            page_token: Pagination token
            page_size: Number of results per page

        Returns:
            Standard resource response

        Raises:
            HTTPException: If guild not found or API call fails
        """
        # Verify user has access to this guild
        guild_resource = await IntegrationResource.get_or_none(
            user=user,
            provider="discord",
            resource_type="guild",
            resource_id=guild_id,
            status="active"
        )
        if not guild_resource:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Guild not found",
                detail=f"Discord guild {guild_id} not found or you don't have access to it",
                status=404
            )

        # Get bot token from config
        bot_token = config.discord_bot_token
        if not bot_token:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Configuration error",
                detail="Discord bot token not configured",
                status=500
            )
        assert bot_token is not None  # For type checker

        # Fetch channels from Discord API
        provider_impl = DiscordProvider()
        try:
            channels = await provider_impl.fetch_guild_channels(
                guild_id=guild_id,
                bot_token=bot_token
            )
        except HTTPException as exc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch channels",
                detail=exc.detail,
                status=exc.status_code
            )

        # Filter channels (only text channels that bot can send messages to)
        # Channel types: 0=GUILD_TEXT, 2=GUILD_VOICE, 4=GUILD_CATEGORY, 5=GUILD_NEWS, 15=GUILD_FORUM
        # We include text channels (0), news channels (5), and forum channels (15)
        text_channels: List[Dict[str, Any]] = [
            ch for ch in channels
            if ch.get("type") in [0, 5, 15]  # GUILD_TEXT, GUILD_NEWS, GUILD_FORUM
        ]

        # Apply search filter if provided
        filtered_channels = text_channels
        if query:
            q_lower = query.lower()
            filtered_channels = [
                ch for ch in text_channels
                if q_lower in (ch.get("name") or "").lower()
            ]

        # Pagination
        offset = parse_offset(page_token)
        paged_channels = filtered_channels[offset:offset + page_size]

        items = [
            {
                "id": str(ch.get("id", "")),
                "name": ch.get("name") or f"Channel {ch.get('id', '')}",
                "display_name": ch.get("name") or f"Channel {ch.get('id', '')}",
                "type": "channel",
                "metadata": {
                    "channel_id": str(ch.get("id", "")),
                    "channel_name": ch.get("name"),
                    "channel_type": ch.get("type"),
                    "guild_id": guild_id,
                },
            }
            for ch in paged_channels
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_channels) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }
