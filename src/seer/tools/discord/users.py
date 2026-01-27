"""
Discord user operations - finding users by username or ID.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.discord.base import DISCORD_API_BASE, DiscordAPIClient

logger = get_logger("shared.tools.discord.users")


class DiscordFindUserTool(DiscordAPIClient):
    """Find a Discord user by username or user ID within a guild."""

    name = "discord_find_user"
    description = (
        "Find a Discord user by username or user ID within a server. "
        "Returns user information including ID, username, discriminator, and avatar."
    )
    required_scopes = ["bot", "identify"]
    integration_type = "discord"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID to search in",
                },
                "query": {
                    "type": "string",
                    "description": "Username (partial match) or exact user ID to search for",
                },
            },
            "required": ["guild_id", "query"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for guild_id parameter."""
        return {
            "guild_id": {
                "resource_type": "guild",
                "display_field": "name",
                "value_field": "resource_id",  # guild_id is stored in resource_id
                "search_enabled": True,
                "filter": {"provider": "discord", "resource_type": "guild"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "description": "Array of matching user objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "User ID"},
                            "username": {"type": "string", "description": "Username"},
                            "discriminator": {"type": "string", "description": "User discriminator (4-digit code)"},
                            "global_name": {"type": "string", "description": "User's global display name"},
                            "avatar": {"type": "string", "description": "User avatar hash"},
                            "bot": {"type": "boolean", "description": "Whether user is a bot"},
                            "member": {"type": "object", "description": "Guild member information if user is in guild"},
                        },
                        "required": ["id", "username"],
                    },
                },
            },
            "required": ["users"],
        }

    async def _lookup_user_by_id(
        self,
        *,
        user_id: str,
        guild_id: str,
        credentials: Optional[ResolvedCredentials],
    ) -> Optional[Dict[str, Any]]:
        """
        Try to resolve a Discord user by ID and (optionally) attach guild member info.

        Returns:
            User dict if found, otherwise None.
        """
        try:
            user_resp = await self._make_request(
                "GET",
                f"{DISCORD_API_BASE}/users/{user_id}",
                credentials=credentials
            )
            user = user_resp.json()

            # Attach member info if user is in the guild (best-effort)
            try:
                member_resp = await self._make_request(
                    "GET",
                    f"{DISCORD_API_BASE}/guilds/{guild_id}/members/{user_id}",
                    credentials=credentials
                )
                user["member"] = member_resp.json()
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise

            return user
        except HTTPException as exc:
            if exc.status_code == 404:
                return None
            raise

    async def _search_guild_members(
        self,
        *,
        guild_id: str,
        query: str,
        credentials: Optional[ResolvedCredentials],
    ) -> list[Dict[str, Any]]:
        """
        Search guild members by username (partial match).

        Note: This endpoint requires the bot to have the GUILD_MEMBERS intent.
        """
        try:
            search_resp = await self._make_request(
                "GET",
                f"{DISCORD_API_BASE}/guilds/{guild_id}/members/search",
                credentials=credentials,
                params={"query": query, "limit": 25}  # Discord allows up to 25 results
            )
            members = search_resp.json()

            users: list[Dict[str, Any]] = []
            for member in members:
                user_info = member.get("user", {})
                user_info["member"] = member
                users.append(user_info)
            return users
        except HTTPException as exc:
            if exc.status_code == 403:
                logger.warning(
                    "Guild member search failed (403): Bot may lack GUILD_MEMBERS intent or permission. "
                    "Guild ID: %s, Query: %s",
                    guild_id,
                    query
                )
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Guild member search requires the bot to have the GUILD_MEMBERS intent "
                        "and appropriate permissions. User ID lookup may still work."
                    )
                ) from exc
            raise

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        guild_id = str(arguments.get("guild_id") or "")
        query = str(arguments.get("query") or "").strip()

        if not guild_id:
            raise HTTPException(status_code=400, detail="Parameter 'guild_id' is required")
        if not query:
            raise HTTPException(status_code=400, detail="Parameter 'query' is required")

        logger.info("Searching for Discord user: guild_id=%s, query=%s", guild_id, query)

        # Check if query is a numeric user ID (exact match)
        if query.isdigit() and len(query) >= 17:  # Discord user IDs are 17-19 digits
            user = await self._lookup_user_by_id(user_id=query, guild_id=guild_id, credentials=credentials)
            if user is not None:
                return {"users": [user]}
            logger.debug("User ID %s not found, trying guild member search", query)

        users = await self._search_guild_members(guild_id=guild_id, query=query, credentials=credentials)
        if not users:
            logger.info("No users found matching query '%s' in guild %s", query, guild_id)
            return {"users": []}

        logger.info("Found %d users matching query '%s' in guild %s", len(users), query, guild_id)
        return {"users": users}
