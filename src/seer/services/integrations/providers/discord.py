from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class DiscordProvider(IntegrationProvider):
    provider = "discord"
    resource_types = {"guild"}  # Discord servers

    # Bot permissions: View Channels (1024) + Send Messages (2048) = 3072
    BOT_PERMISSIONS = 3072

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """Return the bot scope for Discord bot installation."""
        return "bot"

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        """Include bot permissions in OAuth URL."""
        return {
            "state": state,
            "scope": scope,
            "permissions": self.BOT_PERMISSIONS,
        }

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        For bot installations, we don't get user profile in the traditional sense.
        The token response contains guild information.
        """
        # Discord bot installation token response doesn't include user profile
        # We'll extract guild_id from the callback query params instead
        return token

    async def fetch_guild_info(self, guild_id: str, bot_token: str) -> Dict[str, Any]:
        """
        Fetch guild (server) information from Discord API.

        Args:
            guild_id: Discord guild (server) ID
            bot_token: Discord bot token for API authentication

        Returns:
            Guild information dictionary with name, icon, owner_id, etc.

        Raises:
            HTTPException: If API call fails
        """
        if not bot_token:
            logger.error("Discord bot token not configured")
            raise HTTPException(
                status_code=500,
                detail="Discord bot token not configured",
            )

        url = f"https://discord.com/api/v10/guilds/{guild_id}"
        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(url, headers=headers, timeout=10.0)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Discord guild info request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Discord guild info: HTTP {exc.response.status_code}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error fetching Discord guild info")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching Discord guild info: {type(exc).__name__}",
            ) from exc

    async def fetch_guild_channels(self, guild_id: str, bot_token: str) -> list[Dict[str, Any]]:
        """
        Fetch channels for a Discord guild (server).

        Args:
            guild_id: Discord guild (server) ID
            bot_token: Discord bot token for API authentication

        Returns:
            List of channel dictionaries with id, name, type, etc.

        Raises:
            HTTPException: If API call fails
        """
        if not bot_token:
            logger.error("Discord bot token not configured")
            raise HTTPException(
                status_code=500,
                detail="Discord bot token not configured",
            )

        url = f"https://discord.com/api/v10/guilds/{guild_id}/channels"
        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(url, headers=headers, timeout=10.0)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Discord channels request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Discord channels: HTTP {exc.response.status_code}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error fetching Discord channels")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching Discord channels: {type(exc).__name__}",
            ) from exc
