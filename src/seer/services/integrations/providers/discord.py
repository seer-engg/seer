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

    # Default bot permissions if no specific tools are requested
    # VIEW_CHANNELS (1024) + SEND_MESSAGES (2048) = 3072
    DEFAULT_PERMISSIONS = 3072

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        Return the bot scope for Discord bot installation.

        Discord always uses "bot" scope. Permissions are handled separately
        via the 'permissions' query parameter in build_authorize_kwargs().
        """
        return "bot"

    def _calculate_requested_permissions(self, context: OAuthAuthorizeContext) -> int:
        """
        Calculate permission bitfield from requested tools/scopes.

        The frontend passes tool names as space-separated string in the scope parameter.
        We extract those tool names and calculate the minimal required permissions.

        Args:
            context: OAuthAuthorizeContext with requested_scopes (tool names)

        Returns:
            Permission bitfield (int)
        """
        from seer.tools.discord.permissions import calculate_permissions  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import between providers and tools

        # If requested_scopes contains tool names, calculate permissions
        if context.requested_scopes:
            permissions = calculate_permissions(context.requested_scopes)
            if permissions > 0:
                logger.info(
                    "Calculated Discord permissions from tools: tools=%s, permissions=%s",
                    context.requested_scopes,
                    permissions
                )
                return permissions

        # Fallback to default permissions
        logger.warning(
            "No valid Discord tools in requested_scopes, using default permissions: %s",
            self.DEFAULT_PERMISSIONS
        )
        return self.DEFAULT_PERMISSIONS

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        """
        Include bot permissions in OAuth URL with incremental authorization support.

        Calculates minimal permissions based on requested tools. If user already
        has an existing connection, merges new permissions with existing ones.

        Args:
            context: OAuthAuthorizeContext with existing connection info
            state: OAuth state parameter
            scope: OAuth scope (always "bot" for Discord)

        Returns:
            Dict with state, scope, and permissions parameters for OAuth URL
        """
        requested_permissions = self._calculate_requested_permissions(context)

        # Check for incremental authorization
        existing_connection = context.existing_connection
        if existing_connection and existing_connection.provider_metadata:
            existing_perms = existing_connection.provider_metadata.get("permissions", 0)
            new_perms_needed = requested_permissions & ~existing_perms

            if new_perms_needed:
                # Incremental: request all permissions (existing + new)
                # Discord doesn't have include_granted_scopes like Google, so we
                # need to request the full merged set
                final_permissions = existing_perms | requested_permissions
                logger.info(
                    "Discord incremental authorization: existing=%s, requested=%s, new=%s, final=%s",
                    existing_perms,
                    requested_permissions,
                    new_perms_needed,
                    final_permissions
                )
            else:
                # Already have all required permissions
                logger.info(
                    "Discord: Already have all required permissions (%s)",
                    existing_perms
                )
                final_permissions = existing_perms
        else:
            # First-time authorization
            final_permissions = requested_permissions
            logger.info(
                "Discord first-time authorization with permissions: %s",
                final_permissions
            )

        return {
            "state": state,
            "scope": scope,
            "permissions": final_permissions,
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
