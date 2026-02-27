"""
Slack integration provider.

Handles OAuth 2.0 authentication for Slack bot installations.
Slack uses OAuth 2.0 v2 which separates bot scopes from user scopes.
For workflow automation, we primarily use bot tokens.
"""
from __future__ import annotations

from typing import Any, Dict, List

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)

# Slack API base URL
SLACK_API_BASE = "https://slack.com/api"


class SlackProvider(IntegrationProvider):
    """
    Slack integration provider.

    Handles OAuth 2.0 authentication for Slack bots.
    Token response includes:
    - access_token: Bot token (xoxb-...)
    - authed_user.access_token: User token (optional)
    - team.id: Workspace ID
    - team.name: Workspace name
    """
    provider = "slack"
    resource_types = {"workspace", "channel", "user"}

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        Return bot scopes for Slack OAuth.

        Slack bot scopes are space-separated. Common scopes:
        - channels:read: View basic channel info
        - channels:history: View channel messages
        - chat:write: Send messages
        - users:read: View workspace members
        - groups:read: View private channels (bot must be invited)
        - im:write: Start direct messages
        - reactions:write: Add emoji reactions
        """
        return " ".join(context.requested_scopes)

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        """
        Build Slack OAuth authorization URL parameters.

        Note: Slack v2 OAuth separates bot scopes (scope) from user scopes (user_scope).
        We primarily use bot scopes for automation workflows.
        """
        return {
            "state": state,
            "scope": scope,
            # Don't request user_scope - we only need bot token
        }

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """
        Extract granted scopes from Slack token response.

        Slack v2 OAuth returns scope directly in the token response.
        """
        return token.get("scope") or state_data.get("requested_scope") or ""

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        For Slack bot installations, extract team/workspace info from token.

        The token response already contains:
        - team.id: Workspace ID
        - team.name: Workspace name
        - bot_user_id: Bot's user ID in the workspace
        """
        # Slack token response includes team info directly
        team_info = token.get("team", {})
        return {
            "id": team_info.get("id", ""),
            "name": team_info.get("name", ""),
            "bot_user_id": token.get("bot_user_id", ""),
            "type": "bot_installation",
        }

    async def fetch_workspace_info(self, access_token: str) -> Dict[str, Any]:
        """
        Fetch workspace (team) information from Slack API.

        Args:
            access_token: Slack bot token

        Returns:
            Workspace information dictionary

        Raises:
            HTTPException: If API call fails
        """
        url = f"{SLACK_API_BASE}/team.info"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(url, headers=headers, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()

                if not data.get("ok"):
                    error = data.get("error", "unknown_error")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Slack API error: {error}"
                    )

                return data.get("team", {})
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Slack team.info request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Slack workspace info: HTTP {exc.response.status_code}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error fetching Slack workspace info")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching Slack workspace info: {type(exc).__name__}",
            ) from exc

    async def fetch_channels(
        self,
        access_token: str,
        *,
        types: str = "public_channel,private_channel",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch channels from Slack workspace.

        Args:
            access_token: Slack bot token
            types: Channel types to include (public_channel, private_channel)
            limit: Maximum number of channels to return

        Returns:
            List of channel dictionaries

        Raises:
            HTTPException: If API call fails
        """
        url = f"{SLACK_API_BASE}/conversations.list"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        params = {
            "types": types,
            "limit": limit,
            "exclude_archived": "true",
        }

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(url, headers=headers, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()

                if not data.get("ok"):
                    error = data.get("error", "unknown_error")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Slack API error: {error}"
                    )

                return data.get("channels", [])
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Slack conversations.list request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Slack channels: HTTP {exc.response.status_code}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error fetching Slack channels")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching Slack channels: {type(exc).__name__}",
            ) from exc

    async def join_channel(
        self,
        access_token: str,
        channel_id: str,
    ) -> Dict[str, Any]:
        """
        Join the bot to a public Slack channel.

        Note: Only works for public channels. Private channels require manual invite.

        Args:
            access_token: Slack bot token
            channel_id: Channel ID to join

        Returns:
            Channel information dictionary

        Raises:
            HTTPException: If API call fails or channel is private
        """
        url = f"{SLACK_API_BASE}/conversations.join"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    url,
                    headers=headers,
                    json={"channel": channel_id},
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()

                if not data.get("ok"):
                    error = data.get("error", "unknown_error")
                    if error == "method_not_supported_for_channel_type":
                        raise HTTPException(
                            status_code=400,
                            detail="Cannot join private channels. Bot must be manually invited.",
                        )
                    if error == "channel_not_found":
                        raise HTTPException(
                            status_code=404,
                            detail="Channel not found",
                        )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Slack API error: {error}",
                    )

                return data.get("channel", {})
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Slack conversations.join request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to join Slack channel: HTTP {exc.response.status_code}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error joining Slack channel")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error joining Slack channel: {type(exc).__name__}",
            ) from exc

    async def fetch_users(
        self,
        access_token: str,
        *,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch users from Slack workspace.

        Args:
            access_token: Slack bot token
            limit: Maximum number of users to return

        Returns:
            List of user dictionaries (excludes bots and deleted users)

        Raises:
            HTTPException: If API call fails
        """
        url = f"{SLACK_API_BASE}/users.list"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        params = {"limit": limit}

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(url, headers=headers, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()

                if not data.get("ok"):
                    error = data.get("error", "unknown_error")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Slack API error: {error}"
                    )

                # Filter out bots and deleted users
                users = [
                    u for u in data.get("members", [])
                    if not u.get("is_bot") and not u.get("deleted")
                ]
                return users
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Slack users.list request failed: status=%s, body=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Slack users: HTTP {exc.response.status_code}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error fetching Slack users")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error fetching Slack users: {type(exc).__name__}",
            ) from exc
