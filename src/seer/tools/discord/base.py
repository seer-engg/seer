"""
Base class for Discord API tools.

Provides shared functionality for all Discord API integrations:
- HTTP client with bot token authentication
- Consistent error handling
- Common response parsing

All Discord tools should inherit from DiscordAPIClient to eliminate code
duplication and ensure consistent behavior.
"""
# pylint: disable=duplicate-code  # Reason: Discord and Slack base classes share similar error handling patterns intentionally

from abc import ABC
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.config import config
from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.discord.base")

# Discord API base URL
DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Discord API tools.

    Provides centralized HTTP request handling, bot token authentication,
    error translation, and consistent behavior.

    Subclasses only need to implement:
    - Class attributes (name, description, required_scopes, integration_type, required_permissions)
    - execute() method with business logic

    Example:
        class DiscordSendMessageTool(DiscordAPIClient):
            name = "discord_send_message"
            description = "Send a message to Discord"
            required_scopes = ["bot"]
            integration_type = "discord"
            required_permissions = 3072  # VIEW_CHANNEL | SEND_MESSAGES

            async def execute(self, access_token, arguments, credentials=None):
                resp = await self._make_request(
                    "POST",
                    f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                    credentials=credentials,
                    json_body={"content": "Hello"}
                )
                return resp.json()
    """

    provider = "discord"
    default_timeout: float = 30.0
    required_permissions: int = 0  # Override in subclasses with Discord permission bitfield

    def _get_bot_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:  # pylint: disable=unused-argument
        """
        Get Discord bot token from config.

        Args:
            credentials: Optional ResolvedCredentials (not used for Discord, but kept for API consistency)

        Returns:
            Bot token string

        Raises:
            HTTPException: 500 if bot token is not configured
        """
        if not config.discord_bot_token:
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} requires Discord bot token to be configured"
            )
        return config.discord_bot_token

    def _build_headers(self, bot_token: str) -> Dict[str, str]:
        """
        Build HTTP headers for Discord API request.

        Args:
            bot_token: Discord bot token

        Returns:
            Dict with Authorization and Content-Type headers
        """
        return {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (Seer, 1.0)"
        }

    async def _make_request(  # pylint: disable=too-many-arguments  # Reason: Discord API base method requires all HTTP request parameters
        self,
        method: str,
        url: str,
        credentials: Optional[ResolvedCredentials] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """
        Make authenticated HTTP request to Discord API.

        Handles:
        - Runtime permission validation (if required_permissions is set)
        - Bot token retrieval from config
        - Header construction
        - Timeout management
        - Error translation to HTTPException

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE, etc.)
            url: Full API endpoint URL
            credentials: Optional ResolvedCredentials (for API consistency, not used for Discord)
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout in seconds (default: 30.0)

        Returns:
            httpx.Response object (already checked for errors)

        Raises:
            HTTPException: 403 for missing permissions, 500 for config errors, 401 for auth errors,
                          404 for not found, 429 for rate limits, 504 for timeouts, 500 for other errors

        Example:
            resp = await self._make_request(
                "POST",
                f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                credentials=credentials,
                json_body={"content": "Hello"}
            )
            data = resp.json()
        """
        # Validate permissions before making request
        if credentials and hasattr(self, 'required_permissions') and self.required_permissions:
            from seer.tools.discord.validators import validate_discord_permissions  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import between base and validators

            connection = getattr(credentials, 'oauth_connection', None)
            if connection:
                validate_discord_permissions(
                    connection=connection,
                    required_permissions=self.required_permissions,
                    tool_name=self.name
                )

        bot_token = self._get_bot_token(credentials)
        headers = self._build_headers(bot_token)
        timeout_value = timeout or self.default_timeout

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body
                )

                if resp.is_error:
                    raise self._handle_api_error(resp)

                return resp

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s"
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} error: {str(exc)}"
            ) from exc

    def _handle_api_error(self, response: httpx.Response) -> HTTPException:
        """
        Translate Discord API HTTP error responses to FastAPI HTTPExceptions.

        Provides consistent, user-friendly error messages across all Discord tools.

        Args:
            response: httpx.Response with error status code

        Returns:
            HTTPException with appropriate status code and detail message

        Common Status Codes:
            401: Authentication failed (invalid bot token)
            403: Permission denied (bot lacks permissions)
            404: Resource not found (channel, user, guild doesn't exist)
            429: Rate limit exceeded
            500+: Discord API server errors
        """
        body_snippet = response.text[:500] if response.text else ""

        if response.status_code == 401:
            return HTTPException(
                status_code=401,
                detail=f"{self.name}: Authentication failed. Bot token may be invalid."
            )

        if response.status_code == 403:
            return HTTPException(
                status_code=403,
                detail=f"{self.name}: Permission denied. Bot may lack required permissions."
            )

        if response.status_code == 404:
            return HTTPException(
                status_code=404,
                detail=f"{self.name}: Resource not found (channel, user, or guild may not exist)."
            )

        if response.status_code == 429:
            # Discord rate limit responses include retry-after header
            retry_after = response.headers.get("Retry-After", "unknown")
            return HTTPException(
                status_code=429,
                detail=f"{self.name}: Rate limit exceeded. Retry after {retry_after} seconds."
            )

        if response.status_code >= 500:
            return HTTPException(
                status_code=response.status_code,
                detail=f"{self.name}: Discord API server error (status {response.status_code})."
            )

        return HTTPException(
            status_code=response.status_code,
            detail=f"{self.name}: API error (status {response.status_code}): {body_snippet}"
        )
