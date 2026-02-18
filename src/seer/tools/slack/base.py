"""
Base class for Slack API tools.

Provides shared functionality for all Slack API integrations:
- HTTP client with bot token authentication
- Consistent error handling
- Common response parsing
"""

from abc import ABC
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.slack.base")

# Slack API base URL
SLACK_API_BASE = "https://slack.com/api"


class SlackAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Slack API tools.

    Provides centralized HTTP request handling, bot token authentication,
    error translation, and consistent behavior.
    """

    provider = "slack"
    default_timeout: float = 30.0

    def _get_bot_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:
        """
        Get Slack bot token from credentials.

        Args:
            credentials: ResolvedCredentials containing the access token

        Returns:
            Bot token string

        Raises:
            HTTPException: 401 if no token is available
        """
        if credentials and credentials.access_token:
            return credentials.access_token

        raise HTTPException(
            status_code=401,
            detail=f"{self.name} requires a Slack bot token. Please connect your Slack workspace."
        )

    def _build_headers(self, bot_token: str) -> Dict[str, str]:
        """Build HTTP headers for Slack API request."""
        return {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        credentials: Optional[ResolvedCredentials] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to Slack API.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., "chat.postMessage")
            credentials: ResolvedCredentials with access token
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout in seconds

        Returns:
            Slack API response dict

        Raises:
            HTTPException: On API errors
        """
        bot_token = self._get_bot_token(credentials)
        headers = self._build_headers(bot_token)
        timeout_value = timeout or self.default_timeout
        url = f"{SLACK_API_BASE}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                if method.upper() == "GET":
                    resp = await client.get(url, headers=headers, params=params)
                else:
                    resp = await client.post(url, headers=headers, params=params, json=json_body)

                resp.raise_for_status()
                data = resp.json()

                # Slack returns {"ok": false, "error": "..."} on API errors
                if not data.get("ok"):
                    raise self._handle_slack_error(data)

                return data

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

    def _handle_slack_error(self, response: Dict[str, Any]) -> HTTPException:
        """
        Translate Slack API error responses to HTTPExceptions.
        """
        error = response.get("error", "unknown_error")
        error_detail = response.get("response_metadata", {}).get("messages", [])

        error_messages = {
            "channel_not_found": (404, "Channel not found"),
            "not_in_channel": (403, "Bot is not in the specified channel"),
            "is_archived": (400, "Channel is archived"),
            "msg_too_long": (400, "Message is too long (max 40,000 characters)"),
            "no_text": (400, "Message text is required"),
            "rate_limited": (429, "Rate limited. Please try again later."),
            "invalid_auth": (401, "Invalid authentication token"),
            "account_inactive": (401, "Account is inactive"),
            "token_revoked": (401, "Token has been revoked"),
            "user_not_found": (404, "User not found"),
            "user_not_visible": (403, "User is not visible to the bot"),
            "cannot_dm_bot": (400, "Cannot send DM to a bot"),
        }

        status_code, message = error_messages.get(error, (500, f"Slack API error: {error}"))

        if error_detail:
            message = f"{message}. Details: {', '.join(error_detail)}"

        return HTTPException(status_code=status_code, detail=f"{self.name}: {message}")
