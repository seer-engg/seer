"""
Base class for meetingsEA tools.

Provides shared HTTP client for the Attendee meeting bot service.
Uses platform-owned API key (not user OAuth).
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.config import config
from seer.logger import get_logger
from seer.tools.base import BaseTool

logger = get_logger("tools.meetings_ea.base")

# Request timeout for Attendee API calls
_REQUEST_TIMEOUT_SECONDS = 30


class MeetingsEAToolBase(BaseTool, ABC):
    """
    Abstract base class for meetingsEA (Attendee) tools.

    Provides:
    - API key validation from config
    - HTTP client for Attendee API
    - Consistent error handling

    meetingsEA uses platform-owned API keys (not user OAuth), so these tools
    are always available for all customers without connection setup.
    """

    provider = None  # No OAuth provider
    integration_type = "meetings_ea"
    required_scopes: List[str] = []  # Platform-owned API key, no OAuth

    def _require_config(self) -> tuple[str, str]:
        """Get API key and base URL from config or raise 503."""
        if not config.meetings_ea_api_key:
            raise HTTPException(
                status_code=503,
                detail="meetingsEA is not available: MEETINGS_EA_API_KEY not configured.",
            )
        if not config.meetings_ea_base_url:
            raise HTTPException(
                status_code=503,
                detail="meetingsEA is not available: MEETINGS_EA_BASE_URL not configured.",
            )
        return config.meetings_ea_api_key, config.meetings_ea_base_url

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated HTTP request to the Attendee API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/api/v1/bots")
            json: Request body as dict
            params: Query parameters

        Returns:
            Parsed JSON response dict.

        Raises:
            HTTPException: On API errors.
        """
        api_key, base_url = self._require_config()
        url = f"{base_url.rstrip('/')}{path}"  # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime
        headers = {"Authorization": f"Token {api_key}"}

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            # Extract error detail from Attendee response if available
            detail = f"meetingsEA API error ({e.response.status_code})"
            try:
                error_body = e.response.json()
                if isinstance(error_body, dict):
                    detail = error_body.get("detail", detail)
            except (ValueError, KeyError):
                pass
            logger.warning(
                "meetingsEA API error: %s %s -> %d",
                method, path, e.response.status_code,
                extra={"detail": detail},
            )
            raise HTTPException(
                status_code=502,
                detail=f"meetingsEA: {detail}",
            ) from e
        except httpx.RequestError as e:
            logger.exception("meetingsEA API request failed: %s %s", method, path)
            raise HTTPException(
                status_code=502,
                detail=f"meetingsEA service unavailable: {e!s}",
            ) from e

    def _get_webhook_url(self) -> Optional[str]:
        """Construct the webhook callback URL for Attendee to send events to."""
        base = config.webhook_base_url
        if not base:
            return None
        return f"{base.rstrip('/')}/api/v1/webhooks/meetings_ea"  # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime


# Shared schema fragments for tools that operate on a single bot by ID
BOT_ID_PARAMETERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bot_id": {
            "type": "string",
            "description": "The bot identifier returned by meetings_ea_create_bot.",
        },
    },
    "required": ["bot_id"],
}

BOT_ID_STATE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bot_id": {"type": "string"},
        "state": {"type": "string"},
    },
}


__all__ = ["MeetingsEAToolBase", "BOT_ID_PARAMETERS_SCHEMA", "BOT_ID_STATE_OUTPUT_SCHEMA"]
