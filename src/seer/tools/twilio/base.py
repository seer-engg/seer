"""
Base class for Twilio API tools.

Provides shared HTTP client for Twilio REST API with Basic Auth.
Twilio uses API-key auth (account_sid:auth_token), not OAuth.
Credentials are stored as user secrets via the integrations system.
"""

from abc import ABC
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.twilio.base")

TWILIO_API_BASE = "https://api.twilio.com"


class TwilioAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Twilio API tools.

    Uses Basic Auth with account_sid:auth_token from user secrets.
    """

    provider = "twilio"
    integration_type = "twilio"
    required_secrets = ["twilio_account_sid", "twilio_auth_token", "twilio_from_number"]
    default_timeout: float = 30.0

    def _get_credentials(self, credentials: Optional[ResolvedCredentials] = None) -> tuple[str, str, str]:
        """Extract Twilio credentials from resolved secrets, falling back to app config."""
        # Try user-level secrets first
        if credentials and credentials.secrets:
            account_sid = credentials.secrets.get("twilio_account_sid")
            auth_token = credentials.secrets.get("twilio_auth_token")
            from_number = credentials.secrets.get("twilio_from_number")
            if account_sid and auth_token and from_number:
                return account_sid, auth_token, from_number

        # Fall back to app-level config (for system tasks like PULIS escalation)
        from seer.config import config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module level
        account_sid = config.twilio_account_sid
        auth_token = config.twilio_auth_token
        from_number = config.twilio_from_number

        if account_sid and auth_token and from_number:
            return account_sid, auth_token, from_number

        raise HTTPException(
            status_code=401,
            detail=f"{self.name} requires Twilio credentials. Set them via integration secrets or app config.",
        )

    async def _make_request(
        self,
        method: str,
        path: str,
        credentials: Optional[ResolvedCredentials] = None,
        *,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to Twilio REST API."""
        account_sid, auth_token, _ = self._get_credentials(credentials)
        timeout_value = timeout or self.default_timeout
        url = f"{TWILIO_API_BASE}{path}"

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                resp = await client.request(
                    method,
                    url,
                    auth=(account_sid, auth_token),
                    data=data,
                )

                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        error_data = resp.json()
                        detail = error_data.get("message", resp.text)
                    except (ValueError, KeyError):
                        pass
                    raise HTTPException(status_code=resp.status_code, detail=f"{self.name}: {detail}")

                return resp.json()

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(status_code=500, detail=f"{self.name} error: {str(exc)}") from exc
