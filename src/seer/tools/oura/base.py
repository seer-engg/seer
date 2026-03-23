"""
Base class for Oura Ring API tools.

Provides shared functionality for all Oura API integrations:
- HTTP client with OAuth Bearer token authentication
- Consistent error handling
- Date range parameter handling
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.oura.base")

OURA_API_BASE = "https://api.ouraring.com/v2/usercollection"


class OuraAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Oura Ring API tools.

    Subclasses implement:
    - Class attributes (name, description, required_scopes, integration_type)
    - execute() method with business logic
    """

    provider = "oura"
    default_timeout: float = 30.0

    def _get_access_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:
        """Get OAuth access token from resolved credentials."""
        if credentials and credentials.access_token:
            return credentials.access_token
        raise HTTPException(
            status_code=401,
            detail=f"{self.name} requires an Oura Ring OAuth connection"
        )

    async def _make_request(
        self,
        method: str,
        url: str,
        credentials: Optional[ResolvedCredentials] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make authenticated HTTP request to Oura API."""
        token = self._get_access_token(credentials)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        actual_timeout = timeout or self.default_timeout

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    timeout=actual_timeout,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"Oura API timeout: {exc}") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Oura API request error: {exc}") from exc

        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Oura OAuth token expired or invalid. Re-authenticate.")
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Oura API rate limit exceeded. Try again later.")
        if resp.status_code >= 400:
            body = resp.text
            raise HTTPException(status_code=resp.status_code, detail=f"Oura API error: {resp.status_code} {body}")

        return resp

    @staticmethod
    def date_range_params_schema() -> Dict[str, Any]:
        """Shared parameter schema for date-range tools."""
        return {
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format. Defaults to today.",
                "default": None,
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format. Defaults to today.",
                "default": None,
            },
        }
