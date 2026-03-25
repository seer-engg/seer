from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.services.integrations.providers.base import IntegrationProvider

logger = get_logger(__name__)


class OuraProvider(IntegrationProvider):
    """Oura Ring OAuth integration provider."""

    provider = "oura"
    resource_types: set[str] = set()

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch user profile from Oura personal_info endpoint.

        Oura is plain OAuth2 (not OIDC), so the token response doesn't
        include userinfo. We must call the API explicitly.
        """
        _ = client, state_data
        access_token = token.get("access_token")
        if not access_token:
            logger.error("Oura token missing access_token. keys=%s", list(token.keys()))
            raise HTTPException(
                status_code=500,
                detail="No access token in Oura OAuth response.",
            )

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                "https://api.ouraring.com/v2/usercollection/personal_info",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )

        if resp.status_code != 200:
            logger.error("Oura personal_info failed: %s %s", resp.status_code, resp.text)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Oura user profile: {resp.status_code}",
            )

        return resp.json()

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """Normalize Oura scopes by stripping the ``extapi:`` prefix.

        Oura returns scopes like ``extapi:heartrate`` in the token response,
        but tool definitions use bare scope names like ``heartrate``.
        """
        raw = token.get("scope") or state_data.get("requested_scope") or ""
        return " ".join(s.removeprefix("extapi:") for s in raw.split() if s)
