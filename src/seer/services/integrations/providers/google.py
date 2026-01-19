from __future__ import annotations

from typing import Any, Dict, List

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class GoogleProvider(IntegrationProvider):
    provider = "google"
    aliases = {"gmail", "googlesheets", "googledrive"}
    _required_openid_scopes = ["openid", "email", "profile"]

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """Ensure OpenID scopes are always included."""
        scopes: List[str] = list(dict.fromkeys(context.requested_scopes))
        for item in self._required_openid_scopes:
            if item not in scopes:
                scopes.append(item)
        return " ".join(scopes)

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "state": state,
            "scope": scope,
            "access_type": "offline",
            "prompt": "consent",
        }
        connection = context.existing_connection
        helpers = context.helpers
        if connection and connection.scopes and helpers:
            requested_list = scope.split()
            new_scopes = [
                value
                for value in requested_list
                if not helpers.has_required_scopes(connection.scopes or "", [value])
            ]
            if new_scopes:
                kwargs["include_granted_scopes"] = "true"
                logger.info(
                    "Using incremental authorization for Google. "
                    "Existing scopes: %s..., New scopes: %s",
                    connection.scopes[:100],
                    new_scopes,
                )
        return kwargs

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "userinfo" in token:
            logger.info("Using userinfo embedded in Google token")
            return token["userinfo"]

        try:
            userinfo = await client.userinfo(token=token)
            logger.info("Fetched Google userinfo via client.userinfo")
            return userinfo
        except Exception as exc:
            logger.warning("client.userinfo failed: %s; falling back to manual request", exc)

        access_token = token.get("access_token")
        if not access_token:
            logger.error("Google token missing access_token. keys=%s", list(token.keys()))
            raise HTTPException(
                status_code=500,
                detail="No access token in OAuth response; ensure openid scope is requested.",
            )

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
        if resp.status_code != 200:
            logger.error(
                "Google userinfo request failed status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Google user profile: HTTP {resp.status_code}",
            )
        return resp.json()
