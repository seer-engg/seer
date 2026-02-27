from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class GoogleProvider(IntegrationProvider):
    provider = "google"
    aliases = {"gmail", "googlesheets", "googledrive", "googlecalendar"}
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
            "prompt": "select_account consent",
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
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: adapter boundary converting Google API errors to error responses
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

    # -------------------------------------------------------------------------
    # Token Introspection for accurate scope resolution
    # -------------------------------------------------------------------------

    _TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

    async def introspect_token(
        self,
        *,
        access_token: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get token info from Google's tokeninfo endpoint.

        Google tokeninfo endpoint:
        GET https://oauth2.googleapis.com/tokeninfo?access_token=...

        Response (success):
        {
            "azp": "client_id",
            "aud": "client_id",
            "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",
            "exp": "1234567890",
            "access_type": "offline"
        }

        Note: Unlike RFC 7662 introspection, Google's tokeninfo doesn't require
        client credentials in the request - it validates the token itself.
        """
        _ = client_id  # Not needed for Google tokeninfo
        _ = client_secret  # Not needed for Google tokeninfo

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(
                    self._TOKENINFO_URL,
                    params={"access_token": access_token},
                    timeout=10.0,
                )

                if resp.status_code != 200:
                    logger.warning(
                        "Google tokeninfo failed: status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return None

                return resp.json()

        except httpx.RequestError as exc:
            logger.warning(
                "Google tokeninfo error: %s",
                exc,
                exc_info=True,
            )
            return None

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """
        Resolve granted scopes using tokeninfo endpoint.

        Falls back to token response scope or requested scope on failure.
        """
        access_token = token.get("access_token")
        if not access_token:
            logger.warning("No access_token in Google token response, falling back to requested scope")
            return state_data.get("requested_scope") or ""

        # Attempt tokeninfo lookup
        tokeninfo = await self.introspect_token(
            access_token=access_token,
            client_id="",  # Not needed for Google
            client_secret="",  # Not needed for Google
        )

        if tokeninfo and "scope" in tokeninfo:
            logger.info(
                "Google tokeninfo succeeded: scopes=%s",
                tokeninfo["scope"],
            )
            return tokeninfo["scope"]

        # Fallback: token response scope or requested scope
        logger.info("Google falling back to non-introspection scope resolution")
        return token.get("scope") or state_data.get("requested_scope") or ""
