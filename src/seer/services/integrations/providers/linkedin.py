from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class LinkedInProvider(IntegrationProvider):
    """
    LinkedIn integration provider.

    Supports OAuth 2.0 authentication for accessing LinkedIn APIs.
    Common scopes: openid, profile, email, w_member_social (post content)
    """
    provider = "linkedin"

    # Always require OpenID Connect scopes for user identification via /v2/userinfo endpoint.
    # LinkedIn requires openid + at least one of (profile, email) for the userinfo endpoint.
    # Without these, the userinfo endpoint returns 403 and OAuth callback fails.
    _required_openid_scopes = ["openid", "profile"]

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        LinkedIn scopes are space-separated.
        Always includes 'openid' scope for user identification.

        Common scopes:
        - openid: Required for OpenID Connect and /v2/userinfo endpoint
        - profile: Access to basic profile info
        - email: Access to email address
        - w_member_social: Share content on behalf of user
        """
        # Preserve order and remove duplicates
        scopes: List[str] = list(dict.fromkeys(context.requested_scopes))
        # Ensure required scopes are always present
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
        """
        Build authorization kwargs for LinkedIn OAuth.

        Note: LinkedIn refresh tokens are only available to approved Marketing
        Developer Platform (MDP) partners. For non-MDP apps:
        - Access tokens last 60 days (vs 1 hour for Google)
        - Users must re-authenticate when tokens expire
        - No authorization URL parameters can request refresh tokens

        This implementation follows Google's pattern for consistency and
        future-proofing if MDP partnership is obtained.
        """
        return {
            "state": state,
            "scope": scope,
        }

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fetch user profile from LinkedIn API.

        LinkedIn v2 API uses /userinfo endpoint for OpenID Connect.
        Returns profile with sub (user ID), name, email, and picture.
        """
        access_token = token.get("access_token")
        if not access_token:
            logger.error("LinkedIn token missing access_token. keys=%s", list(token.keys()))
            raise HTTPException(
                status_code=500,
                detail="No access token in OAuth response. Check LinkedIn OAuth configuration.",
            )

        async with httpx.AsyncClient() as http_client:
            # LinkedIn v2 uses /userinfo endpoint for OpenID Connect
            resp = await http_client.get(
                "https://api.linkedin.com/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )

        if resp.status_code != 200:
            logger.error(
                "LinkedIn userinfo request failed status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch LinkedIn user profile: HTTP {resp.status_code}",
            )

        return resp.json()

    # -------------------------------------------------------------------------
    # Token Introspection for accurate scope resolution
    # -------------------------------------------------------------------------

    _INTROSPECT_URL = "https://www.linkedin.com/oauth/v2/introspectToken"

    async def introspect_token(
        self,
        *,
        access_token: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Introspect LinkedIn access token to get actual granted scopes.

        LinkedIn introspection endpoint:
        POST https://www.linkedin.com/oauth/v2/introspectToken
        Content-Type: application/x-www-form-urlencoded

        Body: client_id=...&client_secret=...&token=...

        Response (success):
        {
            "active": true,
            "scope": "openid profile email w_member_social",
            "client_id": "...",
            "exp": 1234567890,
            ...
        }
        """
        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    self._INTROSPECT_URL,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "token": access_token,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10.0,
                )

                if resp.status_code != 200:
                    logger.warning(
                        "LinkedIn token introspection failed: status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return None

                data = resp.json()

                # Validate token is active
                if not data.get("active", False):
                    logger.warning("LinkedIn token introspection returned inactive token")
                    return None

                return data

        except httpx.RequestError as exc:
            logger.warning(
                "LinkedIn token introspection error: %s",
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
        Resolve granted scopes using token introspection.

        Falls back to token response scope or requested scope on failure.
        """
        # pylint: disable=import-outside-toplevel
        # Reason: Avoids circular import - config depends on modules that import providers
        from seer.config import config

        access_token = token.get("access_token")
        if not access_token:
            logger.warning("No access_token in LinkedIn token response, falling back to requested scope")
            return state_data.get("requested_scope") or ""

        # Attempt introspection if credentials are available
        if config.linkedin_client_id and config.linkedin_client_secret:
            introspection = await self.introspect_token(
                access_token=access_token,
                client_id=config.linkedin_client_id,
                client_secret=config.linkedin_client_secret,
            )

            if introspection and "scope" in introspection:
                logger.info(
                    "LinkedIn introspection succeeded: scopes=%s",
                    introspection["scope"],
                )
                return introspection["scope"]

        # Fallback: token response scope or requested scope
        logger.info("LinkedIn falling back to non-introspection scope resolution")
        return token.get("scope") or state_data.get("requested_scope") or ""
