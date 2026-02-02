from __future__ import annotations

from typing import Any, Dict

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

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        LinkedIn scopes are space-separated.

        Common scopes:
        - openid: Required for OpenID Connect
        - profile: Access to basic profile info
        - email: Access to email address
        - w_member_social: Share content on behalf of user
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
