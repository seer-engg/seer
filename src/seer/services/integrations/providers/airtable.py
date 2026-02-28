"""
Airtable integration provider.

Implements OAuth 2.0 with PKCE (Proof Key for Code Exchange) as required by Airtable.
PKCE is handled automatically by Authlib when code_challenge_method='S256' is set
in the client_kwargs during registration (see oauth.py).

Airtable access tokens expire in 60 minutes and refresh tokens expire in 60 days.
"""
from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)

# Airtable API base URL
AIRTABLE_API_BASE = "https://api.airtable.com/v0"


class AirtableProvider(IntegrationProvider):
    """
    Airtable OAuth provider.

    Airtable requires PKCE (Proof Key for Code Exchange) for all OAuth flows.
    PKCE is handled automatically by Authlib via code_challenge_method='S256'
    configured in oauth.py. This provider handles user profile fetch from
    the /meta/whoami endpoint.
    """

    provider = "airtable"
    resource_types = {"base", "table"}

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        Format scopes for Airtable OAuth request.

        Airtable uses space-separated scopes.

        Args:
            context: OAuth authorization context with requested scopes

        Returns:
            Space-separated scope string
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
        Build authorization request parameters.

        PKCE is handled automatically by Authlib, so we only need to
        pass through the scope. State is also handled by Authlib.

        Args:
            context: OAuth authorization context
            state: Encoded state parameter (handled by Authlib, not used here)
            scope: Formatted scope string

        Returns:
            Dict with OAuth parameters
        """
        # Authlib handles PKCE and state automatically
        return {"scope": scope}

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """
        Extract granted scopes from Airtable token response.

        Airtable returns scope in the token response.

        Args:
            token: Token response from Airtable
            state_data: State data from OAuth flow

        Returns:
            Space-separated string of granted scopes
        """
        # Airtable returns 'scope' in the token response
        return token.get("scope") or state_data.get("requested_scope") or ""

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fetch user profile from Airtable /meta/whoami endpoint.

        Args:
            client: OAuth client (unused, we make direct HTTP call)
            token: Token response containing access_token
            state_data: State data from OAuth flow

        Returns:
            User profile dict with 'id' field

        Raises:
            HTTPException: If profile fetch fails
        """
        _ = client  # Not used, we make direct HTTP call
        _ = state_data

        access_token = token.get("access_token")
        if not access_token:
            logger.error("Airtable token missing access_token. keys=%s", list(token.keys()))
            raise HTTPException(
                status_code=500,
                detail="No access token in OAuth response. Check Airtable OAuth configuration.",
            )

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                f"{AIRTABLE_API_BASE}/meta/whoami",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )

        if resp.status_code != 200:
            logger.error(
                "Airtable whoami request failed status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Airtable user profile: HTTP {resp.status_code}",
            )

        profile = resp.json()
        logger.info(
            "Fetched Airtable profile: id=%s, scopes=%s",
            profile.get("id"),
            profile.get("scopes", [])[:3],
        )
        return profile
