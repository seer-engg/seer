"""
Notion integration provider.

Notion OAuth 2.0 does not use URL-based scopes. Permissions are configured as
"capabilities" in the Notion dashboard when creating the integration. The
owner=user parameter is required in the authorize URL.

User profile data is embedded directly in the token response under owner.user,
so no separate profile API call is needed.
"""
from __future__ import annotations

from typing import Any, Dict

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class NotionProvider(IntegrationProvider):
    """
    Notion OAuth provider.

    Notion's OAuth flow differs from standard providers:
    - No URL scopes (permissions set as "capabilities" in Notion dashboard)
    - owner=user is required in the authorize URL
    - User profile is embedded in the token response (no separate API call needed)
    - Token endpoint requires Basic Auth (client_secret_basic)
    """

    provider = "notion"

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """
        Notion does not use URL-based scopes.

        Args:
            context: OAuth authorization context

        Returns:
            Empty string - Notion capabilities are configured in dashboard
        """
        return ""

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        """
        Build authorization request parameters for Notion.

        Notion requires owner=user to initiate a user-level OAuth flow.
        State is handled by Authlib.

        Args:
            context: OAuth authorization context
            state: Encoded state parameter (handled by Authlib)
            scope: Formatted scope string (unused for Notion)

        Returns:
            Dict with OAuth parameters
        """
        return {"state": state, "owner": "user"}

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """
        Resolve granted scopes for Notion.

        Notion doesn't return URL scopes in the token response since
        capabilities are dashboard-configured.

        Args:
            token: Token response from Notion
            state_data: State data from OAuth flow

        Returns:
            Empty string - Notion uses capability-based permissions
        """
        return ""

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract user profile from the Notion token response.

        Notion embeds user info under token['owner']['user'], so no
        additional API call is needed.

        Args:
            client: OAuth client (unused - profile is in token)
            token: Token response containing owner.user data
            state_data: State data from OAuth flow

        Returns:
            User profile dict with id, name, email, avatar_url, workspace info
        """
        _ = client  # Profile is embedded in token response
        _ = state_data

        owner = token.get("owner", {})
        user = owner.get("user", {})

        user_id = user.get("id") or token.get("bot_id", "")
        logger.info(
            "Fetched Notion user profile: id=%s, workspace=%s",
            user_id,
            token.get("workspace_name", ""),
        )

        return {
            "id": user_id,
            "name": user.get("name", ""),
            "email": user.get("person", {}).get("email", ""),
            "avatar_url": user.get("avatar_url", ""),
            "workspace_id": token.get("workspace_id", ""),
            "workspace_name": token.get("workspace_name", ""),
        }
