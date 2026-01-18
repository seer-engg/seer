from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import HTTPException

from seer.services.integrations.providers.base import IntegrationProvider, OAuthAuthorizeContext
from seer.logger import get_logger

logger = get_logger(__name__)


class GitHubProvider(IntegrationProvider):
    provider = "github"

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        return " ".join(context.requested_scopes)

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        access_token = token.get("access_token")
        if not access_token:
            logger.error("GitHub token missing access_token. keys=%s", list(token.keys()))
            raise HTTPException(
                status_code=500,
                detail="No access token in OAuth response. Check GitHub OAuth configuration.",
            )

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {access_token}"},
                timeout=10.0,
            )
        if resp.status_code != 200:
            logger.error(
                "GitHub userinfo request failed status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch GitHub user profile: HTTP {resp.status_code}",
            )
        return resp.json()
