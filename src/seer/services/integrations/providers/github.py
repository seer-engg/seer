from __future__ import annotations

import base64
from typing import Any, Dict, Optional

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

    # -------------------------------------------------------------------------
    # Token Introspection for accurate scope resolution
    # -------------------------------------------------------------------------

    _CHECK_TOKEN_URL = "https://api.github.com/applications/{client_id}/token"

    async def introspect_token(
        self,
        *,
        access_token: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check token validity and scopes via GitHub's OAuth App token endpoint.

        GitHub token check endpoint:
        POST https://api.github.com/applications/{client_id}/token
        Authorization: Basic base64(client_id:client_secret)
        Content-Type: application/json
        Body: {"access_token": "..."}

        Response (success):
        {
            "id": 1,
            "token": "gho_xxx",
            "scopes": ["repo", "user:email"],
            "user": {"login": "octocat", "id": 1}
        }

        Note: GitHub returns scopes as an ARRAY, not a space-separated string.
        """
        if not client_id or not client_secret:
            logger.warning("GitHub introspection requires client_id and client_secret")
            return None

        try:
            # Build Basic Auth header
            credentials = f"{client_id}:{client_secret}"
            auth_header = base64.b64encode(credentials.encode()).decode()

            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    self._CHECK_TOKEN_URL.format(client_id=client_id),
                    headers={
                        "Authorization": f"Basic {auth_header}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json={"access_token": access_token},
                    timeout=10.0,
                )

                if resp.status_code == 404:
                    # Token is invalid or revoked
                    logger.warning("GitHub token check returned 404 - token may be invalid or revoked")
                    return None

                if resp.status_code != 200:
                    logger.warning(
                        "GitHub token check failed: status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return None

                return resp.json()

        except httpx.RequestError as exc:
            logger.warning(
                "GitHub token check error: %s",
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
        Resolve granted scopes using GitHub's token check endpoint.

        Falls back to token response scope or requested scope on failure.
        """
        # pylint: disable=import-outside-toplevel
        # Reason: Avoids circular import - config depends on modules that import providers
        from seer.config import config

        access_token = token.get("access_token")
        if not access_token:
            logger.warning("No access_token in GitHub token response, falling back to requested scope")
            return state_data.get("requested_scope") or ""

        # Attempt token check if credentials are available
        if config.github_client_id and config.github_client_secret:
            token_info = await self.introspect_token(
                access_token=access_token,
                client_id=config.github_client_id,
                client_secret=config.github_client_secret,
            )

            if token_info and "scopes" in token_info:
                # GitHub returns scopes as an array, join with space for storage
                scopes = token_info["scopes"]
                if isinstance(scopes, list):
                    scope_str = " ".join(scopes)
                    logger.info(
                        "GitHub token check succeeded: scopes=%s",
                        scope_str,
                    )
                    return scope_str

        # Fallback: token response scope or requested scope
        logger.info("GitHub falling back to non-introspection scope resolution")
        return token.get("scope") or state_data.get("requested_scope") or ""
