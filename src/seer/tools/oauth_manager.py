from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException

from seer.config import config
from seer.database import OAuthConnection, User
from seer.logger import get_logger

logger = get_logger("shared.tools.oauth_manager")


# pylint: disable=too-complex
# Reason: OAuth token refresh handles multiple providers with different auth flows and error cases
async def refresh_oauth_token(connection: OAuthConnection) -> OAuthConnection:
    """
    Refresh an expired OAuth token using the stored refresh token.
    """

    if not connection.refresh_token_enc:
        raise HTTPException(
            status_code=401,
            detail=f"No refresh token available for connection {connection.id}",
        )

    logger.info("Refreshing OAuth token", extra={"connection_id": connection.id, "provider": connection.provider})

    if connection.provider in ["google", "googledrive", "gmail"]:
        if not config.google_client_id or not config.google_client_secret:
            raise HTTPException(status_code=500, detail="Google OAuth client credentials not configured")
        refresh_url = "https://oauth2.googleapis.com/token"
        refresh_data = {
            "client_id": config.google_client_id,
            "client_secret": config.google_client_secret,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token",
        }
    elif connection.provider == "github":
        if not config.github_client_id or not config.github_client_secret:
            raise HTTPException(status_code=500, detail="GitHub OAuth client credentials not configured")
        refresh_url = "https://github.com/login/oauth/access_token"
        refresh_data = {
            "client_id": config.github_client_id,
            "client_secret": config.github_client_secret,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token",
        }
    elif connection.provider in ["supabase", "supabase_mgmt"]:
        if not config.supabase_client_id or not config.supabase_client_secret:
            raise HTTPException(status_code=500, detail="Supabase OAuth client credentials not configured")
        refresh_url = "https://api.supabase.com/v1/oauth/token"
        refresh_data = {
            "client_id": config.supabase_client_id,
            "client_secret": config.supabase_client_secret,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Token refresh not supported for provider: {connection.provider}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(refresh_url, data=refresh_data)
            response.raise_for_status()
            token_data = response.json()

        connection.access_token_enc = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        connection.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        if "scope" in token_data:
            connection.scopes = token_data["scope"]
        connection.updated_at = datetime.now(timezone.utc)
        await connection.save()
        logger.info("OAuth token refreshed", extra={"connection_id": connection.id})
        return connection
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Token refresh failed",
            extra={"connection_id": connection.id, "status_code": exc.response.status_code, "body": exc.response.text[:200]},
        )
        raise HTTPException(status_code=401, detail=f"Token refresh failed: {exc.response.text[:200]}") from exc
    except Exception as exc:
        logger.exception("Unexpected error refreshing token", extra={"connection_id": connection.id})
        raise HTTPException(status_code=500, detail=f"Token refresh error: {str(exc)}") from exc


async def get_oauth_token(
    user: User,
    connection_id: Optional[str] = None,
    provider: Optional[str] = None,
    organization_id: Optional[int] = None,
) -> Tuple[OAuthConnection, str]:
    """
    Resolve an OAuth connection for the user, refreshing tokens if expired.

    Args:
        user: The user requesting the OAuth token
        connection_id: Optional connection ID (format: "provider:id" or "id")
        provider: Optional provider name to look up by provider
        organization_id: Optional organization ID to include shared connections
    """
    from tortoise.expressions import Q  # pylint: disable=import-outside-toplevel  # Reason: avoid import overhead when not needed

    if connection_id:
        if ":" in connection_id:
            _, db_id = connection_id.split(":", 1)
        else:
            db_id = connection_id

        # Build query for user's connection OR organization-shared connection
        query = Q(id=int(db_id), status="active")
        if organization_id:
            query &= (Q(user=user) | Q(shared_with_organization_id=organization_id))
        else:
            query &= Q(user=user)

        connection = await OAuthConnection.filter(query).first()
        if not connection:
            raise HTTPException(status_code=404, detail=f"OAuth connection {connection_id} not found")
    elif provider:
        # Build query for user's connection OR organization-shared connection
        query = Q(provider=provider, status="active")
        if organization_id:
            query &= (Q(user=user) | Q(shared_with_organization_id=organization_id))
        else:
            query &= Q(user=user)

        connection = await OAuthConnection.filter(query).first()
        if not connection:
            raise HTTPException(
                status_code=404,
                detail=f"No active OAuth connection found for provider: {provider}",
            )
    else:
        raise HTTPException(status_code=400, detail="Either connection_id or provider must be provided")

    if connection.expires_at and connection.expires_at < datetime.now(timezone.utc):
        if not connection.refresh_token_enc:
            raise HTTPException(
                status_code=401,
                detail=f"Access token expired and no refresh token available. "
                       f"Please reconnect your {connection.provider} account.",
            )
        connection = await refresh_oauth_token(connection)

    if not connection.access_token_enc:
        raise HTTPException(
            status_code=401,
            detail=f"No access token available for connection {connection.id}",
        )

    return connection, connection.access_token_enc


__all__ = ["get_oauth_token", "refresh_oauth_token"]
