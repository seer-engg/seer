from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException
from sqlmodel import select

from shared.config import config
from shared.database.models import User, OAuthConnection
from shared.database.base import async_session_maker
from shared.logger import get_logger

logger = get_logger("shared.tools.oauth_manager")


def _get_refresh_config(connection: OAuthConnection) -> tuple[str, dict]:
    """Get refresh URL and data for a given OAuth provider."""
    provider = connection.provider

    if provider in ["google", "googledrive", "gmail"]:
        if not config.GOOGLE_CLIENT_ID or not config.GOOGLE_CLIENT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="Google OAuth client credentials not configured"
            )
        return (
            "https://oauth2.googleapis.com/token",
            {
                "client_id": config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "refresh_token": connection.refresh_token_enc,
                "grant_type": "refresh_token",
            }
        )

    if provider == "github":
        if not config.GITHUB_CLIENT_ID or not config.GITHUB_CLIENT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="GitHub OAuth client credentials not configured"
            )
        return (
            "https://github.com/login/oauth/access_token",
            {
                "client_id": config.GITHUB_CLIENT_ID,
                "client_secret": config.GITHUB_CLIENT_SECRET,
                "refresh_token": connection.refresh_token_enc,
                "grant_type": "refresh_token",
            }
        )

    if provider in ["supabase", "supabase_mgmt"]:
        if not config.supabase_client_id or not config.supabase_client_secret:
            raise HTTPException(
                status_code=500,
                detail="Supabase OAuth client credentials not configured"
            )
        return (
            "https://api.supabase.com/v1/oauth/token",
            {
                "client_id": config.supabase_client_id,
                "client_secret": config.supabase_client_secret,
                "refresh_token": connection.refresh_token_enc,
                "grant_type": "refresh_token",
            }
        )

    raise HTTPException(
        status_code=400,
        detail=f"Token refresh not supported for provider: {provider}"
    )


async def refresh_oauth_token(connection: OAuthConnection) -> OAuthConnection:
    """
    Refresh an expired OAuth token using the stored refresh token.
    """

    if not connection.refresh_token_enc:
        raise HTTPException(
            status_code=401,
            detail=f"No refresh token available for connection {connection.id}",
        )

    logger.info(
        "Refreshing OAuth token",
        extra={"connection_id": connection.id, "provider": connection.provider}
    )

    refresh_url, refresh_data = _get_refresh_config(connection)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(refresh_url, data=refresh_data)
            response.raise_for_status()
            token_data = response.json()

        connection.access_token_enc = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        # Convert to timezone-naive for PostgreSQL TIMESTAMP WITHOUT TZ
        expires_at_utc = datetime.now(timezone.utc) + timedelta(
            seconds=expires_in
        )
        connection.expires_at = expires_at_utc.replace(tzinfo=None)
        if "scope" in token_data:
            connection.scopes = token_data["scope"]
        connection.updated_at = datetime.now(timezone.utc).replace(
            tzinfo=None
        )

        async with async_session_maker() as session:
            session.add(connection)
            await session.commit()
            await session.refresh(connection)

        logger.info("OAuth token refreshed", extra={"connection_id": connection.id})
        return connection
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Token refresh failed",
            extra={
                "connection_id": connection.id,
                "status_code": exc.response.status_code,
                "body": exc.response.text[:200]
            },
        )
        raise HTTPException(
            status_code=401,
            detail=f"Token refresh failed: {exc.response.text[:200]}"
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error refreshing token",
            extra={"connection_id": connection.id}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Token refresh error: {str(exc)}"
        )


async def get_oauth_token(
    user: User,
    connection_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[OAuthConnection, str]:
    """
    Resolve an OAuth connection for the user, refreshing tokens if expired.
    """
    async with async_session_maker() as session:
        if connection_id:
            if ":" in connection_id:
                _, db_id = connection_id.split(":", 1)
            else:
                db_id = connection_id
            try:
                stmt = select(OAuthConnection).where(
                    OAuthConnection.id == int(db_id),
                    OAuthConnection.user_id == user.id,
                    OAuthConnection.status == "active"
                )
                result = await session.execute(stmt)
                connection = result.scalar_one()
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail=f"OAuth connection {connection_id} not found"
                )
        elif provider:
            stmt = select(OAuthConnection).where(
                OAuthConnection.user_id == user.id,
                OAuthConnection.provider == provider,
                OAuthConnection.status == "active"
            )
            result = await session.execute(stmt)
            connection = result.scalar_one_or_none()
            if not connection:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active OAuth connection found for provider: {provider}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either connection_id or provider must be provided"
            )

        # Compare timezone-naive datetimes (DB uses TIMESTAMP WITHOUT TZ)
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        if connection.expires_at and connection.expires_at < now_naive:
            connection = await refresh_oauth_token(connection)

        if not connection.access_token_enc:
            raise HTTPException(
                status_code=401,
                detail=f"No access token available for connection {connection.id}",
            )

        return connection, connection.access_token_enc


__all__ = ["get_oauth_token", "refresh_oauth_token"]
