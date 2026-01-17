"""OAuth 2.1 PKCE endpoints for MCP clients."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from api.core.pkce import validate_code_verifier, validate_pkce_parameters
from api.oauth.models import RefreshRequest, RevokeRequest, TokenRequest, TokenResponse
from shared.config import config
from shared.database import OAuthAuthorizationCode, OAuthRefreshToken, User
from shared.logger import get_logger

logger = get_logger("api.oauth")

router = APIRouter()

# JWT configuration for MCP tokens
JWT_SECRET = config.clerk_secret_key or secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRY = 3600  # 1 hour
REFRESH_TOKEN_EXPIRY = 2592000  # 30 days
AUTH_CODE_EXPIRY = 600  # 10 minutes


def create_access_token(user: User, scope: str) -> str:
    """Create a JWT access token for MCP client."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ACCESS_TOKEN_EXPIRY)

    payload = {
        "sub": user.user_id,
        "user_id": user.user_id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "scope": scope,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": config.API_URL or "https://api.getseer.dev",
        "aud": "seer-mcp-client",
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@router.get("/authorize")
async def authorize(  # pylint: disable=too-many-positional-arguments  # Reason: OAuth parameters are standard query params
    code_challenge: Annotated[str, Query(description="PKCE code challenge")],
    redirect_uri: Annotated[str, Query(description="Client redirect URI")],
    code_challenge_method: Annotated[str, Query(description="PKCE challenge method")] = "S256",
    client_id: Annotated[str, Query(description="OAuth client ID")] = "seer-mcp-client",
    scope: Annotated[str, Query(description="Requested scopes")] = "workflow:read workflow:write workflow:execute integration:read integration:write",
    state: Annotated[str | None, Query(description="Client state for CSRF protection")] = None,
):
    """Start OAuth 2.1 authorization flow with PKCE.

    This endpoint validates PKCE parameters and redirects to the frontend authorization page.
    The frontend will show the user what permissions are being requested and allow them
    to authorize the MCP client.

    Args:
        code_challenge: PKCE code challenge (base64url-encoded SHA256 of code_verifier)
        code_challenge_method: Challenge method (must be "S256")
        redirect_uri: Where to redirect after authorization
        client_id: OAuth client identifier
        scope: Requested OAuth scopes (space-separated)
        state: Optional client state for CSRF protection

    Returns:
        RedirectResponse: Redirect to frontend OAuth authorization page
    """
    # Validate PKCE parameters
    valid, error = validate_pkce_parameters(code_challenge, code_challenge_method)
    if not valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)

    # Build authorization URL with query parameters
    frontend_url = config.FRONTEND_URL or "https://app.getseer.dev"
    auth_url = f"{frontend_url}/oauth/authorize"

    # Pass all parameters to frontend
    params = {
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "scope": scope,
    }
    if state:
        params["state"] = state

    # Build query string
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    redirect_url = f"{auth_url}?{query_string}"

    logger.info("OAuth authorization requested - redirecting to frontend", extra={
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
    })

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)


@router.post("/complete-authorization")
async def complete_authorization(  # pylint: disable=too-many-positional-arguments  # Reason: OAuth flow parameters
    user_id: str,
    code_challenge: str,
    code_challenge_method: str,
    redirect_uri: str,
    client_id: str,
    scope: str,
):
    """Complete authorization after user approval (called by frontend).

    This is an internal endpoint called by the Seer frontend after the user
    approves the authorization request. It creates an authorization code
    that the MCP client can exchange for tokens.

    Args:
        user_id: Clerk user ID who approved the request
        code_challenge: PKCE code challenge
        code_challenge_method: PKCE challenge method
        redirect_uri: Client redirect URI
        client_id: OAuth client ID
        scope: Approved scopes

    Returns:
        dict: Contains authorization code to send to client
    """
    # Get or create user
    user = await User.get_or_none(user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Generate authorization code
    auth_code = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=AUTH_CODE_EXPIRY)

    # Store authorization code
    await OAuthAuthorizationCode.create(
        code=auth_code,
        user=user,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
        expires_at=expires_at,
    )

    logger.info("Authorization code created", extra={
        "user_id": user_id,
        "client_id": client_id,
        "expires_at": expires_at.isoformat(),
    })

    return {
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }


@router.post("/token", response_model=TokenResponse)
async def token(request: TokenRequest):
    """Exchange authorization code for access and refresh tokens.

    This endpoint validates the PKCE code_verifier against the stored code_challenge
    and issues JWT access tokens and refresh tokens.

    Args:
        request: Token exchange request with code and code_verifier

    Returns:
        TokenResponse: Access token, refresh token, and metadata
    """
    # Find authorization code
    auth_code = await OAuthAuthorizationCode.get_or_none(
        code=request.code,
        client_id=request.client_id,
    ).prefetch_related("user")

    if not auth_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization code"
        )

    # Check if code is expired or already used
    now = datetime.now(timezone.utc)
    if auth_code.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code expired"
        )

    if auth_code.used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code already used"
        )

    # Validate redirect_uri matches
    if auth_code.redirect_uri != request.redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="redirect_uri mismatch"
        )

    # Validate PKCE code_verifier
    if not validate_code_verifier(
        request.code_verifier,
        auth_code.code_challenge,
        auth_code.code_challenge_method,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid code_verifier"
        )

    # Mark code as used
    auth_code.used = True
    await auth_code.save()

    # Create access token
    access_token = create_access_token(auth_code.user, auth_code.scope)

    # Create refresh token
    refresh_token = secrets.token_urlsafe(32)
    refresh_expires_at = now + timedelta(seconds=REFRESH_TOKEN_EXPIRY)

    await OAuthRefreshToken.create(
        token=refresh_token,
        user=auth_code.user,
        client_id=request.client_id,
        scope=auth_code.scope,
        expires_at=refresh_expires_at,
    )

    logger.info("Tokens issued", extra={
        "user_id": auth_code.user.user_id,
        "client_id": request.client_id,
        "scope": auth_code.scope,
    })

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=ACCESS_TOKEN_EXPIRY,
        refresh_token=refresh_token,
        scope=auth_code.scope,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest):
    """Refresh an access token using a refresh token.

    Args:
        request: Refresh request with refresh_token

    Returns:
        TokenResponse: New access token and same refresh token
    """
    # Find refresh token
    refresh_token_obj = await OAuthRefreshToken.get_or_none(
        token=request.refresh_token,
        client_id=request.client_id,
    ).prefetch_related("user")

    if not refresh_token_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token"
        )

    # Check if token is expired or revoked
    now = datetime.now(timezone.utc)
    if refresh_token_obj.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token expired"
        )

    if refresh_token_obj.revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token revoked"
        )

    # Update last_used_at
    refresh_token_obj.last_used_at = now
    await refresh_token_obj.save()

    # Create new access token
    access_token = create_access_token(refresh_token_obj.user, refresh_token_obj.scope)

    logger.info("Access token refreshed", extra={
        "user_id": refresh_token_obj.user.user_id,
        "client_id": request.client_id,
    })

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=ACCESS_TOKEN_EXPIRY,
        refresh_token=request.refresh_token,
        scope=refresh_token_obj.scope,
    )


@router.post("/revoke", status_code=status.HTTP_200_OK)
async def revoke(request: RevokeRequest):
    """Revoke an access or refresh token.

    Args:
        request: Revoke request with token to revoke

    Returns:
        dict: Success message
    """
    # Try to find as refresh token
    refresh_token_obj = await OAuthRefreshToken.get_or_none(
        token=request.token,
        client_id=request.client_id,
    )

    if refresh_token_obj:
        refresh_token_obj.revoked = True
        await refresh_token_obj.save()
        logger.info("Refresh token revoked", extra={
            "client_id": request.client_id,
        })
        return {"message": "Token revoked successfully"}

    # If not a refresh token, it might be an access token (JWT)
    # For JWTs, we can't revoke them server-side (stateless)
    # But we can log the revocation attempt
    try:
        payload = jwt.decode(
            request.token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"verify_signature": True, "verify_exp": False}
        )
        if payload.get("aud") == request.client_id:
            logger.info("Access token revocation requested (stateless JWT)", extra={
                "user_id": payload.get("user_id"),
                "client_id": request.client_id,
            })
            return {"message": "Access token revocation noted (JWT tokens cannot be revoked server-side)"}
    except jwt.InvalidTokenError:
        pass

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid token or token not found"
    )
