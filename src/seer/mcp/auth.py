"""
MCP Authentication middleware for token verification.

This module provides authentication for the MCP server HTTP transport:
- MCPAuthMiddleware: Validates Clerk session JWTs locally using JWKS
- MCPOpaqueAuthMiddleware: Validates Clerk OAuth opaque tokens via /oauth/userinfo endpoint
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from seer.auth.clerk_verifier import ClerkJWTVerifier, ClerkOpaqueTokenVerifier, VerifiedClerkToken
from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

# Context variable to propagate authenticated user to MCP tool handlers
mcp_authenticated_user: ContextVar[Optional[VerifiedClerkToken]] = ContextVar(
    "mcp_authenticated_user", default=None
)


def www_authenticate_response(request: Request, error_description: str, error: str = "invalid_token") -> JSONResponse:
    """
    Create a 401 response with WWW-Authenticate header per RFC 6750.
    Includes resource_metadata for OAuth discovery.

    Args:
        request: The incoming request (used to construct resource_metadata URL)
        error_description: Human-readable description of the error
        error: OAuth error code (invalid_token, invalid_request, insufficient_scope)

    Returns:
        JSONResponse with 401 status and WWW-Authenticate header
    """
    # Construct base URL from request (same pattern as oauth_protected_resource_metadata)
    scheme = config.redirect_uri_scheme
    resource_url = f"{scheme}://{request.url.netloc}"
    resource_metadata_url = f"{resource_url}/.well-known/oauth-protected-resource"

    www_auth_value = f'Bearer resource_metadata="{resource_metadata_url}", error="{error}", error_description="{error_description}"'

    return JSONResponse(
        status_code=401,
        content={
            "error": "authentication_required",
            "message": error_description,
            "_meta": {
                "mcp/www_authenticate": www_auth_value
            }
        },
        headers={
            "WWW-Authenticate": www_auth_value
        }
    )


def extract_bearer_token(request: Request) -> Optional[str]:
    """
    Extract Bearer token from Authorization header.

    Args:
        request: Starlette request object

    Returns:
        Token string if present, None otherwise
    """
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            return token
    return None


# Paths that don't require authentication
PUBLIC_PATHS = {
    "/.well-known/oauth-protected-resource",
}


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT authentication middleware for MCP HTTP transport.

    Validates Bearer tokens using Clerk JWKS and propagates the authenticated
    user to tool handlers via context variable.
    """

    def __init__(self, app, *, verifier: ClerkJWTVerifier) -> None:
        """
        Initialize the middleware.

        Args:
            app: The Starlette application
            verifier: ClerkJWTVerifier instance for token validation
        """
        super().__init__(app)
        self._verifier = verifier

    async def dispatch(self, request: Request, call_next):
        """Process request with JWT authentication."""
        # Skip authentication for OPTIONS (CORS preflight) and public paths
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Extract token from Authorization header
        token = extract_bearer_token(request)
        if not token:
            logger.debug("MCP request missing Authorization header: %s", path)
            return www_authenticate_response(request, "Missing or invalid Authorization header", "invalid_request")

        # Verify the token
        result, error = self._verifier.verify_token_with_error(token)
        if result is None:
            logger.debug("MCP token verification failed: %s", error)
            return www_authenticate_response(request, error or "Invalid token")

        # Propagate authenticated user to tool handlers via context variable
        ctx_token = mcp_authenticated_user.set(result)
        try:
            # Also attach to request state for middleware chain access
            request.state.mcp_user = result
            return await call_next(request)
        finally:
            mcp_authenticated_user.reset(ctx_token)


class MCPOpaqueAuthMiddleware(BaseHTTPMiddleware):
    """
    Opaque token authentication middleware for MCP HTTP transport.

    Validates Bearer tokens by calling Clerk's /oauth/userinfo endpoint.
    Used for MCP clients (like ChatGPT) that receive opaque OAuth access tokens
    rather than JWTs.

    This middleware follows the same pattern as MCPAuthMiddleware but uses
    async HTTP calls for token validation instead of local JWT verification.
    """

    def __init__(self, app, *, verifier: ClerkOpaqueTokenVerifier) -> None:
        """
        Initialize the middleware.

        Args:
            app: The Starlette application
            verifier: ClerkOpaqueTokenVerifier instance for token validation
        """
        super().__init__(app)
        self._verifier = verifier

    async def dispatch(self, request: Request, call_next):
        """Process request with opaque token authentication."""
        # Skip authentication for OPTIONS (CORS preflight) and public paths
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Extract token from Authorization header
        token = extract_bearer_token(request)
        if not token:
            logger.debug("MCP request missing Authorization header: %s", path)
            return www_authenticate_response(
                request, "Missing or invalid Authorization header", "invalid_request"
            )

        # Verify the token via userinfo endpoint (async)
        result, error = await self._verifier.verify_token_with_error(token)
        if result is None:
            logger.debug("MCP opaque token verification failed: %s", error)
            return www_authenticate_response(request, error or "Invalid token")

        # Propagate authenticated user to tool handlers via context variable
        ctx_token = mcp_authenticated_user.set(result)
        try:
            # Also attach to request state for middleware chain access
            request.state.mcp_user = result
            return await call_next(request)
        finally:
            mcp_authenticated_user.reset(ctx_token)


def get_mcp_authenticated_user() -> Optional[VerifiedClerkToken]:
    """
    Get the authenticated user from the current context.

    This function should be called from MCP tool handlers to access
    the authenticated user set by MCPAuthMiddleware.

    Returns:
        VerifiedClerkToken if authenticated, None otherwise
    """
    return mcp_authenticated_user.get()
