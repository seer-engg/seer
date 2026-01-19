"""OAuth 2.1 Well-Known Endpoints (RFC 8414 & RFC 9728).

This module implements the well-known endpoints required for MCP OAuth compliance:
- RFC 8414: OAuth 2.0 Authorization Server Metadata
- RFC 9728: OAuth 2.0 Protected Resource Metadata
"""

from fastapi import APIRouter

from shared.config import config

router = APIRouter()


@router.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    """OAuth 2.0 Authorization Server Metadata (RFC 8414).

    This endpoint provides metadata about the OAuth authorization server,
    including supported grant types, response types, and PKCE methods.

    Returns:
        dict: Authorization server metadata
    """
    return {
        "issuer": config.FRONTEND_URL or "https://app.getseer.dev",
        "authorization_endpoint": f"{config.FRONTEND_URL or 'https://app.getseer.dev'}/oauth/authorize",
        "token_endpoint": f"{config.API_URL or 'https://api.getseer.dev'}/oauth/token",
        "jwks_uri": f"{config.API_URL or 'https://api.getseer.dev'}/.well-known/jwks.json",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],  # PKCE with SHA-256
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
        ],
        "scopes_supported": [
            "workflow:read",
            "workflow:write",
            "workflow:execute",
            "integration:read",
            "integration:write",
        ],
        "service_documentation": "https://github.com/seer-engg/seer",
    }


@router.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata():
    """OAuth 2.0 Protected Resource Metadata (RFC 9728).

    This endpoint provides metadata about the protected resource (Seer API),
    including which authorization servers it accepts tokens from.

    Returns:
        dict: Protected resource metadata
    """
    return {
        "resource": config.API_URL or "https://api.getseer.dev",
        "authorization_servers": [
            config.FRONTEND_URL or "https://app.getseer.dev",
        ],
        "scopes_supported": [
            "workflow:read",
            "workflow:write",
            "workflow:execute",
            "integration:read",
            "integration:write",
        ],
        "bearer_methods_supported": ["header"],
        "resource_signing_alg_values_supported": ["RS256"],
    }
