"""Pydantic models for OAuth 2.1 MCP endpoints."""

from pydantic import BaseModel, Field


class AuthorizeRequest(BaseModel):
    """OAuth authorization request parameters."""

    code_challenge: str = Field(..., min_length=43, max_length=128, description="PKCE code challenge")
    code_challenge_method: str = Field(default="S256", description="PKCE challenge method")
    redirect_uri: str = Field(..., description="Client redirect URI")
    client_id: str = Field(default="seer-mcp-client", description="OAuth client identifier")
    scope: str = Field(default="workflow:read workflow:write workflow:execute integration:read integration:write", description="Requested scopes")
    state: str | None = Field(default=None, description="Client state for CSRF protection")


class TokenRequest(BaseModel):
    """OAuth token exchange request."""

    code: str = Field(..., description="Authorization code from authorize endpoint")
    code_verifier: str = Field(..., min_length=43, max_length=128, description="PKCE code verifier")
    redirect_uri: str = Field(..., description="Must match authorization redirect_uri")
    client_id: str = Field(default="seer-mcp-client", description="OAuth client identifier")


class TokenResponse(BaseModel):
    """OAuth token response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    refresh_token: str = Field(..., description="Refresh token for getting new access tokens")
    scope: str = Field(..., description="Granted scopes")


class RefreshRequest(BaseModel):
    """OAuth token refresh request."""

    refresh_token: str = Field(..., description="Refresh token from token endpoint")
    client_id: str = Field(default="seer-mcp-client", description="OAuth client identifier")


class RevokeRequest(BaseModel):
    """OAuth token revocation request."""

    token: str = Field(..., description="Access or refresh token to revoke")
    client_id: str = Field(default="seer-mcp-client", description="OAuth client identifier")
