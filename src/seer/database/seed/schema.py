"""
Pydantic models for seed data serialization.

Uses natural keys (user_id, provider+account_id) instead of database IDs
to enable portability across environments.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class UserSeed(BaseModel):
    """User data for seeding."""

    user_id: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    claims: Optional[dict[str, Any]] = None
    signup_source: Optional[str] = None
    default_workflow_creation_mode: str = "ASK_FIRST"


class OAuthConnectionRef(BaseModel):
    """Reference to identify an OAuth connection by natural key."""

    provider: str
    provider_account_id: str


class OAuthConnectionSeed(BaseModel):
    """OAuth connection data for seeding."""

    user_id: str
    provider: str
    provider_account_id: str
    access_token_enc: str
    refresh_token_enc: Optional[str] = None
    expires_at: Optional[datetime] = None
    scopes: Optional[str] = None
    token_type: str = "Bearer"
    provider_metadata: Optional[dict[str, Any]] = None
    status: str = "active"


class IntegrationResourceRef(BaseModel):
    """Reference to identify a resource by natural key."""

    provider: str
    resource_type: str
    resource_id: str


class IntegrationResourceSeed(BaseModel):
    """Integration resource data for seeding."""

    user_id: str
    oauth_connection_ref: Optional[OAuthConnectionRef] = None
    provider: str
    resource_type: str
    resource_id: str
    resource_key: Optional[str] = None
    name: Optional[str] = None
    resource_metadata: Optional[dict[str, Any]] = None
    status: str = "active"


class IntegrationSecretSeed(BaseModel):
    """Integration secret data for seeding."""

    user_id: str
    provider: str
    oauth_connection_ref: Optional[OAuthConnectionRef] = None
    resource_ref: Optional[IntegrationResourceRef] = None
    secret_type: str
    name: str
    value_enc: str
    value_fingerprint: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    expires_at: Optional[datetime] = None
    status: str = "active"


class SeedTables(BaseModel):
    """Container for all table data."""

    users: list[UserSeed] = Field(default_factory=list)
    oauth_connections: list[OAuthConnectionSeed] = Field(default_factory=list)
    integration_resources: list[IntegrationResourceSeed] = Field(default_factory=list)
    integration_secrets: list[IntegrationSecretSeed] = Field(default_factory=list)


class SeedData(BaseModel):
    """Complete seed data structure."""

    version: str = "1.0"
    exported_at: datetime = Field(default_factory=_utcnow)
    source_environment: Optional[str] = None
    tables: SeedTables = Field(default_factory=SeedTables)
