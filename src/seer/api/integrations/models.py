from typing import Optional, List

from pydantic import BaseModel, Field



class SupabaseBindRequest(BaseModel):
    project_ref: str = Field(..., min_length=3, description="Supabase project reference")
    connection_id: Optional[str] = Field(
        default=None,
        description="Specific Supabase OAuth connection ID (optional)",
    )


class SupabaseManualBindRequest(BaseModel):
    project_ref: str = Field(..., min_length=3, description="Supabase project reference")
    connection_id: Optional[str] = Field(
        default=None,
        description="Existing Supabase OAuth connection ID (skips manual secret input)",
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Friendly project display name",
    )
    service_role_key: Optional[str] = Field(
        default=None,
        description="Supabase service role key (required without connection_id)",
        min_length=8,
    )
    anon_key: Optional[str] = Field(
        default=None,
        description="Optional Supabase anon/public key",
    )




class ToolStatus(BaseModel):
    tool_name: str
    integration_type: Optional[str]
    provider: Optional[str]
    supports_oauth: bool
    supports_manual_secrets: bool
    connected: bool
    missing_scopes: List[str] = Field(default_factory=list)
    connection_id: Optional[str] = None
    provider_account_id: Optional[str] = None


class ToolsStatusResponse(BaseModel):
    tools: List[ToolStatus]


# Multi-account OAuth models

class AccountInfo(BaseModel):
    """Information about a connected OAuth account."""
    id: int = Field(..., description="Database ID of the OAuth connection")
    provider: str = Field(..., description="OAuth provider (e.g., 'google', 'github')")
    provider_account_id: str = Field(..., description="Provider's internal account identifier")
    display_name: str = Field(..., description="Human-readable account name (email, username)")
    status: str = Field(default="active", description="Connection status")
    scopes: Optional[str] = Field(default=None, description="Granted OAuth scopes")


class ConnectionsByProvider(BaseModel):
    """Connections grouped by OAuth provider."""
    connections: dict = Field(
        default_factory=dict,
        description="Dict mapping provider -> list of AccountInfo"
    )


class ToolAccountInfo(BaseModel):
    """Account information for a specific tool."""
    id: int = Field(..., description="Database ID of the OAuth connection")
    provider_account_id: str = Field(..., description="Provider's internal account identifier")
    display_name: str = Field(..., description="Human-readable account name (email, username)")
    has_required_scopes: bool = Field(..., description="Whether account has all required scopes")
    missing_scopes: List[str] = Field(default_factory=list, description="List of missing scopes")


class ToolAccountsResponse(BaseModel):
    """Response for tool accounts endpoint."""
    tool_name: str
    provider: Optional[str]
    accounts: List[ToolAccountInfo] = Field(default_factory=list)
    requires_selection: bool = Field(
        ...,
        description="True if user must select an account (multiple available)"
    )
