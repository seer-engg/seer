from pydantic import BaseModel, Field
from typing import Optional, List



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
