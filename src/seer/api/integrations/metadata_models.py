"""
Pydantic models for integration metadata API.

These models define the response schema for GET /api/integrations/metadata
which provides the frontend with all information needed to render integrations
dynamically without hardcoded configurations.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IntegrationIcon(BaseModel):
    """Icon configuration for an integration."""
    type: str = Field(..., description="Icon type: 'url', 'lucide', or 'svg'")
    value: str = Field(..., description="Icon URL, Lucide icon name, or SVG string")


class IntegrationScope(BaseModel):
    """OAuth scope with display metadata."""
    value: str = Field(..., description="OAuth scope value (e.g., 'https://www.googleapis.com/auth/gmail.send')")
    display_name: str = Field(..., description="Human-readable scope name")
    description: Optional[str] = Field(None, description="Detailed description of what the scope allows")


class DetectionPatterns(BaseModel):
    """Patterns used to detect which integration a tool belongs to."""
    tool_name_patterns: List[str] = Field(default_factory=list, description="Patterns to match tool names")
    scope_keywords: List[str] = Field(default_factory=list, description="Keywords in scopes that indicate this integration")


class IntegrationMetadata(BaseModel):
    """Complete metadata for a single integration type."""
    type: str = Field(..., description="Integration type identifier (e.g., 'gmail', 'github')")
    display_name: str = Field(..., description="Human-readable display name")
    oauth_provider: Optional[str] = Field(None, description="OAuth provider name (e.g., 'google', 'github')")
    requires_oauth: bool = Field(True, description="Whether this integration requires OAuth")
    icon: IntegrationIcon = Field(..., description="Icon configuration")
    brand_color: Optional[str] = Field(None, description="Brand color in hex (e.g., '#4285F4')")
    default_scopes: List[str] = Field(default_factory=list, description="Default scopes to request")
    scopes: List[IntegrationScope] = Field(default_factory=list, description="Available scopes with metadata")
    detection_patterns: Optional[DetectionPatterns] = Field(None, description="Patterns to detect tool membership")


class IntegrationMetadataResponse(BaseModel):
    """Response schema for GET /api/integrations/metadata."""
    integrations: List[IntegrationMetadata] = Field(..., description="List of all available integrations")
    provider_to_types: Dict[str, List[str]] = Field(
        ...,
        description="Mapping of OAuth providers to integration types (e.g., {'google': ['gmail', 'googledrive']})"
    )
