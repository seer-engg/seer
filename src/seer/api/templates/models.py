"""Pydantic models for the workflow templates API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from seer.database.template_models import TemplateCategory


class RequiredIntegration(BaseModel):
    """Required integration for a workflow template."""

    provider: str  # "google", "github"
    integration_type: str  # "gmail", "slack"
    reason: str  # "To send email notifications"


class TemplateConfigField(BaseModel):
    """Configurable field extracted from template spec placeholders."""

    name: str  # "recipient_email"
    label: str  # "Recipient Email"
    type: str  # "string", "number", "boolean", "select"
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None
    options: Optional[List[str]] = None  # For select type


class TemplateSummary(BaseModel):
    """Summary response for template listings."""

    template_id: str  # "tpl_{id}"
    slug: str
    name: str
    description: str
    category: str
    tags: List[str]
    icon: Optional[str] = None
    is_featured: bool
    usage_count: int
    required_integrations: List[RequiredIntegration]


class TemplateDetailResponse(TemplateSummary):
    """Detailed template response including spec and config fields."""

    spec: Dict[str, Any]  # Full workflow spec
    config_fields: List[TemplateConfigField]  # Extracted ${config.xxx} fields
    preview_image_url: Optional[str] = None
    source: str
    created_at: datetime
    updated_at: datetime


class IntegrationStatus(BaseModel):
    """Status of a required integration for current user."""

    provider: str
    integration_type: str
    reason: str
    connected: bool
    connection_id: Optional[int] = None
    connection_name: Optional[str] = None


class TemplateRequirementsResponse(BaseModel):
    """Response for checking template requirements against user's integrations."""

    template_id: str
    all_connected: bool
    integrations: List[IntegrationStatus]


class TemplateInstantiateRequest(BaseModel):
    """Request to create a workflow from a template."""

    name: str  # Workflow name
    config: Dict[str, Any] = Field(default_factory=dict)  # Config values for placeholders
    provider_connections: Dict[str, int] = Field(default_factory=dict)
    # Maps integration_type -> connection_id


class TemplateInstantiateResponse(BaseModel):
    """Response after creating a workflow from a template."""

    workflow_id: str
    name: str
    missing_integrations: List[RequiredIntegration] = Field(default_factory=list)  # Warnings


class TemplateCategoryInfo(BaseModel):
    """Information about a template category."""

    key: str
    label: str
    template_count: int


class TemplateCategoriesResponse(BaseModel):
    """Response listing available template categories."""

    categories: List[TemplateCategoryInfo]


class TemplateListResponse(BaseModel):
    """Paginated list of templates."""

    items: List[TemplateSummary]
    total: int
    next_cursor: Optional[str] = None


# Admin models


class TemplateCreateRequest(BaseModel):
    """Request to create a new template from an existing workflow (admin only)."""

    workflow_id: str  # wf_xxx format - the workflow to convert
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(..., min_length=1, max_length=255)
    description: str
    category: str  # Must be a valid TemplateCategory value
    tags: List[str] = Field(default_factory=list)
    required_integrations: Optional[List[RequiredIntegration]] = None  # Optional override, auto-detected if None
    icon: Optional[str] = None
    preview_image_url: Optional[str] = None
    is_published: bool = False
    is_featured: bool = False


class TemplateUpdateRequest(BaseModel):
    """Request to update an existing template (admin only)."""

    slug: Optional[str] = Field(None, min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    spec: Optional[Dict[str, Any]] = None
    required_integrations: Optional[List[RequiredIntegration]] = None
    icon: Optional[str] = None
    preview_image_url: Optional[str] = None
    is_featured: Optional[bool] = None


class TemplatePublishRequest(BaseModel):
    """Request to publish/unpublish a template (admin only)."""

    is_published: bool


class TemplateAdminResponse(TemplateDetailResponse):
    """Admin response including all template fields."""

    is_published: bool


class TemplateAdminListResponse(BaseModel):
    """Admin list response including unpublished templates."""

    items: List[TemplateAdminResponse]
    total: int
    next_cursor: Optional[str] = None


# Category labels mapping
CATEGORY_LABELS = {
    TemplateCategory.MARKETING: "Marketing",
    TemplateCategory.CUSTOMER_SUPPORT: "Customer Support",
    TemplateCategory.SALES: "Sales",
}


__all__ = [
    "RequiredIntegration",
    "TemplateConfigField",
    "TemplateSummary",
    "TemplateDetailResponse",
    "IntegrationStatus",
    "TemplateRequirementsResponse",
    "TemplateInstantiateRequest",
    "TemplateInstantiateResponse",
    "TemplateCategoryInfo",
    "TemplateCategoriesResponse",
    "TemplateListResponse",
    "TemplateCreateRequest",
    "TemplateUpdateRequest",
    "TemplatePublishRequest",
    "TemplateAdminResponse",
    "TemplateAdminListResponse",
    "CATEGORY_LABELS",
]
