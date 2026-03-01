from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from seer.core.schema.models import TriggerIdentity


class TriggerDescriptor(TriggerIdentity):
    event_schema: Dict[str, Any]
    filter_schema: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    is_connected: bool = True


class TriggerCatalogResponse(BaseModel):
    triggers: List[TriggerDescriptor]


class TriggerAccountInfo(BaseModel):
    """Account information for a specific trigger."""

    id: int = Field(..., description="Database ID of the OAuth connection")
    provider_account_id: str = Field(..., description="Provider's internal account identifier")
    display_name: str = Field(..., description="Human-readable account name (email, username)")
    has_required_scopes: bool = Field(..., description="Whether account has all required scopes")
    missing_scopes: List[str] = Field(default_factory=list, description="List of missing scopes")


class TriggerAccountsResponse(BaseModel):
    """Response for trigger accounts endpoint."""

    trigger_key: str
    provider: Optional[str]
    accounts: List[TriggerAccountInfo] = Field(default_factory=list)
    requires_selection: bool = Field(..., description="True if user must select an account (multiple available)")


class TriggerSubscriptionCreateRequest(BaseModel):
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    enabled: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    # Form trigger fields
    form_suffix: Optional[str] = None
    form_fields: Optional[List[Dict[str, Any]]] = None
    form_config: Optional[Dict[str, Any]] = None


class TriggerSubscriptionUpdateRequest(BaseModel):
    provider_connection_id: Optional[int] = None
    enabled: Optional[bool] = None
    filters: Optional[Dict[str, Any]] = None
    provider_config: Optional[Dict[str, Any]] = None


class TriggerSubscriptionResponse(BaseModel):
    subscription_id: int
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    connection_display_name: Optional[str] = None  # Human-readable account name (email, username)
    enabled: bool
    filters: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    secret_token: Optional[str] = None
    webhook_url: Optional[str] = None
    form_url: Optional[str] = None
    # Form trigger fields
    form_suffix: Optional[str] = None
    form_fields: Optional[List[Dict[str, Any]]] = None
    form_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class TriggerSubscriptionListResponse(BaseModel):
    items: List[TriggerSubscriptionResponse] = Field(default_factory=list)


class TriggerSubscriptionListItem(BaseModel):
    """Extended subscription info for management list view."""
    id: int
    trigger_id: str
    trigger_key: str
    title: Optional[str] = None
    enabled: bool
    workflow_id: str
    workflow_title: str
    last_event_at: Optional[datetime] = None
    created_at: datetime


class TriggerSubscriptionListItemsResponse(BaseModel):
    """Response for trigger subscription list with extended info."""
    items: List[TriggerSubscriptionListItem] = Field(default_factory=list)


class TriggerSubscriptionToggleRequest(BaseModel):
    """Request to toggle subscription enabled status."""
    enabled: bool


class TriggerSubscriptionTestRequest(BaseModel):
    event: Optional[Dict[str, Any]] = None


class TriggerSubscriptionTestResponse(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class StartListeningResponse(BaseModel):
    webhook_url: Optional[str] = None
    secret_token: Optional[str] = None
    subscription_id: int
    form_url: Optional[str] = None


class PendingEventItem(BaseModel):
    event_id: int
    data: Dict[str, Any]
    received_at: str


class PendingEventsResponse(BaseModel):
    events: List[PendingEventItem] = Field(default_factory=list)
    latest_event_id: Optional[int] = None


class SubscriptionEventCountResponse(BaseModel):
    """Response containing the count of stored events for a subscription."""
    subscription_id: int
    event_count: int
    has_events: bool


class TriggerEventGenerateRequest(BaseModel):
    """Request to generate a synthetic trigger event for instant triggering."""
    trigger_id: str = Field(..., description="The trigger instance ID from the workflow")
    provider_config: Dict[str, Any] = Field(default_factory=dict, description="Trigger configuration (e.g., cron_expression, timezone)")


class TriggerEventGenerateResponse(BaseModel):
    """Response containing the generated trigger event envelope."""
    envelope: Dict[str, Any] = Field(..., description="The generated event envelope ready for workflow execution")
    display_title: str = Field(..., description="Human-readable title for the trigger event")


__all__ = [
    "TriggerDescriptor",
    "TriggerCatalogResponse",
    "TriggerAccountInfo",
    "TriggerAccountsResponse",
    "TriggerSubscriptionCreateRequest",
    "TriggerSubscriptionUpdateRequest",
    "TriggerSubscriptionResponse",
    "TriggerSubscriptionListResponse",
    "TriggerSubscriptionListItem",
    "TriggerSubscriptionListItemsResponse",
    "TriggerSubscriptionToggleRequest",
    "TriggerSubscriptionTestRequest",
    "TriggerSubscriptionTestResponse",
    "StartListeningResponse",
    "PendingEventItem",
    "PendingEventsResponse",
    "SubscriptionEventCountResponse",
    "TriggerEventGenerateRequest",
    "TriggerEventGenerateResponse",
]
