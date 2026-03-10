"""Pydantic models for Organizations API."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from seer.database.organization_models import (
    ApprovalStatus,
    InvitationStatus,
    MembershipStatus,
    OrganizationRole,
    OrganizationType,
)


# =============================================================================
# Request Models
# =============================================================================


class CreateOrganizationRequest(BaseModel):
    """Request to create a new team organization."""
    name: str = Field(..., min_length=1, max_length=255)
    slug: Optional[str] = Field(None, min_length=1, max_length=255)
    transfer_subscription: bool = Field(
        default=False,
        description="If true, transfer the user's personal subscription to this team org"
    )


class ConvertToTeamRequest(BaseModel):
    """Request to convert personal org to team."""
    name: str = Field(..., min_length=1, max_length=255)


class UpdateOrganizationRequest(BaseModel):
    """Request to update organization settings."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    settings: Optional[dict] = None


class CreateInvitationRequest(BaseModel):
    """Request to invite a user to the organization."""
    email: EmailStr
    role: OrganizationRole = OrganizationRole.USER


class UpdateMemberRequest(BaseModel):
    """Request to update a member's role."""
    role: OrganizationRole


class TransferWorkflowsRequest(BaseModel):
    """Request to transfer workflows from personal org to team."""
    workflow_ids: List[int] = Field(..., min_items=1)


# =============================================================================
# Response Models
# =============================================================================


class OrganizationResponse(BaseModel):
    """Organization details in API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    type: OrganizationType
    created_at: datetime
    updated_at: datetime
    checkout_required: bool = Field(
        default=False,
        description="True if this org needs to complete checkout (no subscription transferred)"
    )


class OrganizationWithRoleResponse(BaseModel):
    """Organization with the user's role included."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    type: OrganizationType
    role: OrganizationRole
    is_owner: bool
    created_at: datetime
    success: bool = True


class SwitchOrganizationResponse(BaseModel):
    """Response after switching organization."""
    organization: OrganizationResponse
    role: OrganizationRole
    message: str


class AcceptInvitationResponse(BaseModel):
    """Response from accepting an invitation."""
    success: bool = True
    organization: OrganizationWithRoleResponse


class MemberResponse(BaseModel):
    """Organization member details."""
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    clerk_user_id: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    role: OrganizationRole
    status: MembershipStatus
    joined_at: Optional[datetime]


class InvitationResponse(BaseModel):
    """Invitation details."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: OrganizationRole
    status: InvitationStatus
    expires_at: datetime
    created_at: datetime
    invited_by_email: Optional[str] = None


class TransferWorkflowsResponse(BaseModel):
    """Response after transferring workflows."""
    transferred_count: int
    workflow_ids: List[int]
    message: str


class OrganizationListResponse(BaseModel):
    """List of organizations."""
    organizations: List[OrganizationWithRoleResponse]
    current_organization_id: Optional[int] = None


class MemberListResponse(BaseModel):
    """List of organization members."""
    members: List[MemberResponse]
    total: int


class InvitationListResponse(BaseModel):
    """List of pending invitations."""
    invitations: List[InvitationResponse]
    total: int


class InvitationDetailsResponse(BaseModel):
    """Public invitation details for the accept page (no auth required)."""
    model_config = ConfigDict(from_attributes=True)

    invitation: InvitationResponse
    organization_name: str
    inviter_name: str


# =============================================================================
# Integration Sharing Models
# =============================================================================


class IntegrationResponse(BaseModel):
    """OAuth connection/integration details."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    provider: str
    provider_account_id: str
    status: str
    is_shared: bool
    shared_by_user_id: Optional[int] = None
    shared_by_email: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class IntegrationListResponse(BaseModel):
    """List of available integrations."""
    integrations: List[IntegrationResponse]
    total: int


class ShareIntegrationResponse(BaseModel):
    """Response after sharing an integration."""
    integration: IntegrationResponse
    message: str


# =============================================================================
# Workflow Approval Models
# =============================================================================


class ReviewApprovalRequest(BaseModel):
    """Request to review a workflow approval."""
    status: ApprovalStatus = Field(..., description="APPROVED or REJECTED")
    notes: Optional[str] = Field(None, max_length=1000, description="Review notes")


class WorkflowApprovalResponse(BaseModel):
    """Workflow approval details."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    workflow_id: int
    workflow_name: str
    status: ApprovalStatus
    requested_by_user_id: int
    requested_by_email: Optional[str]
    requested_at: datetime
    reviewed_by_user_id: Optional[int] = None
    reviewed_by_email: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None


class WorkflowApprovalListResponse(BaseModel):
    """List of workflow approvals."""
    approvals: List[WorkflowApprovalResponse]
    total: int


class RequestApprovalResponse(BaseModel):
    """Response after requesting workflow approval."""
    approval: WorkflowApprovalResponse
    message: str


# =============================================================================
# Billing Models
# =============================================================================


class OrgBillingResponse(BaseModel):
    """Organization billing details."""
    tier: str
    status: str
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False
    has_payment_method: bool = False


class OrgBillingPortalResponse(BaseModel):
    """Response containing Stripe portal URL."""
    portal_url: str


class OrgCheckoutRequest(BaseModel):
    """Request to create checkout session for organization."""
    price_id: str
    success_url: str
    cancel_url: str


class OrgCheckoutResponse(BaseModel):
    """Response containing Stripe checkout URL."""
    checkout_url: str


class OrgInvoiceItem(BaseModel):
    """Invoice data for billing history."""
    id: str
    number: Optional[str] = None
    status: Optional[str] = None
    currency: Optional[str] = None
    total: Optional[int] = None
    amount_paid: Optional[int] = None
    amount_due: Optional[int] = None
    created_at: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    hosted_invoice_url: Optional[str] = None
    invoice_pdf: Optional[str] = None
    billing_reason: Optional[str] = None


class OrgInvoiceListResponse(BaseModel):
    """Paginated invoices list for organization."""
    items: List[OrgInvoiceItem]
    page: int
    page_size: int
    has_more: bool


class OrgUsageSummaryResponse(BaseModel):
    """Organization usage summary for the current period."""
    workflows_count: int
    runs_this_month: int
    llm_credits_this_month: float
    period_start: Optional[str] = None
    period_end: Optional[str] = None
