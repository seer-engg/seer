"""Database models for organization/team management.

This module implements multi-tenant team support where:
- Every user has a personal Organization created on signup
- All resources (workflows, billing, usage) are org-scoped
- Users can belong to multiple organizations with different roles
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict
from tortoise import fields, models


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class OrganizationType(str, Enum):
    """Types of organizations."""
    PERSONAL = "personal"  # User's personal workspace (one per user)
    TEAM = "team"          # Collaborative team workspace


class OrganizationRole(str, Enum):
    """Roles within an organization."""
    OWNER = "owner"           # Full access, billing, can delete org
    ADMIN = "admin"           # Full access except billing/delete org
    USER = "user"             # Can create/manage own workflows
    CONSULTANT = "consultant" # Limited access, needs approval for workflows


class MembershipStatus(str, Enum):
    """Status of organization membership."""
    PENDING = "pending"       # Invitation sent, not yet accepted
    ACTIVE = "active"         # Accepted and active member
    SUSPENDED = "suspended"   # Temporarily suspended


class InvitationStatus(str, Enum):
    """Status of organization invitations."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ApprovalStatus(str, Enum):
    """Status of workflow approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# =============================================================================
# Organization Models
# =============================================================================


class Organization(models.Model):
    """
    Represents a workspace/team. Can be personal or team.

    Every user has exactly one personal org created on signup.
    This unifies the data model - all queries use organization_id.

    Billing is now org-centric:
    - stripe_customer: FK to StripeCustomer (audit trail for Stripe customer)
    - has_payment_method: Whether a valid payment method is attached
    - payment_method_added_at: When payment method was first added
    """
    id = fields.IntField(primary_key=True)

    # Organization metadata
    name = fields.CharField(max_length=255)
    slug = fields.CharField(max_length=255, unique=True, db_index=True)

    # Type: PERSONAL or TEAM
    type = fields.CharEnumField(OrganizationType, default=OrganizationType.PERSONAL)

    # Owner user (creator of the org)
    owner = fields.ForeignKeyField(
        "models.User",
        related_name="owned_organizations",
        on_delete=fields.CASCADE,
    )

    # Settings (org-level preferences)
    settings = fields.JSONField(default=dict)

    # Billing - org-centric (Phase 1 addition)
    stripe_customer = fields.ForeignKeyField(
        "models.StripeCustomer",
        related_name="organizations",
        on_delete=fields.SET_NULL,
        null=True,
    )
    has_payment_method = fields.BooleanField(default=False)
    payment_method_added_at = fields.DatetimeField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "organizations"
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"Organization<{self.id}:{self.name}>"

    @property
    def is_personal(self) -> bool:
        """Check if this is a personal organization."""
        return self.type == OrganizationType.PERSONAL

    @property
    def is_team(self) -> bool:
        """Check if this is a team organization."""
        return self.type == OrganizationType.TEAM


class OrganizationMembership(models.Model):
    """
    User membership in an organization with role.

    Links users to organizations with specific roles and tracks
    invitation/joining metadata.
    """
    id = fields.IntField(primary_key=True)

    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="memberships",
        on_delete=fields.CASCADE,
    )
    user = fields.ForeignKeyField(
        "models.User",
        related_name="organization_memberships",
        on_delete=fields.CASCADE,
    )

    role = fields.CharEnumField(OrganizationRole, default=OrganizationRole.USER)
    status = fields.CharEnumField(MembershipStatus, default=MembershipStatus.ACTIVE)

    # Invitation tracking
    invited_by = fields.ForeignKeyField(
        "models.User",
        null=True,
        related_name="sent_invitations",
        on_delete=fields.SET_NULL,
    )
    invited_at = fields.DatetimeField(null=True)
    joined_at = fields.DatetimeField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "organization_memberships"
        unique_together = (("organization", "user"),)
        indexes = (
            ("user_id", "status"),
            ("organization_id", "role"),
        )

    def __str__(self) -> str:
        # pylint: disable-next=no-member  # Reason: Tortoise ORM generates _id FK shadow attributes at runtime
        return f"OrganizationMembership<org={self.organization_id}, user={self.user_id}, role={self.role}>"

    @property
    def is_owner(self) -> bool:
        """Check if this membership has owner role."""
        return self.role == OrganizationRole.OWNER

    @property
    def is_admin_or_above(self) -> bool:
        """Check if this membership has admin or owner role."""
        return self.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN)

    @property
    def can_invite(self) -> bool:
        """Check if this member can invite others."""
        return self.role in (
            OrganizationRole.OWNER,
            OrganizationRole.ADMIN,
            OrganizationRole.CONSULTANT,
        )

    @property
    def can_manage_billing(self) -> bool:
        """Check if this member can manage billing."""
        return self.role == OrganizationRole.OWNER


class OrganizationInvitation(models.Model):
    """
    Pending invitations (before user accepts/signs up).

    Invitations can be sent to any email. When accepted, creates
    a membership record.
    """
    id = fields.IntField(primary_key=True)

    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="invitations",
        on_delete=fields.CASCADE,
    )

    # Can invite by email (for non-existing users)
    email = fields.CharField(max_length=320, db_index=True)
    role = fields.CharEnumField(OrganizationRole, default=OrganizationRole.USER)

    # Invitation tracking
    invited_by = fields.ForeignKeyField(
        "models.User",
        related_name="created_invitations",
        on_delete=fields.CASCADE,
    )
    token = fields.CharField(max_length=255, unique=True, db_index=True)
    expires_at = fields.DatetimeField()

    # Status
    status = fields.CharEnumField(InvitationStatus, default=InvitationStatus.PENDING)
    accepted_at = fields.DatetimeField(null=True)
    accepted_by = fields.ForeignKeyField(
        "models.User",
        null=True,
        related_name="accepted_invitations",
        on_delete=fields.SET_NULL,
    )

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "organization_invitations"
        indexes = (
            ("organization_id", "status"),
            ("email", "status"),
            ("token",),
        )

    def __str__(self) -> str:
        # pylint: disable-next=no-member  # Reason: Tortoise ORM generates _id FK shadow attributes at runtime
        return f"OrganizationInvitation<org={self.organization_id}, email={self.email}, status={self.status}>"

    @property
    def is_expired(self) -> bool:
        """Check if the invitation has expired."""
        return _now_utc() > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if the invitation is still pending."""
        return self.status == InvitationStatus.PENDING and not self.is_expired


class WorkflowApproval(models.Model):
    """
    Approval requests for consultant-created workflows.

    Consultants must request approval before their workflows
    can be published/used by the team.
    """
    id = fields.IntField(primary_key=True)

    workflow = fields.ForeignKeyField(
        "models.Workflow",
        related_name="approval_requests",
        on_delete=fields.CASCADE,
    )
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="workflow_approvals",
        on_delete=fields.CASCADE,
    )

    # Request tracking
    requested_by = fields.ForeignKeyField(
        "models.User",
        related_name="approval_requests",
        on_delete=fields.CASCADE,
    )
    requested_at = fields.DatetimeField(auto_now_add=True)

    # Review tracking
    status = fields.CharEnumField(ApprovalStatus, default=ApprovalStatus.PENDING)
    reviewed_by = fields.ForeignKeyField(
        "models.User",
        null=True,
        related_name="reviewed_approvals",
        on_delete=fields.SET_NULL,
    )
    reviewed_at = fields.DatetimeField(null=True)
    review_notes = fields.TextField(null=True)

    class Meta:
        table = "workflow_approvals"
        indexes = (
            ("organization_id", "status"),
            ("workflow_id",),
            ("requested_by_id",),
        )

    def __str__(self) -> str:
        # pylint: disable-next=no-member  # Reason: Tortoise ORM generates _id FK shadow attributes at runtime
        return f"WorkflowApproval<workflow={self.workflow_id}, status={self.status}>"


class WorkflowAssignment(models.Model):
    """
    Workflow assignments for consultants.

    Tracks which workflows a consultant is allowed to access/work on.
    """
    id = fields.IntField(primary_key=True)

    workflow = fields.ForeignKeyField(
        "models.Workflow",
        related_name="assignments",
        on_delete=fields.CASCADE,
    )
    user = fields.ForeignKeyField(
        "models.User",
        related_name="workflow_assignments",
        on_delete=fields.CASCADE,
    )
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="workflow_assignments",
        on_delete=fields.CASCADE,
    )

    # Assignment tracking
    assigned_by = fields.ForeignKeyField(
        "models.User",
        related_name="assigned_workflows",
        on_delete=fields.CASCADE,
    )
    assigned_at = fields.DatetimeField(auto_now_add=True)

    # Optional expiration
    expires_at = fields.DatetimeField(null=True)

    class Meta:
        table = "workflow_assignments"
        unique_together = (("workflow", "user"),)
        indexes = (
            ("user_id",),
            ("workflow_id",),
            ("organization_id",),
        )

    def __str__(self) -> str:
        # pylint: disable-next=no-member  # Reason: Tortoise ORM generates _id FK shadow attributes at runtime
        return f"WorkflowAssignment<workflow={self.workflow_id}, user={self.user_id}>"


# =============================================================================
# Pydantic Response Models
# =============================================================================


class OrganizationPublic(BaseModel):
    """Pydantic model for Organization API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    type: OrganizationType
    created_at: datetime
    updated_at: datetime


class OrganizationWithRole(BaseModel):
    """Organization response with the user's role included."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    type: OrganizationType
    role: OrganizationRole
    is_owner: bool
    created_at: datetime


class MemberPublic(BaseModel):
    """Pydantic model for organization member API responses."""
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    role: OrganizationRole
    status: MembershipStatus
    joined_at: Optional[datetime]


class InvitationPublic(BaseModel):
    """Pydantic model for invitation API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: OrganizationRole
    status: InvitationStatus
    expires_at: datetime
    created_at: datetime


class WorkflowApprovalPublic(BaseModel):
    """Pydantic model for workflow approval API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    workflow_id: int
    status: ApprovalStatus
    requested_at: datetime
    reviewed_at: Optional[datetime]
    review_notes: Optional[str]
