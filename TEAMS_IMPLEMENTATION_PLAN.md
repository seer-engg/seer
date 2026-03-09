# Teams/Workspaces Backend Implementation Plan

## Overview

This document outlines the implementation plan for adding multi-tenant team/workspace support to Seer. The system will transform from a single-user model to support organizations where users can collaborate, share resources, and have unified billing.

---

## Architecture Decision: No Clerk Organizations

**We use Clerk for user authentication only.** All organization/team logic is managed in our database.

**Why:**
- Zero Clerk org quota consumption
- Single source of truth (our database)
- No webhook sync complexity between Clerk ↔ our DB
- Full control over custom roles (Owner, Admin, User, Consultant)
- Full control over invitation flow
- Simpler codebase - no sync edge cases

**How it works:**
```
┌─────────────────────────────────────────────────┐
│                    Clerk                         │
│  • User auth (sign-in/sign-up)                  │
│  • JWT with user_id + active_organization_id   │
│  • active_organization_id stored in             │
│    user.publicMetadata (we update via API)      │
└─────────────────────────────────────────────────┘
                      │
                      ▼ JWT (user_id, active_organization_id)
┌─────────────────────────────────────────────────┐
│                 Our Backend                      │
│  • Organizations table (personal + team)        │
│  • Memberships table (roles)                    │
│  • Invitations table                            │
│  • All business logic is ORG-SCOPED            │
│  • Reads org_id from JWT claims (not header)    │
└─────────────────────────────────────────────────┘
```

### Clerk JWT Template Setup

Create a JWT template named `seer` in Clerk Dashboard:
```json
{
  "user_id": "{{user.id}}",
  "email": "{{user.primary_email_address}}",
  "first_name": "{{user.first_name}}",
  "last_name": "{{user.last_name}}",
  "active_organization_id": "{{user.public_metadata.active_organization_id}}"
}
```

Frontend uses: `clerk.session.getToken({ template: "seer" })`

---

## Key Design Principle: Everything is Org-Scoped

**Every user has a personal Organization.** This means:
- All queries are `WHERE organization_id = ?` (never user-scoped)
- Workflows belong to Organization
- Billing belongs to Organization
- Usage tracked per Organization
- One code path, not two (personal vs team)
- Roles/permissions apply uniformly (owner of personal org = owner role)

---

## Requirements Summary

### Roles & Permissions

| Role | Billing | Invite | Manage Workflows | Delete Org | Remove Members | Create Workflows | Approve Workflows |
|------|---------|--------|------------------|------------|----------------|------------------|-------------------|
| Owner | ✅ | ✅ | ✅ (all) | ✅ | ✅ | ✅ | ✅ |
| Admin | ❌ | ✅ | ✅ (all) | ❌ | ✅ | ✅ | ✅ |
| User | ❌ | ❌ | ✅ (own) | ❌ | ❌ | ✅ | ❌ |
| Consultant | ❌ | ✅ (onboard) | ❌ (assigned only) | ❌ | ❌ | ✅ (needs approval) | ❌ |

### Key Behaviors

1. **Personal Org for Every User**: Every user gets a personal organization on signup (unifies data model)
2. **No Clerk Orgs**: We manage all org/team logic in our database
3. **Multi-org**: Any user can belong to multiple teams and switch contexts
4. **Personal + Team**: Users can maintain personal workspace AND be members of teams
5. **Optional Workflow Transfer**: When joining a team, workflows stay personal. User can CHOOSE to transfer specific workflows to the team (one-way ownership transfer)
6. **Shared Integrations**: Members can share OAuth connections per provider; shared connections usable by all team members
7. **Billing Conversion**: Only owners with subscription can create teams; their subscription converts to team subscription

---

## Phase 1: Database Schema Changes

### 1.1 New Models

```python
# /database/organization_models.py

class Organization(Model):
    """
    Represents a workspace/team. Can be personal or team.
    Every user has exactly one personal org created on signup.
    """
    id = fields.IntField(pk=True)

    # Organization metadata
    name = fields.CharField(max_length=255)
    slug = fields.CharField(max_length=255, unique=True)  # URL-friendly identifier

    # Type: PERSONAL or TEAM
    type = fields.CharEnumField(OrganizationType, default=OrganizationType.PERSONAL)

    # Owner user (creator of the org)
    owner = fields.ForeignKeyField("models.User", related_name="owned_organizations")

    # Settings
    settings = fields.JSONField(default=dict)  # org-level preferences

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "organizations"


class OrganizationType(str, Enum):
    PERSONAL = "personal"  # User's personal workspace (one per user)
    TEAM = "team"          # Collaborative team workspace


class OrganizationMembership(Model):
    """
    User membership in an organization with role.
    """
    id = fields.IntField(pk=True)

    organization = fields.ForeignKeyField("models.Organization", related_name="memberships")
    user = fields.ForeignKeyField("models.User", related_name="memberships")

    role = fields.CharEnumField(OrganizationRole)

    # For tracking invitation status
    status = fields.CharEnumField(MembershipStatus, default=MembershipStatus.ACTIVE)
    invited_by = fields.ForeignKeyField("models.User", null=True, related_name="sent_invitations")
    invited_at = fields.DatetimeField(null=True)
    joined_at = fields.DatetimeField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "organization_memberships"
        unique_together = (("organization", "user"),)


class OrganizationRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    USER = "user"
    CONSULTANT = "consultant"


class MembershipStatus(str, Enum):
    PENDING = "pending"    # Invitation sent
    ACTIVE = "active"      # Accepted and active
    SUSPENDED = "suspended"


class OrganizationInvitation(Model):
    """
    Pending invitations (before user accepts/signs up).
    """
    id = fields.IntField(pk=True)

    organization = fields.ForeignKeyField("models.Organization", related_name="invitations")

    # Can invite by email (for non-existing users)
    email = fields.CharField(max_length=255)
    role = fields.CharEnumField(OrganizationRole)

    # Invitation tracking
    invited_by = fields.ForeignKeyField("models.User", related_name="created_invitations")
    token = fields.CharField(max_length=255, unique=True)  # Secure invite token
    expires_at = fields.DatetimeField()

    # Status
    status = fields.CharEnumField(InvitationStatus, default=InvitationStatus.PENDING)
    accepted_at = fields.DatetimeField(null=True)
    accepted_by = fields.ForeignKeyField("models.User", null=True, related_name="accepted_invitations")

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "organization_invitations"


class InvitationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class WorkflowApproval(Model):
    """
    Approval requests for consultant-created workflows.
    """
    id = fields.IntField(pk=True)

    workflow = fields.ForeignKeyField("models.Workflow", related_name="approval_requests")
    organization = fields.ForeignKeyField("models.Organization", related_name="workflow_approvals")

    requested_by = fields.ForeignKeyField("models.User", related_name="approval_requests")
    requested_at = fields.DatetimeField(auto_now_add=True)

    status = fields.CharEnumField(ApprovalStatus, default=ApprovalStatus.PENDING)
    reviewed_by = fields.ForeignKeyField("models.User", null=True, related_name="reviewed_approvals")
    reviewed_at = fields.DatetimeField(null=True)
    review_notes = fields.TextField(null=True)

    class Meta:
        table = "workflow_approvals"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
```

### 1.2 Model Modifications

```python
# Modify existing models to support organization context

# Workflow model changes
class Workflow(Model):
    # ... existing fields ...

    # NEW: Organization ownership (replaces user-only ownership)
    organization = fields.ForeignKeyField("models.Organization", related_name="workflows", null=True)

    # Keep user for creator tracking
    user = fields.ForeignKeyField("models.User", related_name="created_workflows")

    # NEW: Approval status for consultant-created workflows
    approval_status = fields.CharEnumField(ApprovalStatus, null=True)

    # NEW: Workflow visibility within org
    visibility = fields.CharEnumField(WorkflowVisibility, default=WorkflowVisibility.TEAM)


class WorkflowVisibility(str, Enum):
    PRIVATE = "private"  # Only creator can see
    TEAM = "team"        # All org members can see
    ASSIGNED = "assigned"  # Only assigned users can see


# BillingProfile changes
class BillingProfile(Model):
    # ... existing fields ...

    # CHANGE: Can be owned by org instead of just user
    owner_user = fields.ForeignKeyField("models.User", null=True, related_name="personal_billing_profile")
    owner_organization = fields.ForeignKeyField("models.Organization", null=True, related_name="billing_profile")

    # Validation: exactly one of owner_user or owner_organization must be set


# UsageCounter changes
class UsageCounter(Model):
    # ... existing fields ...

    # NEW: Track at org level
    organization = fields.ForeignKeyField("models.Organization", null=True, related_name="usage_counters")

    # Keep user for per-user breakdown within org
    user = fields.ForeignKeyField("models.User", null=True, related_name="usage_counters")


# Integration/OAuth changes
class OAuthConnection(Model):
    # ... existing fields ...

    # NEW: Sharing with organization
    shared_with_organization = fields.ForeignKeyField("models.Organization", null=True, related_name="shared_connections")

    # Owner is still the user who connected
    user = fields.ForeignKeyField("models.User", related_name="oauth_connections")
```

### 1.3 Migration Strategy

```python
# Migration: Create personal organizations for existing users

async def migrate_to_organizations():
    """
    Create virtual personal organizations for all existing users
    and migrate their resources.
    """
    users = await User.all()

    for user in users:
        # Create personal org
        org = await Organization.create(
            name=f"{user.first_name or user.email}'s Workspace",
            slug=f"personal-{user.user_id}",
            type=OrganizationType.PERSONAL,
            owner=user,
        )

        # Create owner membership
        await OrganizationMembership.create(
            organization=org,
            user=user,
            role=OrganizationRole.OWNER,
            status=MembershipStatus.ACTIVE,
            joined_at=user.created_at,
        )

        # Migrate workflows
        await Workflow.filter(user=user).update(organization=org)

        # Migrate billing profile to org ownership
        billing = await BillingProfile.get_or_none(owner_user=user)
        if billing:
            billing.owner_organization = org
            await billing.save()

        # Migrate usage counters
        await UsageCounter.filter(user=user).update(organization=org)
```

---

## Phase 2: API Layer Changes

### 2.1 Organization Context Middleware

```python
# /api/core/middleware/organization.py

class OrganizationContextMiddleware:
    """
    Extracts organization context from JWT claims.
    Sets request.state.organization and request.state.membership.

    The active_organization_id comes from Clerk JWT claims,
    which is populated from user.publicMetadata.active_organization_id
    """

    async def __call__(self, request: Request, call_next):
        user = request.state.db_user
        claims = request.state.user.claims  # JWT claims from Clerk

        # Get org_id from JWT claims (set via Clerk publicMetadata)
        org_id = claims.get("active_organization_id")

        if org_id:
            # Validate membership
            membership = await OrganizationMembership.get_or_none(
                organization_id=int(org_id),
                user=user,
                status=MembershipStatus.ACTIVE,
            ).prefetch_related("organization")

            if not membership:
                # User's metadata points to invalid org - fall back to personal
                org_id = None

            if membership:
                request.state.organization = membership.organization
                request.state.membership = membership
                return await call_next(request)

        # Default to personal org (no org in JWT or invalid org)
        personal_org = await Organization.get(
            owner=user,
            type=OrganizationType.PERSONAL,
        )
        request.state.organization = personal_org
        request.state.membership = await OrganizationMembership.get(
            organization=personal_org,
            user=user,
        )

        return await call_next(request)
```

### 2.2 Permission Decorators

```python
# /api/core/permissions.py

from functools import wraps
from typing import List

def require_role(allowed_roles: List[OrganizationRole]):
    """
    Decorator to enforce role-based access.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            membership = request.state.membership

            if membership.role not in allowed_roles:
                raise HTTPException(
                    403,
                    f"Requires one of roles: {[r.value for r in allowed_roles]}"
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_owner():
    return require_role([OrganizationRole.OWNER])


def require_admin_or_above():
    return require_role([OrganizationRole.OWNER, OrganizationRole.ADMIN])


def require_member():
    return require_role([
        OrganizationRole.OWNER,
        OrganizationRole.ADMIN,
        OrganizationRole.USER,
        OrganizationRole.CONSULTANT,
    ])


# Workflow-specific permissions
async def can_manage_workflow(user: User, workflow: Workflow, membership: OrganizationMembership) -> bool:
    """Check if user can edit/delete a workflow."""
    if membership.role in [OrganizationRole.OWNER, OrganizationRole.ADMIN]:
        return True

    if membership.role == OrganizationRole.USER:
        return workflow.user_id == user.id

    if membership.role == OrganizationRole.CONSULTANT:
        return False  # Consultants can only view assigned workflows

    return False


async def can_view_workflow(user: User, workflow: Workflow, membership: OrganizationMembership) -> bool:
    """Check if user can view a workflow."""
    if membership.role in [OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.USER]:
        if workflow.visibility == WorkflowVisibility.TEAM:
            return True
        if workflow.visibility == WorkflowVisibility.PRIVATE:
            return workflow.user_id == user.id

    if membership.role == OrganizationRole.CONSULTANT:
        # TODO: Check workflow assignment
        return await WorkflowAssignment.exists(workflow=workflow, user=user)

    return False
```

### 2.3 New API Endpoints

```python
# /api/organizations/router.py

router = APIRouter(prefix="/api/organizations", tags=["organizations"])

# ============ Organization Management ============

@router.get("/")
async def list_organizations(request: Request) -> List[OrganizationResponse]:
    """List all organizations the user is a member of."""
    memberships = await OrganizationMembership.filter(
        user=request.state.db_user,
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("organization")

    return [
        OrganizationResponse(
            id=m.organization.id,
            name=m.organization.name,
            slug=m.organization.slug,
            type=m.organization.type,
            role=m.role,
            is_owner=m.role == OrganizationRole.OWNER,
        )
        for m in memberships
    ]


@router.post("/{org_id}/switch")
async def switch_organization(
    request: Request,
    org_id: int,
) -> SwitchOrganizationResponse:
    """
    Switch to a different organization.
    Updates Clerk user metadata so JWT includes the new org_id.
    Frontend must call getToken() after this to get updated JWT.
    """
    user = request.state.db_user

    # Validate user is a member of this org
    membership = await OrganizationMembership.get_or_none(
        organization_id=org_id,
        user=user,
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("organization")

    if not membership:
        raise HTTPException(403, "Not a member of this organization")

    # Update Clerk user metadata
    await update_clerk_user_metadata(
        user_id=user.user_id,
        public_metadata={"active_organization_id": org_id},
    )

    return SwitchOrganizationResponse(
        organization=OrganizationResponse.from_orm(membership.organization),
        role=membership.role,
        message="Organization switched. Please refresh your token.",
    )


@router.post("/")
async def create_organization(
    request: Request,
    body: CreateOrganizationRequest,
) -> OrganizationResponse:
    """
    Create a new team organization.
    Requires active subscription on personal org.
    """
    user = request.state.db_user

    # Verify user has an active subscription
    personal_org = await Organization.get(owner=user, type=OrganizationType.PERSONAL)
    billing = await BillingProfile.get(owner_organization=personal_org)
    subscription = await BillingSubscription.get_or_none(billing_profile=billing)

    if not subscription or subscription.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]:
        raise HTTPException(402, "Active subscription required to create a team")

    # Create organization (without Clerk org yet - lazy creation)
    org = await Organization.create(
        name=body.name,
        slug=generate_unique_slug(body.name),
        type=OrganizationType.TEAM,
        owner=user,
    )

    # Create owner membership
    await OrganizationMembership.create(
        organization=org,
        user=user,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
        joined_at=datetime.utcnow(),
    )

    # Transfer subscription to team
    await transfer_subscription_to_org(billing, subscription, org)

    return OrganizationResponse(...)


@router.post("/{org_id}/convert-to-team")
@require_owner()
async def convert_to_team(
    request: Request,
    org_id: int,
    body: ConvertToTeamRequest,
) -> OrganizationResponse:
    """
    Convert personal org to team.
    Called when user wants to make their personal workspace a team.
    """
    org = request.state.organization

    if org.type == OrganizationType.TEAM:
        raise HTTPException(400, "Already a team organization")

    org.type = OrganizationType.TEAM
    org.name = body.name  # Allow renaming when converting
    await org.save()

    return OrganizationResponse(...)


# ============ Member Management ============

@router.get("/{org_id}/members")
@require_member()
async def list_members(request: Request, org_id: int) -> List[MemberResponse]:
    """List all members of the organization."""
    memberships = await OrganizationMembership.filter(
        organization_id=org_id,
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("user")

    return [MemberResponse(...) for m in memberships]


@router.post("/{org_id}/invitations")
@require_role([OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.CONSULTANT])
async def create_invitation(
    request: Request,
    org_id: int,
    body: CreateInvitationRequest,
) -> InvitationResponse:
    """
    Invite a user to the organization.
    Owner/Admin can invite any role.
    Consultant can only invite Users.
    """
    membership = request.state.membership
    org = request.state.organization

    # Consultants can only invite Users
    if membership.role == OrganizationRole.CONSULTANT:
        if body.role != OrganizationRole.USER:
            raise HTTPException(403, "Consultants can only invite users")

    # Create invitation
    invitation = await OrganizationInvitation.create(
        organization=org,
        email=body.email,
        role=body.role,
        invited_by=request.state.db_user,
        token=generate_secure_token(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )

    # Send invitation email (via your email provider: Resend, SendGrid, etc.)
    await send_invitation_email(
        to_email=body.email,
        organization_name=org.name,
        invited_by_name=request.state.db_user.first_name or request.state.db_user.email,
        role=body.role,
        invite_url=f"{settings.FRONTEND_URL}/invitations/{invitation.token}",
    )

    return InvitationResponse(...)


@router.post("/invitations/{token}/accept")
async def accept_invitation(
    request: Request,
    token: str,
) -> OrganizationResponse:
    """Accept an invitation and join the organization."""
    user = request.state.db_user

    invitation = await OrganizationInvitation.get_or_none(
        token=token,
        status=InvitationStatus.PENDING,
    ).prefetch_related("organization")

    if not invitation:
        raise HTTPException(404, "Invitation not found or expired")

    if invitation.expires_at < datetime.utcnow():
        invitation.status = InvitationStatus.EXPIRED
        await invitation.save()
        raise HTTPException(400, "Invitation has expired")

    # Check email matches
    if invitation.email.lower() != user.email.lower():
        raise HTTPException(403, "Invitation was sent to a different email")

    # Create membership
    membership = await OrganizationMembership.create(
        organization=invitation.organization,
        user=user,
        role=invitation.role,
        status=MembershipStatus.ACTIVE,
        invited_by=invitation.invited_by,
        invited_at=invitation.created_at,
        joined_at=datetime.utcnow(),
    )

    # Update invitation
    invitation.status = InvitationStatus.ACCEPTED
    invitation.accepted_at = datetime.utcnow()
    invitation.accepted_by = user
    await invitation.save()

    # NOTE: Workflow migration is OPTIONAL and happens via separate endpoint
    # User can choose to transfer specific workflows after joining

    return OrganizationResponse(...)


@router.patch("/{org_id}/members/{user_id}")
@require_admin_or_above()
async def update_member_role(
    request: Request,
    org_id: int,
    user_id: int,
    body: UpdateMemberRequest,
) -> MemberResponse:
    """Update a member's role."""
    membership = await OrganizationMembership.get(
        organization_id=org_id,
        user_id=user_id,
    )

    # Cannot demote owner unless transferring ownership
    if membership.role == OrganizationRole.OWNER and body.role != OrganizationRole.OWNER:
        raise HTTPException(400, "Cannot demote owner. Transfer ownership first.")

    membership.role = body.role
    await membership.save()

    return MemberResponse(...)


@router.delete("/{org_id}/members/{user_id}")
@require_admin_or_above()
async def remove_member(
    request: Request,
    org_id: int,
    user_id: int,
):
    """Remove a member from the organization."""
    membership = await OrganizationMembership.get(
        organization_id=org_id,
        user_id=user_id,
    )

    if membership.role == OrganizationRole.OWNER:
        raise HTTPException(400, "Cannot remove owner. Transfer ownership first.")

    membership.status = MembershipStatus.SUSPENDED
    await membership.save()


# ============ Workflow Transfer ============

@router.post("/{org_id}/transfer-workflows")
@require_member()
async def transfer_workflows_to_org(
    request: Request,
    org_id: int,
    body: TransferWorkflowsRequest,
) -> TransferResponse:
    """
    Transfer specific workflows from personal org to team organization.
    This is a one-way ownership transfer (not a copy).
    User must explicitly select which workflows to transfer.
    """
    user = request.state.db_user
    target_org = request.state.organization

    if target_org.type != OrganizationType.TEAM:
        raise HTTPException(400, "Can only transfer workflows to team organizations")

    # Get user's personal org
    personal_org = await Organization.get(owner=user, type=OrganizationType.PERSONAL)

    # Validate workflow_ids are required (no bulk transfer without explicit selection)
    if not body.workflow_ids:
        raise HTTPException(400, "Must specify workflow_ids to transfer")

    # Get workflows to transfer (must belong to user's personal org)
    workflows = await Workflow.filter(
        id__in=body.workflow_ids,
        organization=personal_org,
        user=user,
    )

    if len(workflows) != len(body.workflow_ids):
        raise HTTPException(400, "Some workflows not found or not owned by you")

    # Transfer workflows
    transferred_count = 0
    for workflow in workflows:
        workflow.organization = target_org
        # Keep user as creator for attribution
        await workflow.save()
        transferred_count += 1

    return TransferResponse(
        transferred_count=transferred_count,
        workflow_ids=[w.id for w in workflows],
    )


# ============ Shared Integrations ============

@router.post("/{org_id}/integrations/{connection_id}/share")
@require_member()
async def share_integration(
    request: Request,
    org_id: int,
    connection_id: int,
) -> IntegrationResponse:
    """Share an OAuth connection with the organization."""
    user = request.state.db_user
    org = request.state.organization

    connection = await OAuthConnection.get(id=connection_id, user=user)
    connection.shared_with_organization = org
    await connection.save()

    return IntegrationResponse(...)


@router.delete("/{org_id}/integrations/{connection_id}/share")
@require_member()
async def unshare_integration(
    request: Request,
    org_id: int,
    connection_id: int,
):
    """Remove sharing of an OAuth connection."""
    user = request.state.db_user

    connection = await OAuthConnection.get(id=connection_id, user=user)
    connection.shared_with_organization = None
    await connection.save()


@router.get("/{org_id}/integrations")
@require_member()
async def list_org_integrations(
    request: Request,
    org_id: int,
) -> List[IntegrationResponse]:
    """List all integrations available to the organization (shared + personal)."""
    user = request.state.db_user
    org = request.state.organization

    # Get shared connections
    shared = await OAuthConnection.filter(shared_with_organization=org)

    # Get user's personal connections
    personal = await OAuthConnection.filter(user=user, shared_with_organization=None)

    return [IntegrationResponse(...) for c in [*shared, *personal]]


# ============ Workflow Approval (for Consultants) ============

@router.post("/workflows/{workflow_id}/request-approval")
@require_role([OrganizationRole.CONSULTANT])
async def request_workflow_approval(
    request: Request,
    workflow_id: int,
) -> ApprovalResponse:
    """Request approval for a consultant-created workflow."""
    user = request.state.db_user
    org = request.state.organization

    workflow = await Workflow.get(id=workflow_id, user=user)

    approval = await WorkflowApproval.create(
        workflow=workflow,
        organization=org,
        requested_by=user,
    )

    workflow.approval_status = ApprovalStatus.PENDING
    await workflow.save()

    # Notify admins/owners
    await notify_approval_request(org, workflow, user)

    return ApprovalResponse(...)


@router.post("/approvals/{approval_id}/review")
@require_admin_or_above()
async def review_workflow_approval(
    request: Request,
    approval_id: int,
    body: ReviewApprovalRequest,
) -> ApprovalResponse:
    """Approve or reject a workflow approval request."""
    user = request.state.db_user

    approval = await WorkflowApproval.get(id=approval_id).prefetch_related("workflow")

    approval.status = body.status
    approval.reviewed_by = user
    approval.reviewed_at = datetime.utcnow()
    approval.review_notes = body.notes
    await approval.save()

    approval.workflow.approval_status = body.status
    await approval.workflow.save()

    return ApprovalResponse(...)
```

### 2.4 Billing Changes

```python
# /api/subscriptions/team_billing.py

async def transfer_subscription_to_org(
    personal_billing: BillingProfile,
    subscription: BillingSubscription,
    target_org: Organization,
):
    """
    Transfer a user's subscription to a team organization.
    """
    # Create org billing profile
    org_billing = await BillingProfile.create(
        type=BillingProfileType.TEAM,
        owner_organization=target_org,
        stripe_customer_id=personal_billing.stripe_customer_id,
        has_payment_method=personal_billing.has_payment_method,
        payment_method_added_at=personal_billing.payment_method_added_at,
    )

    # Move subscription to org billing
    subscription.billing_profile = org_billing
    await subscription.save()

    # Update Stripe customer metadata
    await stripe.Customer.modify(
        personal_billing.stripe_customer_id,
        metadata={
            "organization_id": str(target_org.id),
            "billing_type": "team",
        },
    )

    # Clear personal billing profile's subscription link
    personal_billing.stripe_customer_id = None
    personal_billing.has_payment_method = False
    await personal_billing.save()


async def get_org_billing_context(org: Organization) -> BillingContext:
    """Get billing context for an organization."""
    billing = await BillingProfile.get(owner_organization=org)
    subscription = await BillingSubscription.get_or_none(billing_profile=billing)

    return BillingContext(
        billing_profile=billing,
        subscription=subscription,
        tier=subscription.tier if subscription else SubscriptionTier.FREE,
        limits=get_tier_limits(subscription.tier if subscription else SubscriptionTier.FREE),
    )
```

---

## Phase 3: Email & Invitation Service

Since we're not using Clerk organizations, we handle invitation emails ourselves.

### 3.1 Email Service

```python
# /services/email_service.py

from resend import Resend  # or SendGrid, AWS SES, etc.

resend = Resend(api_key=settings.RESEND_API_KEY)


async def send_invitation_email(
    to_email: str,
    organization_name: str,
    invited_by_name: str,
    role: OrganizationRole,
    invite_url: str,
):
    """Send team invitation email."""
    await resend.emails.send(
        from_=f"Seer <{settings.EMAIL_FROM}>",
        to=[to_email],
        subject=f"You've been invited to join {organization_name} on Seer",
        html=render_template(
            "invitation_email.html",
            organization_name=organization_name,
            invited_by_name=invited_by_name,
            role=role.value,
            invite_url=invite_url,
            expires_in="7 days",
        ),
    )


async def send_approval_notification_email(
    to_emails: List[str],
    workflow_name: str,
    requested_by_name: str,
    organization_name: str,
    review_url: str,
):
    """Notify admins/owners about pending workflow approval."""
    for email in to_emails:
        await resend.emails.send(
            from_=f"Seer <{settings.EMAIL_FROM}>",
            to=[email],
            subject=f"Workflow approval requested: {workflow_name}",
            html=render_template(
                "approval_request_email.html",
                workflow_name=workflow_name,
                requested_by_name=requested_by_name,
                organization_name=organization_name,
                review_url=review_url,
            ),
        )


async def send_member_joined_notification(
    to_email: str,
    new_member_name: str,
    organization_name: str,
):
    """Notify owner when new member joins."""
    await resend.emails.send(
        from_=f"Seer <{settings.EMAIL_FROM}>",
        to=[to_email],
        subject=f"{new_member_name} joined {organization_name}",
        html=render_template(
            "member_joined_email.html",
            new_member_name=new_member_name,
            organization_name=organization_name,
        ),
    )
```

### 3.2 Clerk Metadata Service

```python
# /services/clerk_service.py

from clerk_backend_api import Clerk

clerk = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)


async def update_clerk_user_metadata(
    user_id: str,
    public_metadata: dict | None = None,
    private_metadata: dict | None = None,
):
    """
    Update Clerk user metadata.
    Used for storing active_organization_id for JWT claims.
    """
    update_data = {}

    if public_metadata:
        # Merge with existing metadata (don't overwrite)
        existing = await clerk.users.get(user_id)
        merged_public = {**(existing.public_metadata or {}), **public_metadata}
        update_data["public_metadata"] = merged_public

    if private_metadata:
        existing = await clerk.users.get(user_id)
        merged_private = {**(existing.private_metadata or {}), **private_metadata}
        update_data["private_metadata"] = merged_private

    if update_data:
        await clerk.users.update(user_id, **update_data)


async def set_active_organization(user_id: str, org_id: int):
    """Set the user's active organization in Clerk metadata."""
    await update_clerk_user_metadata(
        user_id=user_id,
        public_metadata={"active_organization_id": org_id},
    )


async def clear_active_organization(user_id: str):
    """Clear the user's active organization (revert to personal)."""
    await update_clerk_user_metadata(
        user_id=user_id,
        public_metadata={"active_organization_id": None},
    )
```

### 3.3 Invitation Token Generation

```python
# /services/invitation_service.py

import secrets
from datetime import datetime, timedelta


def generate_secure_token() -> str:
    """Generate a secure random token for invitations."""
    return secrets.token_urlsafe(32)


async def create_invitation(
    organization: Organization,
    email: str,
    role: OrganizationRole,
    invited_by: User,
    expires_in_days: int = 7,
) -> OrganizationInvitation:
    """Create an invitation and send email."""

    # Check for existing pending invitation
    existing = await OrganizationInvitation.get_or_none(
        organization=organization,
        email=email.lower(),
        status=InvitationStatus.PENDING,
    )

    if existing:
        if existing.expires_at > datetime.utcnow():
            raise HTTPException(400, "Invitation already sent to this email")
        # Expired invitation - mark it and create new one
        existing.status = InvitationStatus.EXPIRED
        await existing.save()

    # Create new invitation
    invitation = await OrganizationInvitation.create(
        organization=organization,
        email=email.lower(),
        role=role,
        invited_by=invited_by,
        token=generate_secure_token(),
        expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
    )

    # Send email
    await send_invitation_email(
        to_email=email,
        organization_name=organization.name,
        invited_by_name=invited_by.first_name or invited_by.email,
        role=role,
        invite_url=f"{settings.FRONTEND_URL}/invitations/{invitation.token}",
    )

    return invitation
```

---

## Phase 4: Usage Tracking Changes

### 4.1 Organization-Level Usage

```python
# /observability/org_usage.py

async def track_org_usage(
    org: Organization,
    user: User,
    resource_type: ResourceType,
    count: int = 1,
    value: Decimal = Decimal("0"),
):
    """Track usage at organization level."""
    # Get or create org counter
    counter, _ = await UsageCounter.get_or_create(
        organization=org,
        resource_type=resource_type,
        period_start=get_current_period_start(org),
        period_end=get_current_period_end(org),
        defaults={"count": 0, "value": Decimal("0")},
    )

    counter.count += count
    counter.value += value
    await counter.save()

    # Also track per-user for breakdown
    user_counter, _ = await UsageCounter.get_or_create(
        organization=org,
        user=user,
        resource_type=resource_type,
        period_start=get_current_period_start(org),
        period_end=get_current_period_end(org),
        defaults={"count": 0, "value": Decimal("0")},
    )

    user_counter.count += count
    user_counter.value += value
    await user_counter.save()


async def check_org_limits(org: Organization, resource_type: ResourceType) -> bool:
    """Check if organization is within usage limits."""
    billing = await get_org_billing_context(org)
    limits = billing.limits

    counter = await UsageCounter.get_or_none(
        organization=org,
        user=None,  # Org-level counter
        resource_type=resource_type,
        period_start=get_current_period_start(org),
    )

    if not counter:
        return True

    limit = getattr(limits, f"{resource_type.value}_limit", None)
    if limit is None:
        return True  # Unlimited

    return counter.count < limit or counter.value < limit
```

---

## Phase 5: Testing Plan

### 5.1 Unit Tests

```python
# tests/unit/test_organizations.py

class TestOrganizationCreation:
    async def test_create_personal_org_on_user_signup(self):
        """New users should get a personal org automatically."""
        pass

    async def test_create_team_org_requires_subscription(self):
        """Creating a team requires an active subscription."""
        pass

    async def test_all_queries_are_org_scoped(self):
        """Verify all resource queries use organization context."""
        pass


class TestMembership:
    async def test_owner_permissions(self):
        """Owners should have full access."""
        pass

    async def test_admin_cannot_access_billing(self):
        """Admins should not be able to manage billing."""
        pass

    async def test_consultant_needs_approval(self):
        """Consultant-created workflows need approval."""
        pass

    async def test_consultant_can_invite_users_only(self):
        """Consultants can only invite User role."""
        pass


class TestWorkflowTransfer:
    async def test_transfer_selected_workflows_to_team(self):
        """User can transfer specific workflows to team org."""
        pass

    async def test_transfer_requires_explicit_selection(self):
        """Cannot bulk transfer without selecting workflows."""
        pass

    async def test_workflow_visibility_in_team(self):
        """Team members should see team workflows."""
        pass

    async def test_workflows_stay_personal_by_default(self):
        """Workflows remain in personal org when joining team."""
        pass


class TestSharedIntegrations:
    async def test_share_oauth_connection(self):
        """Members can share their OAuth connections."""
        pass

    async def test_shared_connection_usable_by_team(self):
        """Shared connections should be usable by all team members."""
        pass


class TestBilling:
    async def test_subscription_transfer_to_team(self):
        """Subscription should transfer when creating team."""
        pass

    async def test_team_usage_quota_shared(self):
        """Usage should be tracked at team level."""
        pass
```

### 5.2 E2E Tests

```python
# tests/e2e/test_team_flow.py

class TestTeamE2E:
    async def test_full_team_creation_flow(self):
        """
        1. User with subscription creates team
        2. Invites another user
        3. Invited user accepts
        4. Both can access team resources
        """
        pass

    async def test_consultant_workflow_approval_flow(self):
        """
        1. Consultant joins team
        2. Creates workflow
        3. Requests approval
        4. Admin approves
        5. Workflow becomes visible to team
        """
        pass
```

---

## Phase 6: Migration Plan

### 6.1 Database Migration Steps

1. **Add new tables** (non-breaking):
   - `organizations`
   - `organization_memberships`
   - `organization_invitations`
   - `workflow_approvals`

2. **Add new columns** (nullable, non-breaking):
   - `workflows.organization_id`
   - `workflows.approval_status`
   - `workflows.visibility`
   - `billing_profiles.owner_organization_id`
   - `usage_counters.organization_id`
   - `oauth_connections.shared_with_organization_id`

3. **Data migration** (backfill):
   - Create personal organizations for existing users
   - Link workflows to personal orgs
   - Link billing profiles to personal orgs

4. **Make columns required** (after backfill):
   - `workflows.organization_id` NOT NULL

### 6.2 Rollback Plan

Each migration step should be reversible:
- New tables can be dropped
- New columns can be dropped
- Data migration can be reverted by deleting created orgs

---

## Implementation Order

### Sprint 1: Foundation (Week 1-2)
- [ ] **Clerk Dashboard**: Create "seer" JWT template with `active_organization_id` claim
- [ ] Database models and migrations
- [ ] Personal org creation on signup (+ set Clerk metadata)
- [ ] Organization context middleware (reads org from JWT claims)
- [ ] Switch organization endpoint (updates Clerk metadata)
- [ ] Basic CRUD for organizations

### Sprint 2: Membership (Week 3-4)
- [ ] Role-based permissions
- [ ] Invitation system with email service
- [ ] Invitation accept/decline flow
- [ ] Member management APIs

### Sprint 3: Resources (Week 5-6)
- [ ] Workflow organization scoping
- [ ] Optional workflow transfer
- [ ] Shared integrations
- [ ] Consultant workflow approval

### Sprint 4: Billing (Week 7-8)
- [ ] Organization-level billing
- [ ] Subscription transfer
- [ ] Usage tracking at org level
- [ ] Team billing portal

### Sprint 5: Polish (Week 9-10)
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Performance optimization
- [ ] Monitoring and alerts

---

## Open Questions / Decisions Needed

1. **Seat-based pricing**: Should team plans have per-seat pricing in the future?
2. **Workflow ownership on leave**: When a member leaves, what happens to workflows they created in the team?
   - Option A: Workflows stay with the team (recommended)
   - Option B: Workflows transfer back to their personal org
3. **Organization deletion**: What's the process for deleting a team org? What happens to workflows?
4. **Audit logging**: Should we track membership changes for compliance?
5. **Rate limiting**: Should rate limits be per-org (recommended for unified quota)?
6. **Email provider**: Which email provider to use for invitations? (Resend, SendGrid, AWS SES)

---

## Appendix: Personal Org Creation on Signup

When a new user signs up via Clerk, we need to create their personal organization
and set it as their active org in Clerk metadata.

```python
# /api/core/middleware/auth.py - Update User.get_or_create_from_auth()

@classmethod
async def get_or_create_from_auth(cls, auth_user: AuthenticatedUser) -> "User":
    """Get or create user from Clerk JWT, ensuring personal org exists."""
    user, created = await cls.get_or_create(
        user_id=auth_user.user_id,
        defaults={
            "email": auth_user.email,
            "first_name": auth_user.first_name,
            "last_name": auth_user.last_name,
            "claims": auth_user.claims,
        },
    )

    if created:
        # Create personal organization for new user
        personal_org = await Organization.create(
            name=f"{auth_user.first_name or auth_user.email}'s Workspace",
            slug=f"personal-{auth_user.user_id}",
            type=OrganizationType.PERSONAL,
            owner=user,
        )

        # Create owner membership
        await OrganizationMembership.create(
            organization=personal_org,
            user=user,
            role=OrganizationRole.OWNER,
            status=MembershipStatus.ACTIVE,
            joined_at=datetime.utcnow(),
        )

        # Create billing profile for personal org
        await BillingProfile.create(
            type=BillingProfileType.INDIVIDUAL,
            owner_organization=personal_org,
        )

        # Set personal org as active in Clerk metadata
        # This ensures the JWT includes active_organization_id on next token refresh
        await set_active_organization(auth_user.user_id, personal_org.id)

    return user
```

**Note:** On first request after signup, the JWT won't have `active_organization_id` yet
(since we just set it). The middleware falls back to personal org in this case.
Frontend should call `getToken()` again after first API call to get updated JWT.
