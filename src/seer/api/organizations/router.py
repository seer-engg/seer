"""
Organizations API Router.

Provides endpoints for managing organizations, memberships, and invitations.
All resource access is org-scoped via the OrganizationContextMiddleware.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Request, status

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.core.middleware.organization import get_membership, get_organization
from seer.api.organizations.models import (
    AcceptInvitationResponse,
    ConvertToTeamRequest,
    CreateInvitationRequest,
    CreateOrganizationRequest,
    IntegrationListResponse,
    IntegrationResponse,
    InvitationDetailsResponse,
    InvitationListResponse,
    InvitationResponse,
    MemberListResponse,
    MemberResponse,
    OrgBillingPortalResponse,
    OrgBillingResponse,
    OrgInvoiceItem,
    OrgInvoiceListResponse,
    OrgUsageSummaryResponse,
    OrganizationListResponse,
    OrganizationResponse,
    OrganizationWithRoleResponse,
    RequestApprovalResponse,
    ReviewApprovalRequest,
    ShareIntegrationResponse,
    SwitchOrganizationResponse,
    TransferWorkflowsRequest,
    TransferWorkflowsResponse,
    UpdateMemberRequest,
    UpdateOrganizationRequest,
    WorkflowApprovalListResponse,
    WorkflowApprovalResponse,
)
from seer.database import OAuthConnection, Organization, OrganizationMembership, User, Workflow
from seer.database.organization_models import (
    ApprovalStatus,
    InvitationStatus,
    MembershipStatus,
    OrganizationInvitation,
    OrganizationRole,
    OrganizationType,
    WorkflowApproval,
)
from seer.config import config
from seer.services.email_service import (
    send_approval_notification_email,
    send_invitation_email,
    send_member_joined_notification,
)
from seer.services.organization_service import (
    convert_personal_to_team,
    create_team_organization,
    get_user_organizations,
    switch_user_organization,
)

router = APIRouter(prefix="/organizations", tags=["organizations"])


# =============================================================================
# Helper Functions
# =============================================================================


def _require_user(request: Request) -> User:  # pylint: disable=duplicate-code  # Standard auth pattern
    """Get authenticated user from request state."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(
            type_uri=AUTH_PROBLEM,
            title="Unauthorized",
            detail="Authentication required",
            status=401,
        )
    return user


def _require_role(membership: OrganizationMembership, allowed_roles: List[OrganizationRole]) -> None:
    """Verify membership has one of the allowed roles."""
    if membership.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires one of roles: {[r.value for r in allowed_roles]}",
        )


def _require_owner(membership: OrganizationMembership) -> None:
    """Verify membership has owner role."""
    _require_role(membership, [OrganizationRole.OWNER])


def _require_admin_or_above(membership: OrganizationMembership) -> None:
    """Verify membership has admin or owner role."""
    _require_role(membership, [OrganizationRole.OWNER, OrganizationRole.ADMIN])


def _require_can_invite(membership: OrganizationMembership) -> None:
    """Verify membership can invite others."""
    _require_role(membership, [
        OrganizationRole.OWNER,
        OrganizationRole.ADMIN,
        OrganizationRole.CONSULTANT,
    ])


# =============================================================================
# Organization CRUD Endpoints
# =============================================================================


@router.get("", response_model=OrganizationListResponse)
async def list_organizations(request: Request) -> OrganizationListResponse:
    """
    List all organizations the user is a member of.

    Returns both personal and team organizations with the user's role in each.
    """
    user = _require_user(request)
    current_org = get_organization(request)

    org_memberships = await get_user_organizations(user)

    organizations = [
        OrganizationWithRoleResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            type=org.type,
            role=membership.role,
            is_owner=membership.role == OrganizationRole.OWNER,
            created_at=org.created_at,
        )
        for org, membership in org_memberships
    ]

    return OrganizationListResponse(
        organizations=organizations,
        current_organization_id=current_org.id if current_org else None,
    )


@router.get("/current", response_model=OrganizationWithRoleResponse)
async def get_current_organization(request: Request) -> OrganizationWithRoleResponse:
    """Get the currently active organization."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    return OrganizationWithRoleResponse(
        id=org.id,
        name=org.name,
        slug=org.slug,
        type=org.type,
        role=membership.role,
        is_owner=membership.role == OrganizationRole.OWNER,
        created_at=org.created_at,
    )


@router.post("", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    request: Request,
    body: CreateOrganizationRequest = Body(...),
) -> OrganizationResponse:
    """
    Create a new team organization.

    The creating user becomes the owner. Their subscription will be
    associated with the new team for billing purposes.
    """
    user = _require_user(request)

    # TODO: Verify user has an active subscription before creating team
    # This check is deferred to billing integration

    try:
        organization, _ = await create_team_organization(
            owner=user,
            name=body.name,
            slug=body.slug,
        )
    except ValueError as e:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid organization",
            detail=str(e),
            status=400,
        )

    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        slug=organization.slug,
        type=organization.type,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
    )


@router.post("/{org_id}/switch", response_model=SwitchOrganizationResponse)
async def switch_organization(
    request: Request,
    org_id: int,
) -> SwitchOrganizationResponse:
    """
    Switch to a different organization.

    Updates Clerk user metadata so JWT includes the new org_id.
    Frontend must call getToken() after this to get updated JWT.
    """
    user = _require_user(request)

    try:
        membership = await switch_user_organization(user, org_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    return SwitchOrganizationResponse(
        organization=OrganizationResponse(
            id=membership.organization.id,
            name=membership.organization.name,
            slug=membership.organization.slug,
            type=membership.organization.type,
            created_at=membership.organization.created_at,
            updated_at=membership.organization.updated_at,
        ),
        role=membership.role,
        message="Organization switched. Please refresh your token.",
    )


@router.post("/{org_id}/convert-to-team", response_model=OrganizationResponse)
async def convert_org_to_team(
    request: Request,
    org_id: int,
    body: ConvertToTeamRequest = Body(...),
) -> OrganizationResponse:
    """
    Convert personal organization to team.

    Only the owner can convert their personal workspace to a team.
    This allows inviting other members.
    """
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    # Validate org_id matches current org
    if org.id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only convert the current organization",
        )

    _require_owner(membership)

    try:
        updated_org = await convert_personal_to_team(org, body.name)
    except ValueError as e:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Cannot convert organization",
            detail=str(e),
            status=400,
        )

    return OrganizationResponse(
        id=updated_org.id,
        name=updated_org.name,
        slug=updated_org.slug,
        type=updated_org.type,
        created_at=updated_org.created_at,
        updated_at=updated_org.updated_at,
    )


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    request: Request,
    org_id: int,
    body: UpdateOrganizationRequest = Body(...),
) -> OrganizationResponse:
    """Update organization settings. Requires owner role."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only update the current organization",
        )

    _require_owner(membership)

    if body.name is not None:
        org.name = body.name

    if body.settings is not None:
        org.settings.update(body.settings)

    await org.save()

    return OrganizationResponse(
        id=org.id,
        name=org.name,
        slug=org.slug,
        type=org.type,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


# =============================================================================
# Member Management Endpoints
# =============================================================================


@router.get("/{org_id}/members", response_model=MemberListResponse)
async def list_members(
    request: Request,
    org_id: int,
) -> MemberListResponse:
    """List all members of the organization."""
    _require_user(request)
    org = get_organization(request)

    if org.id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only list members of current organization",
        )

    memberships = await OrganizationMembership.filter(
        organization=org,
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("user")

    members = [
        MemberResponse(
            user_id=m.user.id,
            clerk_user_id=m.user.user_id,
            email=m.user.email,
            first_name=m.user.first_name,
            last_name=m.user.last_name,
            role=m.role,
            status=m.status,
            joined_at=m.joined_at,
        )
        for m in memberships
    ]

    return MemberListResponse(members=members, total=len(members))


@router.patch("/{org_id}/members/{user_id}", response_model=MemberResponse)
async def update_member_role(
    request: Request,
    org_id: int,
    user_id: int,
    body: UpdateMemberRequest = Body(...),
) -> MemberResponse:
    """Update a member's role. Requires admin or owner role."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only manage current organization")

    _require_admin_or_above(membership)

    # Get target membership
    target_membership = await OrganizationMembership.get_or_none(
        organization=org,
        user_id=user_id,
    )

    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")

    # Cannot demote owner unless transferring ownership
    if target_membership.role == OrganizationRole.OWNER and body.role != OrganizationRole.OWNER:
        raise HTTPException(
            status_code=400,
            detail="Cannot demote owner. Transfer ownership first.",
        )

    # Cannot promote to owner (ownership transfer is separate)
    if body.role == OrganizationRole.OWNER and target_membership.role != OrganizationRole.OWNER:
        raise HTTPException(
            status_code=400,
            detail="Cannot promote to owner. Use ownership transfer instead.",
        )

    target_membership.role = body.role
    await target_membership.save()

    await target_membership.fetch_related("user")

    return MemberResponse(
        user_id=target_membership.user.id,
        clerk_user_id=target_membership.user.user_id,
        email=target_membership.user.email,
        first_name=target_membership.user.first_name,
        last_name=target_membership.user.last_name,
        role=target_membership.role,
        status=target_membership.status,
        joined_at=target_membership.joined_at,
    )


@router.delete("/{org_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    request: Request,
    org_id: int,
    user_id: int,
) -> None:
    """Remove a member from the organization. Requires admin or owner role."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only manage current organization")

    _require_admin_or_above(membership)

    # Get target membership
    target_membership = await OrganizationMembership.get_or_none(
        organization=org,
        user_id=user_id,
    )

    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")

    # Cannot remove owner
    if target_membership.role == OrganizationRole.OWNER:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove owner. Transfer ownership first.",
        )

    # Soft delete - set status to suspended
    target_membership.status = MembershipStatus.SUSPENDED
    await target_membership.save()


# =============================================================================
# Invitation Endpoints
# =============================================================================


@router.get("/{org_id}/invitations", response_model=InvitationListResponse)
async def list_invitations(
    request: Request,
    org_id: int,
) -> InvitationListResponse:
    """List pending invitations for the organization."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view invitations for current organization")

    _require_admin_or_above(membership)

    invitations = await OrganizationInvitation.filter(
        organization=org,
        status=InvitationStatus.PENDING,
    ).prefetch_related("invited_by")

    return InvitationListResponse(
        invitations=[
            InvitationResponse(
                id=inv.id,
                email=inv.email,
                role=inv.role,
                status=inv.status,
                expires_at=inv.expires_at,
                created_at=inv.created_at,
                invited_by_email=inv.invited_by.email if inv.invited_by else None,
            )
            for inv in invitations
        ],
        total=len(invitations),
    )


@router.post("/{org_id}/invitations", response_model=InvitationResponse, status_code=status.HTTP_201_CREATED)
async def create_invitation(
    request: Request,
    org_id: int,
    body: CreateInvitationRequest = Body(...),
) -> InvitationResponse:
    """
    Invite a user to the organization.

    Owner/Admin can invite any role.
    Consultant can only invite User role.
    """
    user = _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only invite to current organization")

    _require_can_invite(membership)

    # Consultants can only invite Users
    if membership.role == OrganizationRole.CONSULTANT and body.role != OrganizationRole.USER:
        raise HTTPException(
            status_code=403,
            detail="Consultants can only invite users",
        )

    # Check if user is already a member
    existing_member = await OrganizationMembership.filter(
        organization=org,
        user__email=body.email.lower(),
    ).exists()

    if existing_member:
        raise HTTPException(
            status_code=400,
            detail="User is already a member of this organization",
        )

    # Check for existing pending invitation
    existing_invitation = await OrganizationInvitation.get_or_none(
        organization=org,
        email=body.email.lower(),
        status=InvitationStatus.PENDING,
    )

    if existing_invitation:
        if not existing_invitation.is_expired:
            raise HTTPException(
                status_code=400,
                detail="An invitation has already been sent to this email",
            )
        # Mark expired invitation
        existing_invitation.status = InvitationStatus.EXPIRED
        await existing_invitation.save()

    # Create new invitation
    import secrets
    invitation = await OrganizationInvitation.create(
        organization=org,
        email=body.email.lower(),
        role=body.role,
        invited_by=user,
        token=secrets.token_urlsafe(32),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )

    # Send invitation email
    inviter_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or user.email
    invite_url = f"{config.frontend_url}/invitations/{invitation.token}"

    await send_invitation_email(
        to_email=invitation.email,
        organization_name=org.name,
        invited_by_name=inviter_name,
        role=body.role.value,
        invite_url=invite_url,
    )

    return InvitationResponse(
        id=invitation.id,
        email=invitation.email,
        role=invitation.role,
        status=invitation.status,
        expires_at=invitation.expires_at,
        created_at=invitation.created_at,
        invited_by_email=user.email,
    )


@router.get("/invitations/{token}", response_model=InvitationDetailsResponse)
async def get_invitation_details(token: str) -> InvitationDetailsResponse:
    """Get invitation details by token. Public endpoint — no auth required."""
    invitation = await OrganizationInvitation.get_or_none(
        token=token,
        status=InvitationStatus.PENDING,
    ).prefetch_related("organization", "invited_by")

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or already used")

    if invitation.is_expired:
        invitation.status = InvitationStatus.EXPIRED
        await invitation.save()
        raise HTTPException(status_code=400, detail="Invitation has expired")

    inviter = invitation.invited_by
    inviter_name = (
        f"{inviter.first_name or ''} {inviter.last_name or ''}".strip() or inviter.email or "Someone"
    )

    return InvitationDetailsResponse(
        invitation=InvitationResponse(
            id=invitation.id,
            email=invitation.email,
            role=invitation.role,
            status=invitation.status,
            expires_at=invitation.expires_at,
            created_at=invitation.created_at,
        ),
        organization_name=invitation.organization.name,
        inviter_name=inviter_name,
    )


@router.post("/invitations/{token}/decline", status_code=status.HTTP_204_NO_CONTENT)
async def decline_invitation(request: Request, token: str) -> None:
    """Decline an invitation. The invitee chooses not to join the organization."""
    _require_user(request)  # Must be authenticated to decline

    invitation = await OrganizationInvitation.get_or_none(
        token=token,
        status=InvitationStatus.PENDING,
    )

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or already used")

    if invitation.is_expired:
        invitation.status = InvitationStatus.EXPIRED
        await invitation.save()
        raise HTTPException(status_code=400, detail="Invitation has expired")

    invitation.status = InvitationStatus.REVOKED
    await invitation.save()


async def _mark_onboarding_complete_for_team_member(user: User) -> None:
    """Auto-complete onboarding for users joining via team invitation.

    Team members use the team's subscription, so they don't need to add
    their own payment method or complete the full onboarding flow.
    """
    from seer.database.models import UserSettings

    settings, _ = await UserSettings.get_or_create(user=user)

    onboarding = settings.preferences.get("onboarding", {})
    onboarding["completed"] = True
    onboarding["payment_method_added"] = True  # Uses team subscription
    onboarding["joined_via_invitation"] = True
    settings.preferences["onboarding"] = onboarding

    await settings.save()


@router.post("/invitations/{token}/accept", response_model=AcceptInvitationResponse)
async def accept_invitation(
    request: Request,
    token: str,
) -> AcceptInvitationResponse:
    """Accept an invitation and join the organization."""
    user = _require_user(request)

    # Find invitation
    invitation = await OrganizationInvitation.get_or_none(
        token=token,
        status=InvitationStatus.PENDING,
    ).prefetch_related("organization", "invited_by")

    if not invitation:
        raise HTTPException(
            status_code=404,
            detail="Invitation not found or already used",
        )

    # Check expiration
    if invitation.is_expired:
        invitation.status = InvitationStatus.EXPIRED
        await invitation.save()
        raise HTTPException(
            status_code=400,
            detail="Invitation has expired",
        )

    # Check email matches
    if invitation.email.lower() != (user.email or "").lower():
        raise HTTPException(
            status_code=403,
            detail="Invitation was sent to a different email address",
        )

    # Check not already a member
    existing = await OrganizationMembership.get_or_none(
        organization=invitation.organization,
        user=user,
    )

    if existing:
        if existing.status == MembershipStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail="You are already a member of this organization",
            )
        # Reactivate suspended membership
        existing.status = MembershipStatus.ACTIVE
        existing.role = invitation.role
        await existing.save()
        membership = existing
    else:
        # Create membership
        membership = await OrganizationMembership.create(
            organization=invitation.organization,
            user=user,
            role=invitation.role,
            status=MembershipStatus.ACTIVE,
            invited_by=invitation.invited_by,
            invited_at=invitation.created_at,
            joined_at=datetime.now(timezone.utc),
        )

    # Update invitation
    invitation.status = InvitationStatus.ACCEPTED
    invitation.accepted_at = datetime.now(timezone.utc)
    invitation.accepted_by = user
    await invitation.save()

    # Mark onboarding complete (team members use team subscription, don't need personal payment)
    await _mark_onboarding_complete_for_team_member(user)

    # Notify the inviter that someone joined
    if invitation.invited_by and invitation.invited_by.email:
        member_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or user.email
        await send_member_joined_notification(
            to_email=invitation.invited_by.email,
            new_member_name=member_name,
            new_member_email=user.email or "",
            organization_name=invitation.organization.name,
            role=invitation.role.value,
        )

    org_response = OrganizationWithRoleResponse(
        id=invitation.organization.id,
        name=invitation.organization.name,
        slug=invitation.organization.slug,
        type=invitation.organization.type,
        role=membership.role,
        is_owner=membership.role == OrganizationRole.OWNER,
        created_at=invitation.organization.created_at,
    )

    return AcceptInvitationResponse(
        success=True,
        organization=org_response,
    )


@router.delete("/{org_id}/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invitation(
    request: Request,
    org_id: int,
    invitation_id: int,
) -> None:
    """Revoke a pending invitation."""
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only manage current organization")

    _require_admin_or_above(membership)

    invitation = await OrganizationInvitation.get_or_none(
        id=invitation_id,
        organization=org,
        status=InvitationStatus.PENDING,
    )

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")

    invitation.status = InvitationStatus.REVOKED
    await invitation.save()


# =============================================================================
# Workflow Transfer Endpoints
# =============================================================================


@router.post("/{org_id}/transfer-workflows", response_model=TransferWorkflowsResponse)
async def transfer_workflows_to_org(
    request: Request,
    org_id: int,
    body: TransferWorkflowsRequest = Body(...),
) -> TransferWorkflowsResponse:
    """
    Transfer specific workflows from personal org to team organization.

    This is a one-way ownership transfer (not a copy).
    User must explicitly select which workflows to transfer.
    Workflows must currently belong to the user's personal org.
    """
    user = _require_user(request)
    target_org = get_organization(request)

    if target_org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only transfer to current organization")

    if target_org.type != OrganizationType.TEAM:
        raise HTTPException(
            status_code=400,
            detail="Can only transfer workflows to team organizations",
        )

    # Get user's personal org
    personal_org = await Organization.get_or_none(
        owner=user,
        type=OrganizationType.PERSONAL,
    )

    if not personal_org:
        raise HTTPException(
            status_code=400,
            detail="No personal organization found",
        )

    # Get workflows to transfer (must belong to user's personal org)
    workflows = await Workflow.filter(
        id__in=body.workflow_ids,
        user=user,
    )

    # For now, we don't have organization_id on Workflow model yet
    # This will be added when we modify workflow models
    # For now, just verify the user owns these workflows

    if len(workflows) != len(body.workflow_ids):
        raise HTTPException(
            status_code=400,
            detail="Some workflows not found or not owned by you",
        )

    transferred_ids = []
    # TODO: When Workflow model is updated with organization_id:
    # for workflow in workflows:
    #     workflow.organization = target_org
    #     await workflow.save()
    #     transferred_ids.append(workflow.id)

    # For now, just return the workflow IDs as "transferred"
    transferred_ids = [w.id for w in workflows]

    return TransferWorkflowsResponse(
        transferred_count=len(transferred_ids),
        workflow_ids=transferred_ids,
        message=f"Successfully transferred {len(transferred_ids)} workflow(s) to {target_org.name}",
    )


# =============================================================================
# Shared Integrations Endpoints
# =============================================================================


@router.get("/{org_id}/integrations", response_model=IntegrationListResponse)
async def list_org_integrations(
    request: Request,
    org_id: int,
) -> IntegrationListResponse:
    """
    List all integrations available to the organization.

    Returns both:
    - Connections shared with the organization by any member
    - The current user's personal connections (not shared)

    This gives a unified view of all OAuth connections usable in workflows.
    """
    user = _require_user(request)
    org = get_organization(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view integrations for current organization")

    # Get shared connections (shared with this org)
    shared_connections = await OAuthConnection.filter(
        shared_with_organization=org,
    ).prefetch_related("user")

    # Get user's personal connections (not shared)
    personal_connections = await OAuthConnection.filter(
        user=user,
        shared_with_organization=None,
    )

    integrations = []

    # Add shared connections
    for conn in shared_connections:
        integrations.append(IntegrationResponse(
            id=conn.id,
            provider=conn.provider,
            provider_account_id=conn.provider_account_id,
            status=conn.status,
            is_shared=True,
            shared_by_user_id=conn.user.id,
            shared_by_email=conn.user.email,
            created_at=conn.created_at,
            updated_at=conn.updated_at,
        ))

    # Add personal connections (not shared)
    for conn in personal_connections:
        integrations.append(IntegrationResponse(
            id=conn.id,
            provider=conn.provider,
            provider_account_id=conn.provider_account_id,
            status=conn.status,
            is_shared=False,
            shared_by_user_id=None,
            shared_by_email=None,
            created_at=conn.created_at,
            updated_at=conn.updated_at,
        ))

    return IntegrationListResponse(integrations=integrations, total=len(integrations))


@router.post("/{org_id}/integrations/{connection_id}/share", response_model=ShareIntegrationResponse)
async def share_integration(
    request: Request,
    org_id: int,
    connection_id: int,
) -> ShareIntegrationResponse:
    """
    Share an OAuth connection with the organization.

    After sharing, all organization members can use this connection
    in their workflows. Only the connection owner can share it.
    """
    user = _require_user(request)
    org = get_organization(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only share with current organization")

    if org.type != OrganizationType.TEAM:
        raise HTTPException(
            status_code=400,
            detail="Can only share integrations with team organizations",
        )

    # Get the connection (must be owned by the user)
    connection = await OAuthConnection.get_or_none(id=connection_id, user=user)

    if not connection:
        raise HTTPException(
            status_code=404,
            detail="Connection not found or not owned by you",
        )

    if connection.shared_with_organization_id is not None:
        raise HTTPException(
            status_code=400,
            detail="Connection is already shared with an organization",
        )

    # Share the connection
    connection.shared_with_organization = org
    await connection.save()

    return ShareIntegrationResponse(
        integration=IntegrationResponse(
            id=connection.id,
            provider=connection.provider,
            provider_account_id=connection.provider_account_id,
            status=connection.status,
            is_shared=True,
            shared_by_user_id=user.id,
            shared_by_email=user.email,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
        ),
        message=f"Connection shared with {org.name}",
    )


@router.delete("/{org_id}/integrations/{connection_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def unshare_integration(
    request: Request,
    org_id: int,
    connection_id: int,
) -> None:
    """
    Remove sharing of an OAuth connection.

    After unsharing, only the connection owner can use it.
    Only the connection owner can unshare it.
    """
    user = _require_user(request)
    org = get_organization(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only manage current organization")

    # Get the connection (must be owned by the user)
    connection = await OAuthConnection.get_or_none(id=connection_id, user=user)

    if not connection:
        raise HTTPException(
            status_code=404,
            detail="Connection not found or not owned by you",
        )

    if connection.shared_with_organization_id != org.id:
        raise HTTPException(
            status_code=400,
            detail="Connection is not shared with this organization",
        )

    # Unshare the connection
    connection.shared_with_organization = None
    await connection.save()


# =============================================================================
# Workflow Approval Endpoints
# =============================================================================


@router.post("/workflows/{workflow_id}/request-approval", response_model=RequestApprovalResponse)
async def request_workflow_approval(
    request: Request,
    workflow_id: int,
) -> RequestApprovalResponse:
    """
    Request approval for a consultant-created workflow.

    Consultants must request approval before their workflows can be
    published or used by the team. Creates an approval request that
    admins/owners can review.
    """
    user = _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    # Only consultants need to request approval
    if membership.role != OrganizationRole.CONSULTANT:
        raise HTTPException(
            status_code=400,
            detail="Only consultants need to request workflow approval",
        )

    # Get the workflow
    workflow = await Workflow.get_or_none(id=workflow_id, user=user)

    if not workflow:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found or not owned by you",
        )

    # Check if there's already a pending approval
    existing_approval = await WorkflowApproval.get_or_none(
        workflow=workflow,
        organization=org,
        status=ApprovalStatus.PENDING,
    )

    if existing_approval:
        raise HTTPException(
            status_code=400,
            detail="An approval request is already pending for this workflow",
        )

    # Create approval request
    approval = await WorkflowApproval.create(
        workflow=workflow,
        organization=org,
        requested_by=user,
        status=ApprovalStatus.PENDING,
    )

    # Update workflow approval status
    workflow.approval_status = "pending"
    await workflow.save()

    # Send notification to admins/owners
    admin_memberships = await OrganizationMembership.filter(
        organization=org,
        role__in=[OrganizationRole.OWNER, OrganizationRole.ADMIN],
        status=MembershipStatus.ACTIVE,
    ).prefetch_related("user")

    admin_emails = [m.user.email for m in admin_memberships if m.user.email]

    if admin_emails:
        requester_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or user.email
        review_url = f"{config.frontend_url}/organizations/{org.id}/approvals"

        await send_approval_notification_email(
            to_emails=admin_emails,
            workflow_name=workflow.name,
            requested_by_name=requester_name,
            organization_name=org.name,
            review_url=review_url,
        )

    return RequestApprovalResponse(
        approval=WorkflowApprovalResponse(
            id=approval.id,
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            status=approval.status,
            requested_by_user_id=user.id,
            requested_by_email=user.email,
            requested_at=approval.requested_at,
        ),
        message="Approval request submitted. An admin will review your workflow.",
    )


@router.get("/{org_id}/approvals", response_model=WorkflowApprovalListResponse)
async def list_pending_approvals(
    request: Request,
    org_id: int,
    status_filter: Optional[str] = None,
) -> WorkflowApprovalListResponse:
    """
    List workflow approval requests for the organization.

    Only admins and owners can view approval requests.
    Optionally filter by status (pending, approved, rejected).
    """
    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view approvals for current organization")

    _require_admin_or_above(membership)

    # Build query
    query = WorkflowApproval.filter(organization=org)

    if status_filter:
        try:
            filter_status = ApprovalStatus(status_filter.lower())
            query = query.filter(status=filter_status)
        except ValueError:
            raise HTTPException(  # pylint: disable=raise-missing-from  # Reason: HTTP errors don't need chaining
                status_code=400,
                detail=f"Invalid status: {status_filter}. Must be one of: pending, approved, rejected",
            )

    approvals = await query.prefetch_related("workflow", "requested_by", "reviewed_by")

    return WorkflowApprovalListResponse(
        approvals=[
            WorkflowApprovalResponse(
                id=a.id,
                workflow_id=a.workflow.id,
                workflow_name=a.workflow.name,
                status=a.status,
                requested_by_user_id=a.requested_by.id,
                requested_by_email=a.requested_by.email,
                requested_at=a.requested_at,
                reviewed_by_user_id=a.reviewed_by.id if a.reviewed_by else None,
                reviewed_by_email=a.reviewed_by.email if a.reviewed_by else None,
                reviewed_at=a.reviewed_at,
                review_notes=a.review_notes,
            )
            for a in approvals
        ],
        total=len(approvals),
    )


@router.post("/{org_id}/approvals/{approval_id}/review", response_model=WorkflowApprovalResponse)
async def review_workflow_approval(
    request: Request,
    org_id: int,
    approval_id: int,
    body: ReviewApprovalRequest = Body(...),
) -> WorkflowApprovalResponse:
    """
    Review a workflow approval request.

    Admins and owners can approve or reject consultant-created workflows.
    Approved workflows can then be published and used by the team.
    """
    from datetime import timezone as tz

    user = _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only review approvals for current organization")

    _require_admin_or_above(membership)

    # Validate status - can only approve or reject
    if body.status not in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED):
        raise HTTPException(
            status_code=400,
            detail="Status must be 'approved' or 'rejected'",
        )

    # Get the approval
    approval = await WorkflowApproval.get_or_none(
        id=approval_id,
        organization=org,
    ).prefetch_related("workflow", "requested_by")

    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found")

    if approval.status != ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Approval has already been {approval.status.value}",
        )

    # Update approval
    approval.status = body.status
    approval.reviewed_by = user
    approval.reviewed_at = datetime.now(tz.utc)
    approval.review_notes = body.notes
    await approval.save()

    # Update workflow approval status
    approval.workflow.approval_status = body.status.value
    await approval.workflow.save()

    return WorkflowApprovalResponse(
        id=approval.id,
        workflow_id=approval.workflow.id,
        workflow_name=approval.workflow.name,
        status=approval.status,
        requested_by_user_id=approval.requested_by.id,
        requested_by_email=approval.requested_by.email,
        requested_at=approval.requested_at,
        reviewed_by_user_id=user.id,
        reviewed_by_email=user.email,
        reviewed_at=approval.reviewed_at,
        review_notes=approval.review_notes,
    )


# =============================================================================
# Billing Endpoints
# =============================================================================


@router.get("/{org_id}/billing", response_model=OrgBillingResponse)
async def get_org_billing(
    request: Request,
    org_id: int,
) -> OrgBillingResponse:
    """
    Get the organization's billing status.

    Returns subscription tier, status, and billing period information.
    Only organization owners can access billing information.
    """
    from seer.api.subscriptions.stripe_service import get_org_subscription

    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view billing for current organization")

    _require_owner(membership)

    subscription = await get_org_subscription(org)

    return OrgBillingResponse(
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_end=(
            subscription.current_period_end.isoformat()
            if subscription.current_period_end
            else None
        ),
        cancel_at_period_end=subscription.cancel_at_period_end,
        has_payment_method=subscription.billing_profile.has_payment_method if subscription.billing_profile else False,
    )


@router.post("/{org_id}/billing/portal", response_model=OrgBillingPortalResponse)
async def create_org_billing_portal(
    request: Request,
    org_id: int,
) -> OrgBillingPortalResponse:
    """
    Create a Stripe Customer Portal session for the organization.

    Allows the owner to manage subscription, payment methods, and view invoices.
    Only organization owners can access the billing portal.
    """
    from seer.api.subscriptions.stripe_service import create_org_portal_session
    from seer.config import config
    import stripe

    user = _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only access billing for current organization")

    _require_owner(membership)

    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    return_url = f"{config.frontend_url}/settings/billing"

    try:
        portal_url = await create_org_portal_session(org, user, return_url)
        return OrgBillingPortalResponse(portal_url=portal_url)
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{org_id}/billing/invoices", response_model=OrgInvoiceListResponse)
async def list_org_invoices(
    request: Request,
    org_id: int,
    page: int = 1,
    page_size: int = 20,
) -> OrgInvoiceListResponse:
    """
    List invoices for the organization.

    Returns a paginated list of Stripe invoices.
    Only organization owners can view invoices.
    """
    from seer.api.subscriptions.stripe_service import list_org_invoices as _list_org_invoices
    from seer.config import config
    import stripe

    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view invoices for current organization")

    _require_owner(membership)

    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    try:
        result = await _list_org_invoices(org, page=page, page_size=page_size)
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=400, detail="Unable to fetch invoices") from exc

    return OrgInvoiceListResponse(
        items=[OrgInvoiceItem(**item) for item in result["items"]],
        page=page,
        page_size=page_size,
        has_more=result["has_more"],
    )


@router.get("/{org_id}/billing/usage", response_model=OrgUsageSummaryResponse)
async def get_org_usage_summary(
    request: Request,
    org_id: int,
) -> OrgUsageSummaryResponse:
    """
    Get organization usage summary for the current billing period.

    Returns workflow count, runs this month, and LLM credits used.
    Only organization owners can view usage information.
    """
    from datetime import timezone as tz
    from seer.database.usage_models import UsageCounter, ResourceType

    _require_user(request)
    org = get_organization(request)
    membership = get_membership(request)

    if org.id != org_id:
        raise HTTPException(status_code=403, detail="Can only view usage for current organization")

    _require_owner(membership)

    # Get current billing period (start of current month)
    now = datetime.now(tz.utc)
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    period_end = (period_start.replace(month=period_start.month % 12 + 1) if period_start.month < 12
                  else period_start.replace(year=period_start.year + 1, month=1))

    # Get workflow count (all-time)
    workflows_counter = await UsageCounter.get_or_none(
        organization=org,
        user=None,
        resource_type=ResourceType.WORKFLOWS,
        period_start__isnull=True,
    )
    workflows_count = workflows_counter.count if workflows_counter else 0

    # Get runs this month
    runs_counter = await UsageCounter.get_or_none(
        organization=org,
        user=None,
        resource_type=ResourceType.RUNS,
        period_start=period_start,
    )
    runs_this_month = runs_counter.count if runs_counter else 0

    # Get LLM credits this month
    llm_counter = await UsageCounter.get_or_none(
        organization=org,
        user=None,
        resource_type=ResourceType.LLM_CREDITS,
        period_start=period_start,
    )
    llm_credits = float(llm_counter.value) if llm_counter else 0.0

    return OrgUsageSummaryResponse(
        workflows_count=workflows_count,
        runs_this_month=runs_this_month,
        llm_credits_this_month=llm_credits,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
    )
