"""
Role-based permission decorators and helpers for organization access control.

This module provides decorators and functions to enforce role-based access
control in API endpoints. All permissions are based on the user's membership
role in the current organization.

Usage:
    from seer.api.core.permissions import require_admin_or_above, can_manage_workflow

    @router.post("/some-admin-action")
    @require_admin_or_above()
    async def admin_action(request: Request):
        # Only admins and owners can access this
        ...

    @router.patch("/workflows/{workflow_id}")
    async def update_workflow(request: Request, workflow_id: int):
        workflow = await get_workflow(workflow_id)
        membership = get_membership(request)
        user = get_user(request)

        if not await can_manage_workflow(user, workflow, membership):
            raise HTTPException(403, "Cannot manage this workflow")
        ...
"""
from functools import wraps
from typing import Callable, List, TypeVar

from fastapi import HTTPException, Request

from seer.database import OrganizationMembership, User, Workflow
from seer.database.organization_models import (
    OrganizationRole,
    WorkflowAssignment,
)
from seer.database.workflow_models import WorkflowVisibility

F = TypeVar("F", bound=Callable)


# =============================================================================
# Permission Decorators
# =============================================================================


def require_role(allowed_roles: List[OrganizationRole]):
    """
    Decorator to enforce role-based access.

    Checks that the user's membership in the current organization
    has one of the allowed roles.

    Args:
        allowed_roles: List of roles that are allowed access

    Example:
        @router.post("/admin-only")
        @require_role([OrganizationRole.OWNER, OrganizationRole.ADMIN])
        async def admin_only(request: Request):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            membership = getattr(request.state, "membership", None)

            if membership is None:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                )

            if membership.role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Requires one of roles: {[r.value for r in allowed_roles]}",
                )

            return await func(request, *args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def require_owner():
    """Decorator requiring owner role."""
    return require_role([OrganizationRole.OWNER])


def require_admin_or_above():
    """Decorator requiring admin or owner role."""
    return require_role([OrganizationRole.OWNER, OrganizationRole.ADMIN])


def require_member():
    """Decorator requiring any active membership (any role)."""
    return require_role([
        OrganizationRole.OWNER,
        OrganizationRole.ADMIN,
        OrganizationRole.USER,
        OrganizationRole.CONSULTANT,
    ])


def require_can_invite():
    """Decorator requiring permission to invite members."""
    return require_role([
        OrganizationRole.OWNER,
        OrganizationRole.ADMIN,
        OrganizationRole.CONSULTANT,
    ])


def require_can_manage_billing():
    """Decorator requiring permission to manage billing."""
    return require_role([OrganizationRole.OWNER])


# =============================================================================
# Permission Check Functions
# =============================================================================


async def can_manage_workflow(
    user: User,
    workflow: Workflow,
    membership: OrganizationMembership,
) -> bool:
    """
    Check if user can edit/delete a workflow.

    - Owner/Admin: Can manage all workflows in the org
    - User: Can manage only their own workflows
    - Consultant: Cannot manage workflows (only assigned for viewing)

    Args:
        user: The user attempting the action
        workflow: The workflow being managed
        membership: User's membership in the organization

    Returns:
        True if user can manage the workflow
    """
    if membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
        return True

    if membership.role == OrganizationRole.USER:
        # Users can only manage their own workflows
        return workflow.user_id == user.id

    if membership.role == OrganizationRole.CONSULTANT:
        # Consultants cannot manage workflows
        return False

    return False


async def can_view_workflow(
    user: User,
    workflow: Workflow,
    membership: OrganizationMembership,
) -> bool:
    """
    Check if user can view a workflow.

    Visibility rules:
    - TEAM: All org members can view
    - PRIVATE: Only creator can view
    - ASSIGNED: Only assigned users can view

    Owner/Admin can always view all workflows.
    Consultants can only view assigned workflows.

    Args:
        user: The user attempting to view
        workflow: The workflow being viewed
        membership: User's membership in the organization

    Returns:
        True if user can view the workflow
    """
    # Owner and Admin can see everything
    if membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
        return True

    # Check workflow visibility
    visibility = getattr(workflow, "visibility", WorkflowVisibility.TEAM)

    if membership.role == OrganizationRole.USER:
        if visibility == WorkflowVisibility.TEAM:
            return True
        if visibility == WorkflowVisibility.PRIVATE:
            return workflow.user_id == user.id
        if visibility == WorkflowVisibility.ASSIGNED:
            return await WorkflowAssignment.exists(workflow=workflow, user=user)

    if membership.role == OrganizationRole.CONSULTANT:
        # Consultants can only view assigned workflows
        return await WorkflowAssignment.exists(workflow=workflow, user=user)

    return False


async def can_publish_workflow(
    user: User,
    workflow: Workflow,
    membership: OrganizationMembership,
) -> bool:
    """
    Check if user can publish/release a workflow.

    - Owner/Admin: Can publish any workflow
    - User: Can publish their own workflows
    - Consultant: Must request approval (cannot publish directly)

    Args:
        user: The user attempting to publish
        workflow: The workflow being published
        membership: User's membership in the organization

    Returns:
        True if user can publish the workflow
    """
    if membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
        return True

    if membership.role == OrganizationRole.USER:
        return workflow.user_id == user.id

    # Consultants cannot publish directly - need approval
    return False


async def can_approve_workflows(membership: OrganizationMembership) -> bool:
    """
    Check if user can approve consultant workflow submissions.

    Only Owner and Admin can approve workflows.
    """
    return membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN)


async def can_delete_organization(membership: OrganizationMembership) -> bool:
    """
    Check if user can delete the organization.

    Only Owner can delete the organization.
    """
    return membership.role == OrganizationRole.OWNER


async def can_transfer_ownership(membership: OrganizationMembership) -> bool:
    """
    Check if user can transfer organization ownership.

    Only the current Owner can transfer ownership.
    """
    return membership.role == OrganizationRole.OWNER


# =============================================================================
# Helper Functions
# =============================================================================


def check_role(
    membership: OrganizationMembership,
    allowed_roles: List[OrganizationRole],
) -> None:
    """
    Check if membership has one of the allowed roles.

    Raises HTTPException if role check fails.

    Args:
        membership: User's membership in the organization
        allowed_roles: List of allowed roles

    Raises:
        HTTPException: If role is not in allowed_roles
    """
    if membership.role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Requires one of roles: {[r.value for r in allowed_roles]}",
        )


def check_owner(membership: OrganizationMembership) -> None:
    """Check that membership has owner role."""
    check_role(membership, [OrganizationRole.OWNER])


def check_admin_or_above(membership: OrganizationMembership) -> None:
    """Check that membership has admin or owner role."""
    check_role(membership, [OrganizationRole.OWNER, OrganizationRole.ADMIN])


def check_can_invite(membership: OrganizationMembership) -> None:
    """Check that membership can invite others."""
    check_role(membership, [
        OrganizationRole.OWNER,
        OrganizationRole.ADMIN,
        OrganizationRole.CONSULTANT,
    ])
