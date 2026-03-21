"""
Organization context middleware.

This middleware extracts the active organization from the database and
attaches the organization and membership to the request state.

The active_organization_id is stored on the User row and repaired
automatically when it is missing or points to an invalid membership.
"""
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from seer.database import Organization, OrganizationMembership
from seer.database.organization_models import MembershipStatus, OrganizationType
from seer.logger import get_logger

logger = get_logger("api.middleware.organization")


class OrganizationContextMiddleware(BaseHTTPMiddleware):
    """
    Extracts organization context from the authenticated database user.

    After authentication, this middleware:
    1. Reads active_organization_id from the database user
    2. Validates user's membership in that organization
    3. Attaches organization and membership to request.state
    4. Falls back to personal org if no valid org is specified
    5. Repairs persisted active_organization_id when needed

    Request State (after middleware):
        - request.state.organization: Organization model instance
        - request.state.membership: OrganizationMembership model instance
    """

    async def dispatch(self, request: Request, call_next):
        # Initialize organization state to None
        request.state.organization = None
        request.state.membership = None

        # Skip if no authenticated user
        db_user = getattr(request.state, "db_user", None)
        if db_user is None:
            return await call_next(request)

        org_id = self._extract_org_id(db_user)

        if org_id:
            # Validate membership in the specified organization
            membership = await self._get_active_membership(db_user, org_id)
            if membership:
                await membership.fetch_related("organization")
                request.state.organization = membership.organization
                request.state.membership = membership
                return await call_next(request)
            # User claims an org they're not a member of
            logger.warning(
                "User %s has invalid active org %s with no active membership",
                db_user.user_id,
                org_id,
            )
            # Fall through to personal org

        # Default to personal org
        org_result = await self._get_or_create_personal_org(db_user)
        if org_result is None:
            # This shouldn't happen in normal flow, but handle gracefully
            logger.error("Failed to get/create personal org for user %s", db_user.user_id)
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to initialize user organization"},
            )

        organization, membership = org_result
        await self._persist_active_org(db_user, organization.id)
        request.state.organization = organization
        request.state.membership = membership

        return await call_next(request)

    def _extract_org_id(self, db_user) -> Optional[int]:
        """
        Extract active_organization_id from the authenticated DB user.
        """
        org_id = getattr(db_user, "active_organization_id", None)

        if org_id is None:
            return None

        try:
            return int(org_id)
        except (ValueError, TypeError):
            logger.warning("Invalid active_organization_id on user %s: %s", db_user.user_id, org_id)
            return None

    async def _get_active_membership(
        self,
        db_user,
        org_id: int,
    ) -> Optional[OrganizationMembership]:
        """
        Get the user's active membership in the specified organization.
        """
        return await OrganizationMembership.get_or_none(
            organization_id=org_id,
            user=db_user,
            status=MembershipStatus.ACTIVE,
        )

    async def _get_or_create_personal_org(
        self,
        db_user,
    ) -> Optional[tuple[Organization, OrganizationMembership]]:
        """
        Get or create the user's personal organization.

        Every user has exactly one personal organization created on signup.
        This method handles the edge case where signup flow might have failed
        to create the personal org.
        """
        # Try to find existing personal org
        personal_org = await Organization.get_or_none(
            owner=db_user,
            type=OrganizationType.PERSONAL,
        )

        if personal_org:
            membership = await OrganizationMembership.get_or_none(
                organization=personal_org,
                user=db_user,
            )
            if membership:
                return personal_org, membership

        # Personal org doesn't exist - this is a legacy user or signup failed
        # Create it now (should be rare in production)
        logger.info("Creating missing personal org for user %s", db_user.user_id)

        # pylint: disable-next=import-outside-toplevel  # Reason: avoids circular import between middleware and service layer
        from seer.services.organization_service import create_personal_organization
        return await create_personal_organization(db_user)

    async def _persist_active_org(self, db_user, org_id: int) -> None:
        """Persist active organization when fallback or repair logic changes it."""
        if getattr(db_user, "active_organization_id", None) == org_id:
            return

        db_user.active_organization_id = org_id
        await db_user.save(update_fields=["active_organization_id"])


def get_organization(request: Request) -> Organization:
    """
    Get the current organization from request state.

    Use this in route handlers to get the authenticated organization.

    Raises:
        ValueError: If no organization is set (should not happen after middleware)
    """
    org = getattr(request.state, "organization", None)
    if org is None:
        raise ValueError("No organization in request state - middleware may have failed")
    return org


def get_membership(request: Request) -> OrganizationMembership:
    """
    Get the current user's membership from request state.

    Use this in route handlers to get the authenticated membership
    (which includes the user's role in the organization).

    Raises:
        ValueError: If no membership is set (should not happen after middleware)
    """
    membership = getattr(request.state, "membership", None)
    if membership is None:
        raise ValueError("No membership in request state - middleware may have failed")
    return membership


def get_org_and_membership(request: Request) -> tuple[Organization, OrganizationMembership]:
    """
    Get both organization and membership from request state.

    Convenience function for handlers that need both.
    """
    return get_organization(request), get_membership(request)
