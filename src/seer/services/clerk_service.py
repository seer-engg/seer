"""
Clerk Backend API service for user metadata management.

This module provides functions to update Clerk user metadata, which is used
to store the user's active organization ID for JWT claims.

Note: We use direct HTTP calls instead of the clerk-backend-api SDK to avoid
adding an extra dependency. The httpx library is already available.
"""
from typing import Any, Dict, Optional

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

# Clerk API base URL
CLERK_API_BASE = "https://api.clerk.com/v1"


class ClerkServiceError(Exception):
    """Error interacting with Clerk API."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def _get_clerk_headers() -> Dict[str, str]:
    """Get headers for Clerk API requests."""
    secret_key = config.clerk_secret_key
    if not secret_key:
        raise ClerkServiceError("CLERK_SECRET_KEY not configured")

    return {
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": "application/json",
    }


async def get_clerk_user(user_id: str) -> Dict[str, Any]:
    """
    Get a user from Clerk by their user ID.

    Args:
        user_id: The Clerk user ID (e.g., "user_xxx")

    Returns:
        User data from Clerk API

    Raises:
        ClerkServiceError: If the API call fails
    """
    headers = await _get_clerk_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{CLERK_API_BASE}/users/{user_id}",
            headers=headers,
        )

        if response.status_code == 404:
            raise ClerkServiceError(f"User not found: {user_id}", status_code=404)

        if not response.is_success:
            logger.error(
                "Clerk API error getting user: %s %s",
                response.status_code,
                response.text[:200],
            )
            raise ClerkServiceError(
                f"Failed to get Clerk user: {response.status_code}",
                status_code=response.status_code,
            )

        return response.json()


async def update_clerk_user_metadata(
    user_id: str,
    public_metadata: Optional[Dict[str, Any]] = None,
    private_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Update Clerk user metadata.

    This is used for storing active_organization_id for JWT claims.
    The metadata is merged with existing values (not replaced).

    Args:
        user_id: The Clerk user ID (e.g., "user_xxx")
        public_metadata: Public metadata to merge (visible in JWT and client)
        private_metadata: Private metadata to merge (server-side only)

    Returns:
        Updated user data from Clerk API

    Raises:
        ClerkServiceError: If the API call fails
    """
    headers = await _get_clerk_headers()

    # Build the update payload
    # Clerk's PATCH endpoint merges metadata by default
    update_data: Dict[str, Any] = {}

    if public_metadata:
        update_data["public_metadata"] = public_metadata

    if private_metadata:
        update_data["private_metadata"] = private_metadata

    if not update_data:
        # Nothing to update, just get current user
        return await get_clerk_user(user_id)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.patch(
            f"{CLERK_API_BASE}/users/{user_id}",
            headers=headers,
            json=update_data,
        )

        if response.status_code == 404:
            raise ClerkServiceError(f"User not found: {user_id}", status_code=404)

        if not response.is_success:
            logger.error(
                "Clerk API error updating user metadata: %s %s",
                response.status_code,
                response.text[:200],
            )
            raise ClerkServiceError(
                f"Failed to update Clerk user metadata: {response.status_code}",
                status_code=response.status_code,
            )

        logger.debug("Updated Clerk metadata for user %s", user_id)
        return response.json()


async def set_active_organization(user_id: str, org_id: int) -> Dict[str, Any]:
    """
    Set the user's active organization in Clerk metadata.

    This updates the public_metadata.active_organization_id field,
    which is included in the JWT via the Clerk JWT template.

    After calling this, the frontend should call getToken() to get
    a new JWT with the updated organization ID.

    Args:
        user_id: The Clerk user ID (e.g., "user_xxx")
        org_id: The organization ID to set as active

    Returns:
        Updated user data from Clerk API
    """
    logger.info("Setting active organization for user %s to org %s", user_id, org_id)
    return await update_clerk_user_metadata(
        user_id=user_id,
        public_metadata={"active_organization_id": org_id},
    )


async def clear_active_organization(user_id: str) -> Dict[str, Any]:
    """
    Clear the user's active organization (revert to personal org).

    Sets active_organization_id to None in public metadata.
    The middleware will fall back to the user's personal org.

    Args:
        user_id: The Clerk user ID (e.g., "user_xxx")

    Returns:
        Updated user data from Clerk API
    """
    logger.info("Clearing active organization for user %s", user_id)
    return await update_clerk_user_metadata(
        user_id=user_id,
        public_metadata={"active_organization_id": None},
    )


async def get_user_active_organization_id(user_id: str) -> Optional[int]:
    """
    Get the user's currently active organization ID from Clerk metadata.

    Args:
        user_id: The Clerk user ID (e.g., "user_xxx")

    Returns:
        The active organization ID, or None if not set
    """
    try:
        user_data = await get_clerk_user(user_id)
        public_metadata = user_data.get("public_metadata", {})
        org_id = public_metadata.get("active_organization_id")
        return int(org_id) if org_id is not None else None
    except ClerkServiceError:
        logger.warning("Could not get active organization for user %s", user_id)
        return None
