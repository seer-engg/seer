# pylint: disable=broad-exception-caught
# Reason: API router needs graceful error handling for user-facing responses
"""
Browser profile management API endpoints.

Provides endpoints for creating, listing, and managing browser profiles
that store authenticated sessions for browser automation workflows.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from seer.database import User
from seer.logger import get_logger
from seer.services.browser import BrowserProfileManager

logger = get_logger("api.browser.router")

router = APIRouter(prefix="/browser", tags=["browser"])


# ------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------
class CreateProfileRequest(BaseModel):
    """Request to create a new browser profile."""
    name: str = Field(min_length=1, max_length=100)


class ProfileResponse(BaseModel):
    """Browser profile metadata response."""
    id: str
    name: str
    logged_in_domains: List[str]
    created_at: Optional[str]
    last_used_at: Optional[str]


class LoginRequest(BaseModel):
    """Request to start interactive login session."""
    target_url: Optional[str] = Field(
        default=None,
        description="Optional starting URL for login (e.g., 'https://slack.com/signin')"
    )


class LoginResponse(BaseModel):
    """Response from interactive login session."""
    profile_id: str
    logged_in_domains: List[str]
    status: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@router.post("/profiles", response_model=ProfileResponse)
async def create_profile(
    request: Request,
    body: CreateProfileRequest,
) -> ProfileResponse:
    """
    Create a new browser profile.

    Browser profiles store authenticated sessions (cookies, localStorage)
    that can be reused across workflow executions.
    """
    user: User = request.state.db_user
    logger.info("Creating browser profile '%s' for user %s", body.name, user.user_id)

    manager = BrowserProfileManager()
    try:
        profile = await manager.create_profile(user, body.name)
    except Exception as e:
        if "UNIQUE constraint" in str(e) or "duplicate key" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail=f"A profile with name '{body.name}' already exists"
            ) from e
        logger.error("Failed to create profile: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create profile") from e

    return ProfileResponse(
        id=str(profile.id),
        name=profile.name,
        logged_in_domains=[],
        created_at=profile.created_at.isoformat() if profile.created_at else None,
        last_used_at=None,
    )


@router.get("/profiles", response_model=List[ProfileResponse])
async def list_profiles(request: Request) -> List[ProfileResponse]:
    """
    List all browser profiles for the current user.

    Returns profile metadata including which domains have authenticated sessions.
    """
    user: User = request.state.db_user
    manager = BrowserProfileManager()
    profiles = await manager.list_profiles(user)

    return [
        ProfileResponse(
            id=p["id"],
            name=p["name"],
            logged_in_domains=p["logged_in_domains"],
            created_at=p["created_at"],
            last_used_at=p["last_used_at"],
        )
        for p in profiles
    ]


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    request: Request,
    profile_id: UUID,
) -> ProfileResponse:
    """Get a specific browser profile by ID."""
    user: User = request.state.db_user
    manager = BrowserProfileManager()
    profile = await manager.get_profile(user, profile_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return ProfileResponse(
        id=str(profile.id),
        name=profile.name,
        logged_in_domains=profile.logged_in_domains or [],
        created_at=profile.created_at.isoformat() if profile.created_at else None,
        last_used_at=profile.last_used_at.isoformat() if profile.last_used_at else None,
    )


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    request: Request,
    profile_id: UUID,
) -> dict:
    """
    Delete a browser profile.

    This soft-deletes the profile. The encrypted session data is removed.
    """
    user: User = request.state.db_user
    logger.info("Deleting browser profile %s for user %s", profile_id, user.user_id)

    manager = BrowserProfileManager()
    deleted = await manager.delete_profile(user, profile_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"deleted": True, "profile_id": str(profile_id)}


@router.post("/profiles/{profile_id}/login", response_model=LoginResponse)
async def start_login(
    request: Request,
    profile_id: UUID,
    body: LoginRequest,
) -> LoginResponse:
    """
    Start an interactive login session.

    .. deprecated::
        Use POST /api/browser/sessions to create a streaming session,
        connect via WebSocket, then POST /api/browser/sessions/{id}/complete.

    Opens a browser window where the user can log into services.
    The session (cookies, localStorage) is captured and saved when
    the browser is closed.

    **Note:** This endpoint is intended for local/development use.
    The browser window opens on the server machine.
    """
    user: User = request.state.db_user
    logger.info(
        "Starting interactive login for profile %s, target_url=%s",
        profile_id, body.target_url
    )

    manager = BrowserProfileManager()

    # Verify profile exists
    profile = await manager.get_profile(user, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        result = await manager.start_interactive_login(
            user, profile_id, body.target_url
        )
    except Exception as e:
        logger.error("Interactive login failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start login session: {str(e)}"
        ) from e

    return LoginResponse(
        profile_id=result["profile_id"],
        logged_in_domains=result["logged_in_domains"],
        status=result["status"],
    )
