"""User profile API endpoints."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, ConfigDict, field_validator

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.database import User
from seer.database.profile_models import UserProfile, validate_username

router = APIRouter(prefix="/users/me", tags=["user-profile"])


class UserProfileResponse(BaseModel):
    """Response model for user profile."""
    model_config = ConfigDict(from_attributes=True)

    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    social_links: Dict[str, Any] = {}
    tags: List[str] = []
    is_public: bool = False


class UserProfileUpdate(BaseModel):
    """Request model for updating user profile."""
    username: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    social_links: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None

    @field_validator("username")
    @classmethod
    def check_username(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_username(v)
        return v


def _require_user(request: Request) -> User:  # pylint: disable=duplicate-code  # Standard auth pattern duplicated across routers
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(
            type_uri=AUTH_PROBLEM,
            title="Unauthorized",
            detail="Authentication required",
            status=401,
        )
    return user


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(request: Request):
    """Get current user's profile."""
    user = _require_user(request)
    profile = await UserProfile.filter(user=user).first()
    if profile is None:
        return UserProfileResponse(username="")
    return UserProfileResponse.model_validate(profile, from_attributes=True)


@router.patch("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    request: Request,
    update_data: UserProfileUpdate = Body(...),
):
    """Create or update current user's profile."""
    user = _require_user(request)
    profile = await UserProfile.filter(user=user).first()

    if profile is None:
        # Creating new profile — username is required
        if not update_data.username:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Username required",
                detail="Username is required when creating a profile",
                status=400,
            )
        # Check uniqueness
        if await UserProfile.filter(username=update_data.username).exists():
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Username taken",
                detail=f"Username '{update_data.username}' is already in use",
                status=409,
            )
        profile = await UserProfile.create(
            user=user,
            username=update_data.username,
            display_name=update_data.display_name,
            bio=update_data.bio,
            avatar_url=update_data.avatar_url,
            social_links=update_data.social_links or {},
            tags=update_data.tags or [],
            is_public=update_data.is_public or False,
        )
    else:
        # Updating existing profile
        if update_data.username is not None and update_data.username != profile.username:
            if await UserProfile.filter(username=update_data.username).exists():
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Username taken",
                    detail=f"Username '{update_data.username}' is already in use",
                    status=409,
                )
            profile.username = update_data.username

        if update_data.display_name is not None:
            profile.display_name = update_data.display_name
        if update_data.bio is not None:
            profile.bio = update_data.bio
        if update_data.avatar_url is not None:
            profile.avatar_url = update_data.avatar_url
        if update_data.social_links is not None:
            profile.social_links = update_data.social_links
        if update_data.tags is not None:
            profile.tags = update_data.tags
        if update_data.is_public is not None:
            profile.is_public = update_data.is_public

        await profile.save()

    return UserProfileResponse.model_validate(profile, from_attributes=True)
