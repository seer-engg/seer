"""User profile API endpoints."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Request, status as http_status
from pydantic import BaseModel, ConfigDict, field_validator

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.database import User
from seer.database.profile_models import UserProfile, validate_username
from seer.database.template_models import TemplateSource, WorkflowTemplate

router = APIRouter(prefix="/users/me", tags=["user-profile"])


class UserProfileResponse(BaseModel):
    """Response model for user profile."""
    model_config = ConfigDict(from_attributes=True)

    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    avatar_file_id: Optional[str] = None
    social_links: Dict[str, Any] = {}
    tags: List[str] = []
    is_public: bool = False


class UserProfileUpdate(BaseModel):
    """Request model for updating user profile."""
    username: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_file_id: Optional[str] = None
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
    response = UserProfileResponse.model_validate(profile, from_attributes=True)
    if profile.avatar_file_id and config.is_workflow_file_system_configured:
        response.avatar_url = await _resolve_avatar_url(profile.avatar_file_id)
    return response


async def _resolve_avatar_url(file_id: str) -> Optional[str]:
    """Generate a fresh presigned URL for an avatar file_id."""
    from seer.core.files.service import WorkflowFileSystem  # pylint: disable=import-outside-toplevel  # Avoid circular imports

    try:
        fs = WorkflowFileSystem.instance()
        from seer.database.workflow_models import WorkflowFile  # pylint: disable=import-outside-toplevel  # Avoid circular imports
        from seer.core.files.service import file_to_ref  # pylint: disable=import-outside-toplevel  # Avoid circular imports

        file = await WorkflowFile.filter(file_id=file_id).first()
        if not file:
            return None
        return await fs.get_presigned_url(file_to_ref(file), config.workflow_file_presigned_url_expiry_seconds, inline=True)
    except Exception:  # pylint: disable=broad-except  # Fallback: don't break profile load if file resolution fails
        return None


async def _check_username_available(username: str) -> None:
    if await UserProfile.filter(username=username).exists():
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Username taken",
            detail=f"Username '{username}' is already in use",
            status=409,
        )


async def _create_profile(user: User, update_data: UserProfileUpdate) -> UserProfile:
    if not update_data.username:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Username required",
            detail="Username is required when creating a profile",
            status=400,
        )
    await _check_username_available(update_data.username)
    return await UserProfile.create(
        user=user,
        username=update_data.username,
        display_name=update_data.display_name,
        bio=update_data.bio,
        avatar_file_id=update_data.avatar_file_id,
        social_links=update_data.social_links or {},
        tags=update_data.tags or [],
        is_public=update_data.is_public or False,
    )


async def _apply_profile_updates(profile: UserProfile, update_data: UserProfileUpdate) -> None:
    if update_data.username is not None and update_data.username != profile.username:
        await _check_username_available(update_data.username)
        profile.username = update_data.username
    for field in ("display_name", "bio", "avatar_file_id", "social_links", "tags", "is_public"):
        value = getattr(update_data, field)
        if value is not None:
            setattr(profile, field, value)
    await profile.save()


@router.patch("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    request: Request,
    update_data: UserProfileUpdate = Body(...),
):
    """Create or update current user's profile."""
    user = _require_user(request)
    profile = await UserProfile.filter(user=user).first()
    if profile is None:
        profile = await _create_profile(user, update_data)
    else:
        await _apply_profile_updates(profile, update_data)
    response = UserProfileResponse.model_validate(profile, from_attributes=True)
    if profile.avatar_file_id and config.is_workflow_file_system_configured:
        response.avatar_url = await _resolve_avatar_url(profile.avatar_file_id)
    return response


@router.delete("/templates/{slug}", status_code=http_status.HTTP_204_NO_CONTENT)
async def unpublish_my_template(request: Request, slug: str):
    """Unpublish (delete) a community template owned by the current user."""
    user = _require_user(request)
    template = await WorkflowTemplate.filter(slug=slug, created_by=user, source=TemplateSource.COMMUNITY).first()
    if not template:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Template not found",
            detail=f"No community template with slug '{slug}' found for your account",
            status=404,
        )
    await template.delete()
