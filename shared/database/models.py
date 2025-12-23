from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from tortoise import fields, models

if TYPE_CHECKING:
    from api.middleware.auth import AuthenticatedUser


class User(models.Model):
    """Database model for authenticated users."""

    id = fields.IntField(pk=True)
    user_id = fields.CharField(max_length=255, unique=True, index=True) # Clerk user ID
    email = fields.CharField(max_length=320, null=True)
    first_name = fields.CharField(max_length=255, null=True)
    last_name = fields.CharField(max_length=255, null=True)
    claims = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "users"
        ordering = ("user_id",)

    def __str__(self) -> str:
        return f"User<{self.user_id}>"

    @classmethod
    async def get_or_create_from_auth(cls, auth_user: "AuthenticatedUser") -> "User":
        """Fetch or persist a user based on Clerk claims."""
        defaults: Dict[str, Any] = {
            "email": auth_user.email,
            "first_name": auth_user.first_name,
            "last_name": auth_user.last_name,
            "claims": auth_user.claims,
        }

        user, created = await cls.get_or_create(
            user_id=auth_user.user_id,
            defaults=defaults,
        )
        if created:
            return user

        updated_fields = []
        for field, value in defaults.items():
            if getattr(user, field) != value:
                setattr(user, field, value)
                updated_fields.append(field)

        if updated_fields:
            await user.save(update_fields=updated_fields)

        return user


class UserPublic(BaseModel):
    """Pydantic model for User API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class Project(models.Model):
    """Database model for projects."""

    id = fields.IntField(primary_key=True)
    project_name = fields.CharField(max_length=255, unique=True)
    description = fields.TextField(null=True)
    metadata = fields.JSONField(null=True)
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "projects"
        ordering = ("project_name",)

    def __str__(self) -> str:
        return f"Project<{self.project_name}>"


class ProjectBase(BaseModel):
    """Shared attributes for create/update."""

    project_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True


class ProjectCreate(ProjectBase):
    """Payload for creating a project."""


class ProjectUpdate(BaseModel):
    """Payload for updating a project."""

    project_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ProjectPublic(ProjectBase):
    """Response model returned to API clients."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    """Wrapper response for list endpoints."""

    projects: list[ProjectPublic]
    user: AuthenticatedUser


__all__ = [
    "User",
    "UserPublic",
    "Project",
    "ProjectBase",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectPublic",
    "ProjectListResponse",
]
