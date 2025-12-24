from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from tortoise import fields, models
from api.middleware.auth import AuthenticatedUser


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
    "Project",
    "ProjectBase",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectPublic",
    "ProjectListResponse",
]
