from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict
from tortoise import fields, models

if TYPE_CHECKING:
    from seer.api.core.middleware.auth import AuthenticatedUser


class User(models.Model):
    """Database model for authenticated users."""

    id = fields.IntField(primary_key=True)
    user_id = fields.CharField(max_length=255, unique=True, db_index=True)  # Clerk user ID
    email = fields.CharField(max_length=320, null=True)
    first_name = fields.CharField(max_length=255, null=True)
    last_name = fields.CharField(max_length=255, null=True)
    claims = fields.JSONField(null=True)
    signup_source = fields.CharField(max_length=50, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "users"
        ordering = ("user_id",)

    def __str__(self) -> str:
        return f"User<{self.user_id}>"

    @classmethod
    async def get_or_create_from_auth(
        cls, auth_user: "AuthenticatedUser", signup_source: Optional[str] = None
    ) -> "User":
        """Fetch or persist a user based on Clerk claims."""
        defaults: Dict[str, Any] = {
            "email": auth_user.email,
            "first_name": auth_user.first_name,
            "last_name": auth_user.last_name,
            "claims": auth_user.claims,
        }

        # Only set signup_source on creation, not on update
        if signup_source:
            defaults["signup_source"] = signup_source

        user, created = await cls.get_or_create(
            user_id=auth_user.user_id,
            defaults=defaults,
        )
        if created:
            return user

        updated_fields = []
        for field, value in defaults.items():
            # Don't update signup_source if user already exists
            if field == "signup_source":
                continue
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
    signup_source: Optional[str] = None
    created_at: datetime
    updated_at: datetime
