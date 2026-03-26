# pylint: disable=cyclic-import
# Reason: models.py lazy-imports seer.services.organization_service inside _setup_new_user()
# to avoid a runtime circular import; the cycle is intentional and safe at runtime.
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
    active_organization_id = fields.IntField(null=True)
    default_workflow_creation_mode = fields.CharField(max_length=20, default="ASK_FIRST")
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
        """
        Fetch or persist a user based on Clerk claims.

        For new users, this also creates:
        - A personal Organization (every user has one)
        - An owner OrganizationMembership
        - A FREE tier BillingSubscription

        After creation, the user's active organization is initialized to
        their personal organization.
        """
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
            # Create personal organization for new users
            await cls._setup_new_user(user)
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

    @classmethod
    async def _setup_new_user(cls, user: "User") -> None:
        """
        Set up a newly created user with personal organization.

        This is called automatically when a new user is created via get_or_create_from_auth.
        Creates:
        - Personal Organization
        - Owner membership
        - Billing profile
        - Sets active_organization_id to the personal organization

        Note: This uses lazy imports to avoid circular dependencies.
        """
        # Lazy import to avoid circular dependency
        # pylint: disable=import-outside-toplevel  # Reason: avoids circular import between database models and service layer
        from seer.services.organization_service import create_personal_organization
        from seer.logger import get_logger  # pylint: disable=import-outside-toplevel  # Reason: avoids circular import

        logger = get_logger("database.models")

        try:
            organization, _ = await create_personal_organization(user)
            user.active_organization_id = organization.id
            await user.save(update_fields=["active_organization_id"])

            logger.info(
                "Set up new user %s with personal org %s",
                user.user_id,
                organization.id,
            )

        except Exception as org_err:  # pylint: disable=broad-exception-caught  # Reason: Org creation failure should be logged but handled gracefully
            # Log the error but don't fail user creation
            # The middleware will create the org on next request
            logger.error(
                "Failed to create personal org for user %s: %s",
                user.user_id,
                org_err,
            )


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


class UserSettings(models.Model):
    """Database model for per-user settings."""

    id = fields.IntField(primary_key=True)
    user = fields.OneToOneField("models.User", related_name="settings")
    max_agent_steps = fields.IntField(null=True)
    preferences = fields.JSONField(default=dict)
    timezone = fields.CharField(max_length=64, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_settings"


class UserSettingsPublic(BaseModel):
    """Pydantic model for UserSettings API responses."""

    model_config = ConfigDict(from_attributes=True)

    max_agent_steps: Optional[int] = None
    preferences: Dict[str, Any] = {}
    timezone: Optional[str] = None
    updated_at: datetime
