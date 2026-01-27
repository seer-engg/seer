"""User settings API endpoints."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.database import User
from seer.database.models import UserSettings, UserSettingsPublic

router = APIRouter(prefix="/users/me", tags=["user-settings"])


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""
    max_agent_steps: Optional[int] = None
    preferences: Optional[Dict[str, Any]] = None
    per_run_cost_cap_usd: Optional[float] = None


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


@router.get("/settings", response_model=UserSettingsPublic)
async def get_user_settings(request: Request):
    """Get current user's settings."""
    user = _require_user(request)
    settings, _ = await UserSettings.get_or_create(user=user)
    return UserSettingsPublic.model_validate(settings, from_attributes=True)


@router.patch("/settings", response_model=UserSettingsPublic)
async def update_user_settings(
    request: Request,
    update_data: UserSettingsUpdate = Body(...),
):
    """Update current user's settings."""
    user = _require_user(request)
    settings, _ = await UserSettings.get_or_create(user=user)

    if update_data.max_agent_steps is not None:
        if not 10 <= update_data.max_agent_steps <= 200:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid max_agent_steps",
                detail="Must be between 10 and 200",
                status=400,
            )
        settings.max_agent_steps = update_data.max_agent_steps

    if update_data.per_run_cost_cap_usd is not None:
        if not 0.10 <= update_data.per_run_cost_cap_usd <= 1000.0:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid per-run cost cap",
                detail="Cost cap must be between $0.10 and $1000.00",
                status=400,
            )
        settings.preferences["per_run_cost_cap_usd"] = update_data.per_run_cost_cap_usd

    if update_data.preferences:
        settings.preferences.update(update_data.preferences)

    await settings.save()
    return UserSettingsPublic.model_validate(settings, from_attributes=True)
