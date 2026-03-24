"""User settings API endpoints."""
from typing import Any, Dict, Optional

import pytz
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.workflows.services.catalog import DEFAULT_MODEL_REGISTRY
from seer.database import User
from seer.database.models import UserSettings, UserSettingsPublic

router = APIRouter(prefix="/users/me", tags=["user-settings"])


_VALID_MODEL_IDS: set[str] = set()


def _get_valid_model_ids() -> set[str]:
    """Lazily cache the set of valid model IDs from the registry."""
    if not _VALID_MODEL_IDS:
        _VALID_MODEL_IDS.update(m.id for m in DEFAULT_MODEL_REGISTRY)
    return _VALID_MODEL_IDS


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""
    max_agent_steps: Optional[int] = None
    timezone: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    per_run_cost_cap_usd: Optional[float] = None
    default_model: Optional[str] = None


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
    from seer.config import config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import at module load time

    user = _require_user(request)
    settings, _ = await UserSettings.get_or_create(user=user)
    result = UserSettingsPublic.model_validate(settings, from_attributes=True)
    if config.disable_usage_limits:
        prefs = result.preferences or {}
        prefs.setdefault("onboarding", {})
        prefs["onboarding"]["completed"] = True
        prefs["onboarding"]["payment_method_added"] = True
        result.preferences = prefs
    return result


def _validate_max_agent_steps(value: int) -> None:
    if not 10 <= value <= 200:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid max_agent_steps", detail="Must be between 10 and 200", status=400)


def _validate_cost_cap(value: float) -> None:
    if not 0.10 <= value <= 1000.0:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid per-run cost cap", detail="Cost cap must be between $0.10 and $1000.00", status=400)


def _validate_timezone(value: str) -> None:
    try:
        pytz.timezone(value)
    except pytz.exceptions.UnknownTimeZoneError:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid timezone", detail=f"Unknown timezone: {value}", status=400)


def _validate_default_model(value: str) -> None:
    if value not in _get_valid_model_ids():
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid default model", detail=f"Unknown model: {value}", status=400)


def _apply_field_updates(settings: UserSettings, update_data: UserSettingsUpdate) -> None:
    """Validate and apply individual field updates to settings."""
    if update_data.max_agent_steps is not None:
        _validate_max_agent_steps(update_data.max_agent_steps)
        settings.max_agent_steps = update_data.max_agent_steps

    if update_data.per_run_cost_cap_usd is not None:
        _validate_cost_cap(update_data.per_run_cost_cap_usd)
        settings.preferences["per_run_cost_cap_usd"] = update_data.per_run_cost_cap_usd

    if update_data.timezone is not None:
        _validate_timezone(update_data.timezone)
        settings.timezone = update_data.timezone

    if update_data.default_model is not None:
        _validate_default_model(update_data.default_model)
        settings.preferences["default_model"] = update_data.default_model

    if update_data.preferences:
        settings.preferences.update(update_data.preferences)


@router.patch("/settings", response_model=UserSettingsPublic)
async def update_user_settings(
    request: Request,
    update_data: UserSettingsUpdate = Body(...),
):
    """Update current user's settings."""
    user = _require_user(request)
    settings, _ = await UserSettings.get_or_create(user=user)
    _apply_field_updates(settings, update_data)
    await settings.save()
    return UserSettingsPublic.model_validate(settings, from_attributes=True)
