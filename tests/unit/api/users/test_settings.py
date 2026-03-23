"""Unit tests for user settings timezone handling."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.api.users.settings import update_user_settings, UserSettingsUpdate


def _make_request(user=None):
    req = MagicMock()
    req.state.db_user = user or MagicMock()
    return req


def _make_settings(**overrides):
    defaults = {
        "max_agent_steps": 50,
        "preferences": {},
        "timezone": None,
        "updated_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    s.save = AsyncMock()
    return s


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_timezone_valid():
    """Valid IANA timezone saves successfully."""
    settings = _make_settings()

    with patch("seer.api.users.settings.UserSettings") as MockSettings:
        MockSettings.get_or_create = AsyncMock(return_value=(settings, True))

        await update_user_settings(
            request=_make_request(),
            update_data=UserSettingsUpdate(timezone="Asia/Kolkata"),
        )

    assert settings.timezone == "Asia/Kolkata"
    settings.save.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_timezone_invalid():
    """Invalid timezone raises 400."""
    settings = _make_settings()

    with patch("seer.api.users.settings.UserSettings") as MockSettings:
        MockSettings.get_or_create = AsyncMock(return_value=(settings, True))

        with pytest.raises(HTTPException) as exc_info:
            await update_user_settings(
                request=_make_request(),
                update_data=UserSettingsUpdate(timezone="Not/Real"),
            )

    assert exc_info.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_timezone_null_allowed():
    """timezone=None skips the timezone block entirely."""
    settings = _make_settings(timezone="America/Chicago")

    with patch("seer.api.users.settings.UserSettings") as MockSettings:
        MockSettings.get_or_create = AsyncMock(return_value=(settings, True))

        await update_user_settings(
            request=_make_request(),
            update_data=UserSettingsUpdate(timezone=None),
        )

    # timezone should remain unchanged
    assert settings.timezone == "America/Chicago"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_default_model_valid():
    """Valid model ID saves to preferences."""
    settings = _make_settings(preferences={})

    with (
        patch("seer.api.users.settings.UserSettings") as MockSettings,
        patch("seer.api.users.settings.DEFAULT_MODEL_REGISTRY", [
            MagicMock(id="qwen/qwen3-235b-a22b-2507"),
            MagicMock(id="openai/gpt-4o"),
        ]),
    ):
        # Clear cache so it re-reads the patched registry
        from seer.api.users.settings import _VALID_MODEL_IDS
        _VALID_MODEL_IDS.clear()

        MockSettings.get_or_create = AsyncMock(return_value=(settings, True))

        await update_user_settings(
            request=_make_request(),
            update_data=UserSettingsUpdate(default_model="openai/gpt-4o"),
        )

    assert settings.preferences["default_model"] == "openai/gpt-4o"
    settings.save.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_default_model_invalid():
    """Unknown model ID raises 400."""
    settings = _make_settings(preferences={})

    with (
        patch("seer.api.users.settings.UserSettings") as MockSettings,
        patch("seer.api.users.settings.DEFAULT_MODEL_REGISTRY", [
            MagicMock(id="qwen/qwen3-235b-a22b-2507"),
        ]),
    ):
        from seer.api.users.settings import _VALID_MODEL_IDS
        _VALID_MODEL_IDS.clear()

        MockSettings.get_or_create = AsyncMock(return_value=(settings, True))

        with pytest.raises(HTTPException) as exc_info:
            await update_user_settings(
                request=_make_request(),
                update_data=UserSettingsUpdate(default_model="fake/model"),
            )

    assert exc_info.value.status_code == 400
