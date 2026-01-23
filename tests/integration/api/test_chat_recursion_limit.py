"""
Integration tests for chat recursion limit configuration.
"""
import pytest
from tortoise.exceptions import DoesNotExist


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatRecursionLimit:
    """Integration tests for recursion_limit configuration logic."""

    async def test_get_recursion_limit_default(self, db_engine, test_user):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Test recursion limit falls back to config default when no user settings exist."""
        from seer.config import config  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation
        from seer.database.models import UserSettings  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation

        # Verify user has no settings
        with pytest.raises(DoesNotExist):
            await UserSettings.get(user=test_user)

        # Simulate the logic from router
        try:
            user_settings = await UserSettings.get(user=test_user)
            max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        except DoesNotExist:
            max_agent_steps = config.nexus_max_agent_steps

        assert max_agent_steps == config.nexus_max_agent_steps
        assert max_agent_steps == 75  # Default value

    async def test_get_recursion_limit_user_override(self, db_engine, test_user):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Test recursion limit uses user settings when they exist."""
        from seer.config import config  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation
        from seer.database.models import UserSettings  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation

        # Create user settings with custom value
        await UserSettings.create(user=test_user, max_agent_steps=120)

        # Simulate the logic from router
        try:
            user_settings = await UserSettings.get(user=test_user)
            max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        except DoesNotExist:
            max_agent_steps = config.nexus_max_agent_steps

        assert max_agent_steps == 120

    async def test_get_recursion_limit_user_null_falls_back(self, db_engine, test_user):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Test recursion limit falls back to config when user setting is null."""
        from seer.config import config  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation
        from seer.database.models import UserSettings  # pylint: disable=import-outside-toplevel  # Dynamic import for test isolation

        # Create user settings with null max_agent_steps (uses preferences only)
        await UserSettings.create(user=test_user, max_agent_steps=None, preferences={"theme": "dark"})

        # Simulate the logic from router
        try:
            user_settings = await UserSettings.get(user=test_user)
            max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        except DoesNotExist:
            max_agent_steps = config.nexus_max_agent_steps

        assert max_agent_steps == config.nexus_max_agent_steps
        assert max_agent_steps == 75
