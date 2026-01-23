"""
Integration tests for user settings API endpoints.
"""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestUserSettingsAPI:
    """Integration tests for user settings endpoints."""

    @pytest.fixture
    async def settings_client(self, mock_app: FastAPI, db_engine, test_user) -> AsyncClient:  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Create authenticated API client with settings router."""
        from seer.api.users.settings import router  # pylint: disable=import-outside-toplevel  # Dynamic import for test fixture

        # Add router to test app
        mock_app.include_router(router)

        # Mock authentication to inject test_user
        from fastapi import Request  # pylint: disable=import-outside-toplevel  # Dynamic import for test fixture
        async def mock_auth_middleware(request: Request, call_next):
            request.state.db_user = test_user
            response = await call_next(request)
            return response

        mock_app.middleware("http")(mock_auth_middleware)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_get_settings_creates_default(self, settings_client):
        """Test getting settings creates default if not exists."""
        response = await settings_client.get("/users/me/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["max_agent_steps"] is None
        assert data["preferences"] == {}
        assert "updated_at" in data

    async def test_update_max_agent_steps(self, settings_client):
        """Test updating max_agent_steps."""
        response = await settings_client.patch(
            "/users/me/settings",
            json={"max_agent_steps": 100}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["max_agent_steps"] == 100

        # Verify persistence
        response = await settings_client.get("/users/me/settings")
        assert response.status_code == 200
        assert response.json()["max_agent_steps"] == 100

    async def test_update_max_agent_steps_validation(self, settings_client):
        """Test max_agent_steps validation bounds."""
        # Too low
        response = await settings_client.patch(
            "/users/me/settings",
            json={"max_agent_steps": 5}
        )
        assert response.status_code == 400

        # Too high
        response = await settings_client.patch(
            "/users/me/settings",
            json={"max_agent_steps": 250}
        )
        assert response.status_code == 400

        # Valid boundaries
        response = await settings_client.patch(
            "/users/me/settings",
            json={"max_agent_steps": 10}
        )
        assert response.status_code == 200

        response = await settings_client.patch(
            "/users/me/settings",
            json={"max_agent_steps": 200}
        )
        assert response.status_code == 200

    async def test_update_preferences(self, settings_client):
        """Test updating user preferences."""
        response = await settings_client.patch(
            "/users/me/settings",
            json={"preferences": {"theme": "dark", "notifications": True}}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["preferences"]["theme"] == "dark"
        assert data["preferences"]["notifications"] is True

    async def test_update_preferences_merges(self, settings_client):
        """Test preferences are merged, not replaced."""
        # Set initial preferences
        await settings_client.patch(
            "/users/me/settings",
            json={"preferences": {"theme": "dark", "language": "en"}}
        )

        # Update with new key
        response = await settings_client.patch(
            "/users/me/settings",
            json={"preferences": {"notifications": True}}
        )

        data = response.json()
        assert data["preferences"]["theme"] == "dark"
        assert data["preferences"]["language"] == "en"
        assert data["preferences"]["notifications"] is True

    async def test_update_both_settings_and_preferences(self, settings_client):
        """Test updating max_agent_steps and preferences together."""
        response = await settings_client.patch(
            "/users/me/settings",
            json={
                "max_agent_steps": 150,
                "preferences": {"theme": "light"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["max_agent_steps"] == 150
        assert data["preferences"]["theme"] == "light"
