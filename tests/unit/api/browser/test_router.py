"""Tests for browser profile management router."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.api.browser.router import router


def _make_db_user(user_id="user-123"):
    """Create a mock db_user."""
    user = MagicMock()
    user.user_id = user_id
    user.id = user_id
    return user


def _make_profile(
    profile_id=None,
    name="Test Profile",
    logged_in_domains=None,
    created_at=None,
    last_used_at=None,
):
    """Create a mock BrowserProfile."""
    profile = MagicMock()
    profile.id = profile_id or uuid4()
    profile.name = name
    profile.logged_in_domains = logged_in_domains or []
    profile.created_at = created_at or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    profile.last_used_at = last_used_at
    profile.status = "active"
    return profile


def _make_test_app(user):
    """Create a FastAPI app with the router and auth middleware."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")

    @test_app.middleware("http")
    async def add_user(request, call_next):
        request.state.db_user = user
        return await call_next(request)

    return test_app


@pytest.fixture
def mock_db_user():
    return _make_db_user()


@pytest.mark.asyncio
@pytest.mark.unit
class TestCreateProfile:
    """Test POST /api/browser/profiles endpoint."""

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_create_profile_success(self, mock_manager_cls, mock_db_user):
        mock_profile = _make_profile(name="Work Profile")

        mock_manager = MagicMock()
        mock_manager.create_profile = AsyncMock(return_value=mock_profile)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": "Work Profile"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Work Profile"
        assert "id" in data

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_create_profile_returns_correct_fields(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(profile_id=profile_id, name="My Profile")

        mock_manager = MagicMock()
        mock_manager.create_profile = AsyncMock(return_value=mock_profile)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": "My Profile"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(profile_id)
        assert data["name"] == "My Profile"
        assert data["logged_in_domains"] == []
        assert data["created_at"] is not None
        assert data["last_used_at"] is None

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_create_profile_duplicate_name_409(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.create_profile = AsyncMock(
            side_effect=Exception("UNIQUE constraint failed")
        )
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": "Duplicate"})

        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_create_profile_duplicate_key_409(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.create_profile = AsyncMock(
            side_effect=Exception("duplicate key value violates unique constraint")
        )
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": "Duplicate"})

        assert resp.status_code == 409

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_create_profile_general_error_500(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.create_profile = AsyncMock(
            side_effect=RuntimeError("Database connection failed")
        )
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": "Test"})

        assert resp.status_code == 500
        assert "Failed to create profile" in resp.json()["detail"]

    async def test_create_profile_empty_name_422(self, mock_db_user):
        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": ""})

        assert resp.status_code == 422

    async def test_create_profile_name_too_long_422(self, mock_db_user):
        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        long_name = "x" * 101  # Max is 100

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/browser/profiles", json={"name": long_name})

        assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.unit
class TestListProfiles:
    """Test GET /api/browser/profiles endpoint."""

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_list_profiles_success(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.list_profiles = AsyncMock(return_value=[
            {
                "id": str(uuid4()),
                "name": "Profile 1",
                "logged_in_domains": ["example.com"],
                "created_at": "2024-01-01T12:00:00+00:00",
                "last_used_at": None,
            },
        ])
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/browser/profiles")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Profile 1"
        assert data[0]["logged_in_domains"] == ["example.com"]

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_list_profiles_empty(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.list_profiles = AsyncMock(return_value=[])
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/browser/profiles")

        assert resp.status_code == 200
        assert resp.json() == []

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_list_profiles_multiple(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.list_profiles = AsyncMock(return_value=[
            {"id": str(uuid4()), "name": "Profile 1", "logged_in_domains": [], "created_at": None, "last_used_at": None},
            {"id": str(uuid4()), "name": "Profile 2", "logged_in_domains": [], "created_at": None, "last_used_at": None},
            {"id": str(uuid4()), "name": "Profile 3", "logged_in_domains": [], "created_at": None, "last_used_at": None},
        ])
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/browser/profiles")

        assert resp.status_code == 200
        assert len(resp.json()) == 3


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetProfile:
    """Test GET /api/browser/profiles/{profile_id} endpoint."""

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_get_profile_success(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(
            profile_id=profile_id,
            name="My Profile",
            logged_in_domains=["slack.com", "github.com"],
        )

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(profile_id)
        assert data["name"] == "My Profile"
        assert "slack.com" in data["logged_in_domains"]

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_get_profile_not_found_404(self, mock_manager_cls, mock_db_user):
        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=None)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        profile_id = uuid4()
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_get_profile_formats_dates(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(
            profile_id=profile_id,
            created_at=datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
            last_used_at=datetime(2024, 6, 16, 14, 0, 0, tzinfo=timezone.utc),
        )

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["created_at"] == "2024-06-15T10:30:00+00:00"
        assert data["last_used_at"] == "2024-06-16T14:00:00+00:00"

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_get_profile_handles_null_dates(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(profile_id=profile_id)
        mock_profile.created_at = None
        mock_profile.last_used_at = None

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["created_at"] is None
        assert data["last_used_at"] is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestDeleteProfile:
    """Test DELETE /api/browser/profiles/{profile_id} endpoint."""

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_delete_profile_success(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()

        mock_manager = MagicMock()
        mock_manager.delete_profile = AsyncMock(return_value=True)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["profile_id"] == str(profile_id)

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_delete_profile_not_found_404(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()

        mock_manager = MagicMock()
        mock_manager.delete_profile = AsyncMock(return_value=False)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(f"/api/browser/profiles/{profile_id}")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.unit
class TestStartLogin:
    """Test POST /api/browser/profiles/{profile_id}/login endpoint (deprecated)."""

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_start_login_success(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(profile_id=profile_id)

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager.start_interactive_login = AsyncMock(return_value={
            "profile_id": str(profile_id),
            "logged_in_domains": ["slack.com", "github.com"],
            "status": "session_saved",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(f"/api/browser/profiles/{profile_id}/login", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["profile_id"] == str(profile_id)
        assert "slack.com" in data["logged_in_domains"]
        assert data["status"] == "session_saved"

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_start_login_profile_not_found_404(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=None)
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(f"/api/browser/profiles/{profile_id}/login", json={})

        assert resp.status_code == 404

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_start_login_passes_target_url(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(profile_id=profile_id)

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager.start_interactive_login = AsyncMock(return_value={
            "profile_id": str(profile_id),
            "logged_in_domains": [],
            "status": "session_saved",
        })
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/api/browser/profiles/{profile_id}/login",
                json={"target_url": "https://slack.com/signin"}
            )

        assert resp.status_code == 200
        # Verify target_url was passed
        mock_manager.start_interactive_login.assert_called_once()
        call_args = mock_manager.start_interactive_login.call_args
        assert call_args[0][2] == "https://slack.com/signin"

    @patch("seer.api.browser.router.BrowserProfileManager")
    async def test_start_login_error_500(self, mock_manager_cls, mock_db_user):
        profile_id = uuid4()
        mock_profile = _make_profile(profile_id=profile_id)

        mock_manager = MagicMock()
        mock_manager.get_profile = AsyncMock(return_value=mock_profile)
        mock_manager.start_interactive_login = AsyncMock(
            side_effect=RuntimeError("Browser launch failed")
        )
        mock_manager_cls.return_value = mock_manager

        test_app = _make_test_app(mock_db_user)
        transport = ASGITransport(app=test_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(f"/api/browser/profiles/{profile_id}/login", json={})

        assert resp.status_code == 500
        assert "Failed to start login session" in resp.json()["detail"]
