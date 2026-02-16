"""Tests for BrowserProfileManager - profile CRUD and interactive sessions."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from seer.services.browser.profile_manager import BrowserProfileManager


@pytest.fixture
def manager():
    """Create a fresh BrowserProfileManager instance."""
    return BrowserProfileManager()


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock()
    user.user_id = "user-123"
    user.id = "user-123"
    return user


@pytest.fixture
def profile_id():
    """Generate a fresh UUID for each test."""
    return uuid4()


@pytest.fixture
def sample_storage_state():
    """Reusable Playwright storage_state dict."""
    return {
        "cookies": [
            {
                "name": "session",
                "value": "abc123",
                "domain": ".example.com",
                "path": "/",
                "expires": 1700000000,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
            },
            {
                "name": "auth",
                "value": "xyz789",
                "domain": ".github.com",
                "path": "/",
                "expires": 1700000000,
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            },
        ],
        "origins": [],
    }


@pytest.fixture
def mock_profile(profile_id, sample_storage_state):
    """Create a mock BrowserProfile."""
    profile = MagicMock()
    profile.id = profile_id
    profile.name = "Test Profile"
    profile.session_state_enc = None
    profile.logged_in_domains = ["example.com"]
    profile.status = "active"
    profile.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    profile.last_used_at = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    profile.save = AsyncMock()
    return profile


@pytest.mark.asyncio
@pytest.mark.unit
class TestCreateProfile:
    """Test create_profile method."""

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_profile_success(self, mock_bp, manager, mock_user):
        mock_profile = MagicMock()
        mock_profile.id = uuid4()
        mock_profile.name = "Work Profile"
        mock_bp.create = AsyncMock(return_value=mock_profile)

        result = await manager.create_profile(mock_user, "Work Profile")

        assert result == mock_profile
        mock_bp.create.assert_called_once_with(user=mock_user, name="Work Profile")

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_profile_returns_created_object(self, mock_bp, manager, mock_user):
        expected_id = uuid4()
        mock_profile = MagicMock()
        mock_profile.id = expected_id
        mock_profile.name = "Personal"
        mock_bp.create = AsyncMock(return_value=mock_profile)

        result = await manager.create_profile(mock_user, "Personal")

        assert result.id == expected_id
        assert result.name == "Personal"


@pytest.mark.asyncio
@pytest.mark.unit
class TestListProfiles:
    """Test list_profiles method."""

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_returns_formatted_dicts(self, mock_bp, manager, mock_user, mock_profile):
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[mock_profile])
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.list_profiles(mock_user)

        assert len(result) == 1
        assert result[0]["id"] == str(mock_profile.id)
        assert result[0]["name"] == "Test Profile"
        assert result[0]["logged_in_domains"] == ["example.com"]
        assert "created_at" in result[0]
        assert "last_used_at" in result[0]

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_filters_by_user_and_status(self, mock_bp, manager, mock_user):
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[])
        mock_bp.filter = MagicMock(return_value=mock_query)

        await manager.list_profiles(mock_user)

        mock_bp.filter.assert_called_once_with(user=mock_user, status="active")

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_empty_list(self, mock_bp, manager, mock_user):
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[])
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.list_profiles(mock_user)

        assert result == []

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_handles_null_domains(self, mock_bp, manager, mock_user, mock_profile):
        mock_profile.logged_in_domains = None
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[mock_profile])
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.list_profiles(mock_user)

        assert result[0]["logged_in_domains"] == []

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_formats_dates_as_iso(self, mock_bp, manager, mock_user, mock_profile):
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[mock_profile])
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.list_profiles(mock_user)

        assert result[0]["created_at"] == "2024-01-01T12:00:00+00:00"
        assert result[0]["last_used_at"] == "2024-01-02T12:00:00+00:00"

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_list_profiles_handles_null_dates(self, mock_bp, manager, mock_user, mock_profile):
        mock_profile.created_at = None
        mock_profile.last_used_at = None
        mock_query = MagicMock()
        mock_query.all = AsyncMock(return_value=[mock_profile])
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.list_profiles(mock_user)

        assert result[0]["created_at"] is None
        assert result[0]["last_used_at"] is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetProfile:
    """Test get_profile method."""

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_get_profile_found(self, mock_bp, manager, mock_user, profile_id, mock_profile):
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        result = await manager.get_profile(mock_user, profile_id)

        assert result == mock_profile
        mock_bp.get_or_none.assert_called_once_with(id=profile_id, user=mock_user, status="active")

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_get_profile_not_found(self, mock_bp, manager, mock_user, profile_id):
        mock_bp.get_or_none = AsyncMock(return_value=None)

        result = await manager.get_profile(mock_user, profile_id)

        assert result is None

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_get_profile_filters_by_active_status(self, mock_bp, manager, mock_user, profile_id):
        mock_bp.get_or_none = AsyncMock(return_value=None)

        await manager.get_profile(mock_user, profile_id)

        # Verify that status="active" is passed
        call_kwargs = mock_bp.get_or_none.call_args.kwargs
        assert call_kwargs["status"] == "active"


@pytest.mark.asyncio
@pytest.mark.unit
class TestDeleteProfile:
    """Test delete_profile method."""

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_delete_profile_success(self, mock_bp, manager, mock_user, profile_id):
        mock_query = MagicMock()
        mock_query.update = AsyncMock(return_value=1)
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.delete_profile(mock_user, profile_id)

        assert result is True
        mock_bp.filter.assert_called_once_with(id=profile_id, user=mock_user)
        mock_query.update.assert_called_once_with(status="deleted")

    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_delete_profile_not_found(self, mock_bp, manager, mock_user, profile_id):
        mock_query = MagicMock()
        mock_query.update = AsyncMock(return_value=0)
        mock_bp.filter = MagicMock(return_value=mock_query)

        result = await manager.delete_profile(mock_user, profile_id)

        assert result is False


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetSessionState:
    """Test get_session_state method."""

    async def test_get_session_state_delegates_to_context_manager(self, manager, mock_user, profile_id, sample_storage_state):
        manager._session_context.load_session_state = AsyncMock(return_value=sample_storage_state)

        result = await manager.get_session_state(mock_user, profile_id)

        assert result == sample_storage_state
        manager._session_context.load_session_state.assert_called_once_with(mock_user, profile_id)

    async def test_get_session_state_returns_none_for_missing(self, manager, mock_user, profile_id):
        manager._session_context.load_session_state = AsyncMock(return_value=None)

        result = await manager.get_session_state(mock_user, profile_id)

        assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestUpdateSessionState:
    """Test update_session_state method."""

    async def test_update_session_state_delegates_to_context_manager(self, manager, mock_user, profile_id, sample_storage_state):
        manager._session_context.save_session_state = AsyncMock()

        await manager.update_session_state(mock_user, profile_id, sample_storage_state)

        manager._session_context.save_session_state.assert_called_once_with(mock_user, profile_id, sample_storage_state)


@pytest.mark.asyncio
@pytest.mark.unit
class TestCreateInteractiveSession:
    """Test create_interactive_session method."""

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_gets_profile(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = False
        mock_config.browser_interactive_timeout_seconds = 300
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.cdp_client.send = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        await manager.create_interactive_session(mock_user, profile_id)

        mock_bp.get.assert_called_once_with(id=profile_id, user=mock_user, status="active")

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_loads_existing_state(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile, sample_storage_state):
        mock_config.browser_recording_enabled = False
        mock_config.browser_interactive_timeout_seconds = 300
        encrypted_state = "encrypted_data"
        mock_profile.session_state_enc = encrypted_state
        mock_bp.get = AsyncMock(return_value=mock_profile)

        manager._encryptor.decrypt = MagicMock(return_value=sample_storage_state)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        await manager.create_interactive_session(mock_user, profile_id)

        manager._encryptor.decrypt.assert_called_once_with(encrypted_state)
        # Verify pool was called with decrypted storage_state
        call_kwargs = mock_pool.create_session.call_args.kwargs
        assert call_kwargs["storage_state"] == sample_storage_state

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_creates_pool_session_for_interactive(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = False
        mock_config.browser_interactive_timeout_seconds = 300
        mock_profile.session_state_enc = None
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        await manager.create_interactive_session(mock_user, profile_id)

        # Verify session_type is set correctly
        call_kwargs = mock_pool.create_session.call_args.kwargs
        assert call_kwargs["session_type"] == "interactive"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_starts_recording_when_enabled(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = True
        mock_config.browser_interactive_timeout_seconds = 300
        mock_profile.session_state_enc = None
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(return_value="rec-456")
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        result = await manager.create_interactive_session(mock_user, profile_id, target_url="https://example.com")

        mock_recorder.start_recording.assert_called_once()
        assert result["recording_id"] == "rec-456"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_returns_session_info(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = False
        mock_config.browser_interactive_timeout_seconds = 300
        mock_profile.session_state_enc = None
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-abc"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        result = await manager.create_interactive_session(mock_user, profile_id)

        assert result["session_id"] == "sess-abc"
        assert result["profile_id"] == str(profile_id)
        assert result["status"] == "created"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_handles_navigation_failure(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = False
        mock_config.browser_interactive_timeout_seconds = 300
        mock_profile.session_state_enc = None
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock(side_effect=RuntimeError("Page error"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        # Should not raise, just log warning
        result = await manager.create_interactive_session(mock_user, profile_id, target_url="https://example.com")

        assert result["status"] == "created"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    @patch("seer.services.browser.profile_manager.BrowserProfile")
    async def test_create_session_handles_recording_failure(self, mock_bp, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, mock_profile):
        mock_config.browser_recording_enabled = True
        mock_config.browser_interactive_timeout_seconds = 300
        mock_profile.session_state_enc = None
        mock_bp.get = AsyncMock(return_value=mock_profile)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.session.must_get_current_page = AsyncMock()
        mock_managed.session.cdp_client = MagicMock()
        mock_managed.session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="cdp-1"))

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(side_effect=RuntimeError("Recording failed"))
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        # Should not raise, just log warning
        result = await manager.create_interactive_session(mock_user, profile_id)

        assert result["recording_id"] is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestCompleteInteractiveSession:
    """Test complete_interactive_session method."""

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_gets_pool_session(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        mock_pool.get_session.assert_called_once_with("sess-123")

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_validates_user_ownership(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = "different-user"

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        with pytest.raises(PermissionError, match="Session does not belong to this user"):
            await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_session_not_found(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id):
        mock_config.browser_recording_enabled = False

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        with pytest.raises(ValueError, match="Session sess-123 not found in pool"):
            await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_saves_recording(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = "rec-123"
        mock_managed.start_url = "https://example.com"

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.save_recording = AsyncMock(return_value="rec-saved-123")
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        manager._session_context.save_session_state = AsyncMock()

        result = await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        mock_recorder.save_recording.assert_called_once()
        assert result["recording_id"] == "rec-saved-123"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_releases_pool_session(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        mock_pool.release_session.assert_called_once_with("sess-123")

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_saves_encrypted_state(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        manager._session_context.save_session_state.assert_called_once_with(mock_user, profile_id, sample_storage_state)

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_extracts_domains(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        result = await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        # sample_storage_state has example.com and github.com cookies
        assert "example.com" in result["logged_in_domains"]
        assert "github.com" in result["logged_in_domains"]

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_returns_result_dict(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        result = await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        assert result["profile_id"] == str(profile_id)
        assert "logged_in_domains" in result
        assert result["status"] == "session_saved"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_handles_recording_failure(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id, sample_storage_state):
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = "rec-123"
        mock_managed.start_url = "https://example.com"

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=sample_storage_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.save_recording = AsyncMock(side_effect=RuntimeError("Save failed"))
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        manager._session_context.save_session_state = AsyncMock()

        # Should not raise, just log warning and continue
        result = await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        assert result["recording_id"] is None
        assert result["status"] == "session_saved"

    @patch("seer.services.browser.profile_manager.config")
    @patch("seer.services.browser.profile_manager.RecordingService")
    @patch("seer.services.browser.profile_manager.BrowserPoolManager")
    async def test_complete_session_no_storage_state(self, mock_pool_cls, mock_rec_cls, mock_config, manager, mock_user, profile_id):
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.user_id = str(mock_user.user_id)
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        manager._session_context.save_session_state = AsyncMock()

        result = await manager.complete_interactive_session(mock_user, profile_id, "sess-123")

        # Should not call save_session_state if no storage_state returned
        manager._session_context.save_session_state.assert_not_called()
        assert result["logged_in_domains"] == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestExtractDomains:
    """Test _extract_domains method."""

    def test_extract_domains_delegates_to_context_manager(self, manager, sample_storage_state):
        with patch("seer.services.browser.profile_manager.SessionContextManager.extract_domains") as mock_extract:
            mock_extract.return_value = ["example.com", "github.com"]

            result = manager._extract_domains(sample_storage_state)

            mock_extract.assert_called_once_with(sample_storage_state)
            assert result == ["example.com", "github.com"]
