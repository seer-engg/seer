"""Tests for SessionContextManager - encrypted session persistence."""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from seer.services.browser.encryption import SessionEncryptor
from seer.services.browser.session_context_manager import SessionContextManager


@pytest.fixture
def fernet_key():
    return Fernet.generate_key()


@pytest.fixture
def encryptor(fernet_key):
    return SessionEncryptor(key=fernet_key)


@pytest.fixture
def manager(encryptor):
    return SessionContextManager(encryptor=encryptor)


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.user_id = "test-user-123"
    return user


@pytest.fixture
def profile_id():
    return uuid4()


@pytest.fixture
def sample_storage_state():
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


class TestLoadSessionState:
    """Test loading and decrypting session state."""

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_load_encrypted_state(self, mock_bp, manager, encryptor, mock_user, profile_id, sample_storage_state):
        encrypted = encryptor.encrypt(sample_storage_state)
        mock_profile = MagicMock()
        mock_profile.session_state_enc = encrypted
        mock_profile.save = AsyncMock()
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        result = await manager.load_session_state(mock_user, profile_id)
        assert result == sample_storage_state
        mock_profile.save.assert_called_once()

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_load_plain_json_fallback(self, mock_bp, manager, mock_user, profile_id, sample_storage_state):
        mock_profile = MagicMock()
        mock_profile.session_state_enc = json.dumps(sample_storage_state)
        mock_profile.save = AsyncMock()
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        result = await manager.load_session_state(mock_user, profile_id)
        assert result == sample_storage_state

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_load_no_profile(self, mock_bp, manager, mock_user, profile_id):
        mock_bp.get_or_none = AsyncMock(return_value=None)
        result = await manager.load_session_state(mock_user, profile_id)
        assert result is None

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_load_no_session_data(self, mock_bp, manager, mock_user, profile_id):
        mock_profile = MagicMock()
        mock_profile.session_state_enc = None
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        result = await manager.load_session_state(mock_user, profile_id)
        assert result is None


class TestSaveSessionState:
    """Test encrypting and saving session state."""

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_save_encrypts(self, mock_bp, manager, encryptor, mock_user, profile_id, sample_storage_state):
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        result = await manager.save_session_state(mock_user, profile_id, sample_storage_state)
        assert result is True
        mock_profile.save.assert_called_once()

        # Verify it was encrypted (not plain JSON)
        saved_enc = mock_profile.session_state_enc
        assert saved_enc != json.dumps(sample_storage_state)
        # Verify we can decrypt it back
        decrypted = encryptor.decrypt(saved_enc)
        assert decrypted == sample_storage_state

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_save_extracts_domains(self, mock_bp, manager, mock_user, profile_id, sample_storage_state):
        mock_profile = MagicMock()
        mock_profile.save = AsyncMock()
        mock_bp.get_or_none = AsyncMock(return_value=mock_profile)

        await manager.save_session_state(mock_user, profile_id, sample_storage_state)
        assert mock_profile.logged_in_domains == ["example.com", "github.com"]

    @patch("seer.services.browser.session_context_manager.BrowserProfile")
    async def test_save_no_profile(self, mock_bp, manager, mock_user, profile_id, sample_storage_state):
        mock_bp.get_or_none = AsyncMock(return_value=None)
        result = await manager.save_session_state(mock_user, profile_id, sample_storage_state)
        assert result is False


class TestValidateSession:
    """Test session validation."""

    def test_valid_session(self, sample_storage_state):
        assert SessionContextManager.validate_session(sample_storage_state) is True

    def test_empty_cookies(self):
        assert SessionContextManager.validate_session({"cookies": [], "origins": []}) is False

    def test_none_state(self):
        assert SessionContextManager.validate_session(None) is False

    def test_no_cookies_key(self):
        assert SessionContextManager.validate_session({"origins": []}) is False


class TestExtractDomains:
    """Test domain extraction from cookies."""

    def test_extracts_domains(self):
        data = {
            "cookies": [
                {"domain": ".example.com"},
                {"domain": ".github.com"},
                {"domain": "example.com"},
            ]
        }
        domains = SessionContextManager.extract_domains(data)
        assert domains == ["example.com", "github.com"]

    def test_empty_cookies(self):
        domains = SessionContextManager.extract_domains({"cookies": []})
        assert domains == []

    def test_no_domain_field(self):
        data = {"cookies": [{"name": "test", "value": "123"}]}
        domains = SessionContextManager.extract_domains(data)
        assert domains == []
