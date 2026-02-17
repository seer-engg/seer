"""Tests for SessionEncryptor - Fernet encrypt/decrypt with backward compat."""
import json

import pytest
from cryptography.fernet import Fernet

from seer.services.browser.encryption import SessionEncryptor


@pytest.fixture
def fernet_key():
    """Generate a valid Fernet key for testing."""
    return Fernet.generate_key()


@pytest.fixture
def encryptor(fernet_key):
    """Create a SessionEncryptor with a test key."""
    return SessionEncryptor(key=fernet_key)


@pytest.fixture
def sample_storage_state():
    """Sample Playwright storage_state dict."""
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
            }
        ],
        "origins": [],
    }


class TestEncryptDecrypt:
    """Test encrypt/decrypt roundtrip."""

    def test_roundtrip(self, encryptor, sample_storage_state):
        encrypted = encryptor.encrypt(sample_storage_state)
        assert isinstance(encrypted, str)
        assert encrypted != json.dumps(sample_storage_state)

        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == sample_storage_state

    def test_encrypted_output_is_not_plaintext(self, encryptor, sample_storage_state):
        encrypted = encryptor.encrypt(sample_storage_state)
        # Should not be valid JSON (it's Fernet-encoded)
        with pytest.raises(json.JSONDecodeError):
            json.loads(encrypted)

    def test_different_encryptions_differ(self, encryptor, sample_storage_state):
        enc1 = encryptor.encrypt(sample_storage_state)
        enc2 = encryptor.encrypt(sample_storage_state)
        # Fernet uses timestamps, so different calls produce different ciphertext
        assert enc1 != enc2

    def test_empty_cookies(self, encryptor):
        data = {"cookies": [], "origins": []}
        encrypted = encryptor.encrypt(data)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == data

    def test_complex_storage_state(self, encryptor):
        data = {
            "cookies": [
                {"name": f"cookie_{i}", "value": f"val_{i}", "domain": f".site{i}.com"}
                for i in range(50)
            ],
            "origins": [
                {"origin": "https://example.com", "localStorage": [{"name": "key", "value": "val"}]}
            ],
        }
        encrypted = encryptor.encrypt(data)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == data


class TestBackwardCompatibility:
    """Test plain JSON fallback for pre-encryption data."""

    def test_plain_json_fallback(self, encryptor, sample_storage_state):
        plain_json = json.dumps(sample_storage_state)
        decrypted = encryptor.decrypt(plain_json)
        assert decrypted == sample_storage_state

    def test_plain_json_non_dict(self, encryptor):
        plain_json = json.dumps([1, 2, 3])
        result = encryptor.decrypt(plain_json)
        assert result is None

    def test_invalid_data(self, encryptor):
        result = encryptor.decrypt("not-valid-anything")
        assert result is None

    def test_empty_string(self, encryptor):
        result = encryptor.decrypt("")
        assert result is None


class TestWrongKey:
    """Test decryption with wrong key fails gracefully."""

    def test_wrong_key_falls_back(self, sample_storage_state):
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        enc1 = SessionEncryptor(key=key1)
        enc2 = SessionEncryptor(key=key2)

        encrypted = enc1.encrypt(sample_storage_state)
        # Wrong key can't decrypt, falls back to plain JSON parse which also fails
        result = enc2.decrypt(encrypted)
        assert result is None
