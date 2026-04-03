"""Tests for BYOK key vault encryption service."""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from seer.services.byok.key_vault import KeyVault


@pytest.fixture
def vault():
    key = Fernet.generate_key()
    return KeyVault(encryption_key=key)


def test_encrypt_decrypt_roundtrip(vault: KeyVault):
    raw = "sk-proj-abc123def456"
    encrypted = vault.encrypt(raw)
    assert encrypted != raw
    assert vault.decrypt(encrypted) == raw


def test_mask_key():
    assert KeyVault.mask("sk-proj-abc123def456") == "sk-p...f456"
    assert KeyVault.mask("short") == "...hort"


def test_suffix():
    assert KeyVault.suffix("sk-proj-abc123def456") == "f456"


def test_decrypt_invalid_returns_none(vault: KeyVault):
    assert vault.decrypt("not-valid-ciphertext") is None
