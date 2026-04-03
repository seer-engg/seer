"""Fernet-based encryption vault for BYOK LLM API keys."""
from __future__ import annotations

from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from seer.logger import get_logger

logger = get_logger(__name__)


class KeyVault:
    """Encrypts, decrypts, and masks LLM API keys using Fernet symmetric encryption."""

    def __init__(self, encryption_key: bytes | None = None) -> None:
        if encryption_key is None:
            from seer.config import config  # pylint: disable=import-outside-toplevel  # Reason: Avoid import at module level for testability
            encryption_key = config.browser_encryption_key_bytes
        self._fernet = Fernet(encryption_key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> Optional[str]:
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except (InvalidToken, Exception) as e:  # pylint: disable=broad-exception-caught  # Reason: Fernet may raise undocumented errors on malformed input
            logger.warning("BYOK key decryption failed: %s", type(e).__name__)
            return None

    @staticmethod
    def mask(key: str) -> str:
        """Mask a key for display, showing first 4 and last 4 chars."""
        if len(key) <= 8:
            return f"...{key[-4:]}"
        return f"{key[:4]}...{key[-4:]}"

    @staticmethod
    def suffix(key: str) -> str:
        """Extract last 4 chars for storage as key_suffix."""
        return key[-4:] if len(key) >= 4 else key


_VAULT: KeyVault | None = None


def get_key_vault() -> KeyVault:
    global _VAULT  # noqa: PLW0603  # pylint: disable=global-statement  # Reason: Module-level singleton pattern
    if _VAULT is None:
        _VAULT = KeyVault()
    return _VAULT
