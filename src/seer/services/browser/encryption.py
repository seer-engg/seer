# pylint: disable=broad-exception-caught  # Reason: Encryption failures must not crash workflows
"""
Fernet-based encryption for browser session state.

Encrypts Playwright storage_state dicts (cookies + localStorage)
before persisting to the database, and decrypts on load.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


class SessionEncryptor:
    """Encrypts and decrypts browser session state using Fernet symmetric encryption."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        self._key = key or config.browser_encryption_key_bytes
        self._fernet = Fernet(self._key)

    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt a session state dict to a string.

        Args:
            data: Playwright storage_state dict with cookies/origins

        Returns:
            Base64-encoded encrypted string
        """
        plaintext = json.dumps(data).encode("utf-8")
        return self._fernet.encrypt(plaintext).decode("utf-8")

    def decrypt(self, ciphertext: str) -> Optional[Dict[str, Any]]:
        """Decrypt an encrypted session state string.

        Includes backward-compatibility fallback: if decryption fails,
        attempts to parse as plain JSON (for pre-encryption data).

        Args:
            ciphertext: Encrypted string or legacy plain JSON

        Returns:
            Decrypted storage_state dict, or None on failure
        """
        try:
            plaintext = self._fernet.decrypt(ciphertext.encode("utf-8"))
            return json.loads(plaintext)
        except InvalidToken:
            logger.debug("Fernet decryption failed, attempting plain JSON fallback")
            return self._try_plain_json(ciphertext)
        except Exception as e:
            logger.warning("Session decryption failed: %s", e)
            return self._try_plain_json(ciphertext)

    @staticmethod
    def _try_plain_json(data: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse data as plain JSON for backward compatibility."""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                logger.info("Read legacy plain-JSON session state (will re-encrypt on next save)")
                return parsed
            return None
        except (json.JSONDecodeError, TypeError):
            logger.warning("Session data is neither valid encrypted nor valid JSON")
            return None
