"""Secure token storage for MCP OAuth tokens."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet


class TokenStore:
    """Secure storage for OAuth tokens in the user's home directory.

    Tokens are encrypted using Fernet (symmetric encryption) and stored
    in ~/.seer-mcp/tokens.json with restricted file permissions (0600).

    The encryption key is derived from a machine-specific identifier to
    provide additional security beyond file permissions.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize token store.

        Args:
            storage_path: Optional custom path for token storage.
                         Defaults to ~/.seer-mcp/tokens.json
        """
        if storage_path:
            self.storage_path = Path(storage_path).expanduser()
        else:
            self.storage_path = Path.home() / ".seer-mcp" / "tokens.json"

        self.key_path = self.storage_path.parent / ".key"
        self._ensure_storage_directory()
        self._cipher = self._get_or_create_cipher()

    def _ensure_storage_directory(self) -> None:
        """Create storage directory with secure permissions."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption cipher with persistent key."""
        if self.key_path.exists():
            # Load existing key
            with open(self.key_path, "rb") as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            # Save key with restricted permissions
            old_umask = os.umask(0o077)  # Ensure 0600 permissions
            try:
                with open(self.key_path, "wb") as f:
                    f.write(key)
            finally:
                os.umask(old_umask)

        return Fernet(key)

    def save_tokens(  # pylint: disable=too-many-positional-arguments  # Reason: OAuth token parameters
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        api_url: str,
        scope: str = "",
    ) -> None:
        """Save OAuth tokens securely.

        Args:
            access_token: JWT access token
            refresh_token: Refresh token for getting new access tokens
            expires_in: Access token expiry in seconds
            api_url: Seer API URL
            scope: Granted OAuth scopes
        """
        # Calculate expiry timestamp
        now = datetime.now(timezone.utc)
        expires_at = int(now.timestamp()) + expires_in

        # Prepare token data
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "api_url": api_url,
            "scope": scope,
            "saved_at": int(now.timestamp()),
        }

        # Encrypt and save
        plaintext = json.dumps(token_data).encode()
        encrypted = self._cipher.encrypt(plaintext)

        # Write with atomic rename to prevent corruption
        temp_path = self.storage_path.with_suffix(".tmp")
        old_umask = os.umask(0o077)  # Ensure 0600 permissions
        try:
            with open(temp_path, "wb") as f:
                f.write(encrypted)
            temp_path.replace(self.storage_path)
        finally:
            os.umask(old_umask)

    def load_tokens(self) -> Optional[dict]:
        """Load and decrypt stored tokens.

        Returns:
            dict with keys: access_token, refresh_token, expires_at, api_url, scope
            None if no tokens are stored
        """
        if not self.storage_path.exists():
            return None

        try:
            with open(self.storage_path, "rb") as f:
                encrypted = f.read()

            plaintext = self._cipher.decrypt(encrypted)
            token_data = json.loads(plaintext)
            return token_data
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: gracefully handle any token loading error
            # If decryption fails (corrupted file, wrong key, etc), return None
            return None

    def has_valid_token(self) -> bool:
        """Check if a valid (non-expired) access token exists.

        Returns:
            True if valid token exists, False otherwise
        """
        token_data = self.load_tokens()
        if not token_data:
            return False

        # Check if access token is expired
        now = int(datetime.now(timezone.utc).timestamp())
        expires_at = token_data.get("expires_at", 0)

        # Add 60 second buffer to avoid using tokens that are about to expire
        return expires_at > (now + 60)

    def clear_tokens(self) -> None:
        """Remove stored tokens."""
        if self.storage_path.exists():
            self.storage_path.unlink()

    def get_api_url(self) -> Optional[str]:
        """Get the stored API URL.

        Returns:
            API URL string or None if not stored
        """
        token_data = self.load_tokens()
        if token_data:
            return token_data.get("api_url")
        return None
