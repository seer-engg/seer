# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Session management requires flexible exception handling and dynamic logging
"""
Session context manager for browser profile state persistence.

Handles loading, saving, and validating encrypted browser session state
(cookies + localStorage) for workflow execution.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from seer.database import User
from seer.database.models_browser import BrowserProfile
from seer.services.browser.encryption import SessionEncryptor

logger = logging.getLogger(__name__)


class SessionContextManager:
    """Manages encrypted browser session state lifecycle."""

    def __init__(self, encryptor: Optional[SessionEncryptor] = None) -> None:
        self._encryptor = encryptor or SessionEncryptor()

    async def load_session_state(
        self, user: User, profile_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Load and decrypt session state for a browser profile.

        Args:
            user: Profile owner
            profile_id: UUID of the browser profile

        Returns:
            Decrypted Playwright storage_state dict, or None if unavailable
        """
        profile = await BrowserProfile.get_or_none(
            id=profile_id, user=user, status="active"
        )
        if not profile or not profile.session_state_enc:
            return None

        session_data = self._encryptor.decrypt(profile.session_state_enc)
        if session_data is None:
            logger.error(f"Failed to decrypt session state for profile {profile_id}")
            return None

        profile.last_used_at = datetime.now(timezone.utc)
        await profile.save()

        logger.info(f"Loaded session state for profile {profile_id}")
        return session_data

    async def save_session_state(
        self,
        user: User,
        profile_id: UUID,
        storage_state: Dict[str, Any],
    ) -> bool:
        """Encrypt and save session state to a browser profile.

        Args:
            user: Profile owner
            profile_id: UUID of the browser profile
            storage_state: Playwright storage_state dict from export_storage_state()

        Returns:
            True if saved successfully, False otherwise
        """
        profile = await BrowserProfile.get_or_none(
            id=profile_id, user=user, status="active"
        )
        if not profile:
            logger.warning(f"Cannot save session state: profile {profile_id} not found")
            return False

        try:
            encrypted = self._encryptor.encrypt(storage_state)
            domains = self._extract_domains(storage_state)

            profile.session_state_enc = encrypted
            profile.logged_in_domains = domains
            profile.last_used_at = datetime.now(timezone.utc)
            await profile.save()

            logger.info(f"Saved encrypted session state for profile {profile_id} (domains: {domains})")
            return True
        except Exception as e:
            logger.error(f"Failed to save session state for profile {profile_id}: {e}")
            return False

    @staticmethod
    def validate_session(storage_state: Optional[Dict[str, Any]]) -> bool:
        """Check if a storage state has valid, non-empty cookies.

        Args:
            storage_state: Playwright storage_state dict

        Returns:
            True if cookies exist (expiry validation is left to the browser)
        """
        if not storage_state:
            return False
        cookies = storage_state.get("cookies", [])
        return len(cookies) > 0

    @staticmethod
    def _extract_domains(session_data: Any) -> List[str]:
        """Extract unique domains from session cookies.

        Args:
            session_data: Playwright storage_state dict

        Returns:
            Sorted list of unique domain strings
        """
        domains = set()
        for cookie in session_data.get("cookies", []):
            if "domain" in cookie:
                domain = cookie["domain"].lstrip(".")
                domains.add(domain)
        return sorted(domains)
