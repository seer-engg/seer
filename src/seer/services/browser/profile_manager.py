# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Browser automation requires flexible exception handling and dynamic logging
"""
Browser profile management with interactive login flow.

Handles creating profiles, launching interactive login sessions,
and persisting encrypted session state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from seer.config import config
from seer.database import User
from seer.database.models_browser import BrowserProfile
from seer.services.browser.encryption import SessionEncryptor
from seer.services.browser.pool_manager import BrowserPoolManager
from seer.services.browser.session_context_manager import SessionContextManager
from seer.logger import get_logger

logger = get_logger(__name__)


class BrowserProfileManager:
    """
    Manages browser profiles and interactive login sessions.

    Browser profiles store encrypted Playwright storage state (cookies, localStorage)
    that can be reused across workflow executions for authenticated browser automation.
    """

    def __init__(self) -> None:
        self._encryptor = SessionEncryptor()
        self._session_context = SessionContextManager(self._encryptor)

    async def create_profile(self, user: User, name: str) -> BrowserProfile:
        """
        Create a new empty browser profile.

        Args:
            user: Profile owner
            name: Human-readable profile name (e.g., "Work Profile", "Personal")

        Returns:
            Created BrowserProfile instance
        """
        logger.info(f"Creating browser profile '{name}' for user {user.user_id}")
        profile = await BrowserProfile.create(user=user, name=name)
        return profile

    async def list_profiles(self, user: User) -> List[Dict[str, Any]]:
        """
        List all active browser profiles for a user.

        Returns:
            List of profile metadata dicts (excludes encrypted session data)
        """
        profiles = await BrowserProfile.filter(user=user, status="active").all()
        return [
            {
                "id": str(p.id),
                "name": p.name,
                "logged_in_domains": p.logged_in_domains or [],
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "last_used_at": p.last_used_at.isoformat() if p.last_used_at else None,
            }
            for p in profiles
        ]

    async def get_profile(self, user: User, profile_id: UUID) -> Optional[BrowserProfile]:
        """
        Get a specific browser profile by ID.

        Args:
            user: Profile owner
            profile_id: UUID of the profile

        Returns:
            BrowserProfile if found and active, None otherwise
        """
        return await BrowserProfile.get_or_none(
            id=profile_id, user=user, status="active"
        )

    async def delete_profile(self, user: User, profile_id: UUID) -> bool:
        """
        Soft-delete a browser profile.

        Args:
            user: Profile owner
            profile_id: UUID of the profile to delete

        Returns:
            True if deleted, False if not found
        """
        updated = await BrowserProfile.filter(
            id=profile_id, user=user
        ).update(status="deleted")
        if updated > 0:
            logger.info(f"Deleted browser profile {profile_id}")
        return updated > 0

    async def get_session_state(self, user: User, profile_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Load session state for workflow execution.

        Delegates to SessionContextManager for decryption and backward compatibility.

        Args:
            user: Profile owner
            profile_id: UUID of the profile

        Returns:
            Playwright storage_state dict, or None if not found
        """
        return await self._session_context.load_session_state(user, profile_id)

    async def update_session_state(
        self,
        user: User,
        profile_id: UUID,
        storage_state: Dict[str, Any],
    ) -> None:
        """
        Update profile session state with new storage state.

        Called after workflow execution to persist any new cookies/sessions.

        Args:
            user: Profile owner
            profile_id: Profile to update
            storage_state: Playwright storage_state dict
        """
        await self._session_context.save_session_state(user, profile_id, storage_state)

    async def create_interactive_session(
        self,
        user: User,
        profile_id: UUID,
        target_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pooled session for interactive login via streaming.

        Creates a remote browser session via Browserless that can be
        viewed and controlled remotely via WebSocket streaming.

        Args:
            user: Profile owner
            profile_id: Profile to use for session state
            target_url: Optional starting URL

        Returns:
            Dict with session_id, profile_id, and status
        """
        profile = await BrowserProfile.get(id=profile_id, user=user, status="active")
        logger.info(f"Creating interactive session for profile '{profile.name}' ({profile_id})")

        # Load existing session state
        storage_state = None
        if profile.session_state_enc:
            storage_state = self._encryptor.decrypt(profile.session_state_enc)
            cookie_count = len(storage_state.get("cookies", [])) if storage_state else 0
            logger.info(f"Loaded {cookie_count} cookies from encrypted profile {profile_id}")

        # Create pooled session with stealth mode for Google auth compatibility
        pool = await BrowserPoolManager.get_instance()
        managed = await pool.create_session(
            user_id=str(user.user_id),
            profile_id=str(profile_id),
            session_type="interactive",
            storage_state=storage_state,
            timeout=config.browser_interactive_timeout_seconds,
        )

        # Navigate to target URL
        try:
            page = await managed.session.must_get_current_page()
            url = target_url or "about:blank"
            await managed.session.cdp_client.send.Page.navigate(
                params={"url": url},
                # pylint: disable-next=protected-access  # Reason: browser-use library stores target_id as private _target_id (no public property exposed)
                session_id=(await managed.session.get_or_create_cdp_session(page._target_id)).session_id,
            )
        except Exception as e:
            logger.warning(f"Failed to navigate to {target_url}: {e}")

        return {
            "session_id": managed.id,
            "profile_id": str(profile_id),
            "status": "created",
        }

    async def complete_interactive_session(
        self,
        user: User,
        profile_id: UUID,
        session_id: str,
    ) -> Dict[str, Any]:
        """Complete interactive session: export state, save encrypted, release pool.

        Args:
            user: Profile owner
            profile_id: Profile to update
            session_id: Pool session ID to complete

        Returns:
            Dict with profile_id, logged_in_domains, and status
        """
        pool = await BrowserPoolManager.get_instance()
        managed = pool.get_session(session_id)
        if not managed:
            raise ValueError(f"Session {session_id} not found in pool")
        if managed.user_id != str(user.user_id):
            raise PermissionError("Session does not belong to this user")

        # Release session (exports storage state and stops browser)
        storage_state = await pool.release_session(session_id)

        # Save encrypted state if we got storage_state back
        domains: List[str] = []
        if storage_state:
            await self._session_context.save_session_state(user, profile_id, storage_state)
            domains = SessionContextManager.extract_domains(storage_state)

        logger.info(f"Completed interactive session {session_id} for profile {profile_id}")
        return {
            "profile_id": str(profile_id),
            "logged_in_domains": domains,
            "status": "session_saved",
        }

    def _extract_domains(self, session_data: Any) -> List[str]:
        """
        Extract unique domains from session cookies.

        Args:
            session_data: Playwright storage_state dict

        Returns:
            List of unique domain strings
        """
        return SessionContextManager.extract_domains(session_data)
