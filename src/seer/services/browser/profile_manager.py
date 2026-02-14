# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Browser automation requires flexible exception handling and dynamic logging
"""
Browser profile management with interactive login flow.

Handles creating profiles, launching interactive login sessions,
and persisting encrypted session state.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from playwright.async_api import async_playwright

from seer.config import config
from seer.database import User
from seer.database.models_browser import BrowserProfile
from seer.services.browser.encryption import SessionEncryptor
from seer.services.browser.pool_manager import BrowserPoolManager
from seer.services.browser.recording_service import RecordingService
from seer.services.browser.session_context_manager import SessionContextManager
from seer.services.browser.stealth_config import CHROME_USER_AGENTS, get_headed_stealth_args

logger = logging.getLogger(__name__)


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

    async def start_interactive_login(
        self,
        user: User,
        profile_id: UUID,
        target_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Launch interactive browser for user to log in.

        .. deprecated::
            Use :meth:`create_interactive_session` + streaming WebSocket +
            :meth:`complete_interactive_session` instead for remote/cloud use.

        Opens a visible browser window where the user can navigate
        and log into services. The session is captured when the browser closes.

        Args:
            user: Profile owner
            profile_id: Profile to update with login session
            target_url: Optional starting URL (e.g., "https://slack.com/signin")

        Returns:
            Status dict with profile_id, logged_in_domains, and status
        """
        profile = await BrowserProfile.get(id=profile_id, user=user)
        logger.info(f"Starting interactive login for profile '{profile.name}' ({profile_id})")

        async with async_playwright() as p:
            # Load existing session if any (decrypt)
            storage_state = None
            if profile.session_state_enc:
                storage_state = self._encryptor.decrypt(profile.session_state_enc)

            # Launch VISIBLE browser with stealth args (headless=False for interactive login)
            browser = await p.chromium.launch(
                headless=False,
                args=get_headed_stealth_args(),  # Stealth args without --headless=new
            )
            context = await browser.new_context(
                storage_state=storage_state,
                viewport={"width": 1280, "height": 800},
                user_agent=CHROME_USER_AGENTS.get("linux"),  # Realistic user agent
            )

            page = await context.new_page()

            # Navigate to target or a blank page
            if target_url:
                await page.goto(target_url)
            else:
                await page.goto("about:blank")

            logger.info("Browser opened for interactive login. Waiting for user to close browser...")

            # Wait for user to finish (they close the browser window)
            try:
                while len(context.pages) > 0:
                    await asyncio.sleep(1)
            except Exception:
                pass

            # Capture final session state
            final_state = await context.storage_state()

            # Save encrypted session state via context manager
            await self._session_context.save_session_state(user, profile_id, final_state)

            await browser.close()

        domains = SessionContextManager._extract_domains(final_state)
        logger.info(f"Session saved for profile '{profile.name}'. Domains: {domains}")
        return {
            "profile_id": str(profile_id),
            "logged_in_domains": domains,
            "status": "session_saved"
        }

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
        """Create a pooled headless session for interactive login via streaming.

        Unlike start_interactive_login(), this creates a headless browser session
        that can be viewed and controlled remotely via WebSocket streaming.

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

        # Create pooled session with stealth mode for Google auth compatibility
        pool = await BrowserPoolManager.get_instance()
        managed = await pool.create_session(
            user_id=str(user.user_id),
            profile_id=str(profile_id),
            session_type="interactive",
            storage_state=storage_state,
            timeout=config.browser_interactive_timeout_seconds,
            stealth_mode=True,  # Enable stealth with --headless=new for auth flows
        )

        # Navigate to target URL
        try:
            page = await managed.session.must_get_current_page()
            url = target_url or "about:blank"
            await managed.session.cdp_client.send.Page.navigate(
                params={"url": url},
                # NOTE: browser-use stores target_id as private _target_id (no public property)
                session_id=(await managed.session.get_or_create_cdp_session(page._target_id)).session_id,
            )
        except Exception as e:
            logger.warning(f"Failed to navigate to {target_url}: {e}")

        # Start recording if enabled
        recording_id = None
        if config.browser_recording_enabled:
            try:
                recorder = RecordingService()
                recording_id = await recorder.start_recording(
                    managed.id, managed.session, start_url=target_url
                )
            except Exception as e:
                logger.warning(f"Failed to start recording: {e}")

        return {
            "session_id": managed.id,
            "profile_id": str(profile_id),
            "recording_id": recording_id,
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
            Dict with profile_id, logged_in_domains, recording_id, and status
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
        recording_id = None
        domains: List[str] = []
        if storage_state:
            await self._session_context.save_session_state(user, profile_id, storage_state)
            domains = SessionContextManager._extract_domains(storage_state)

        logger.info(f"Completed interactive session {session_id} for profile {profile_id}")
        return {
            "profile_id": str(profile_id),
            "logged_in_domains": domains,
            "recording_id": recording_id,
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
        return SessionContextManager._extract_domains(session_data)
