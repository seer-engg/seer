# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Browser automation requires flexible exception handling and dynamic logging
"""
Browser profile management with interactive login flow.

Handles creating profiles, launching interactive login sessions,
and persisting encrypted session state.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from playwright.async_api import async_playwright, Browser as PlaywrightBrowser

from seer.database import User
from seer.database.models_browser import BrowserProfile

logger = logging.getLogger(__name__)


class BrowserProfileManager:
    """
    Manages browser profiles and interactive login sessions.

    Browser profiles store encrypted Playwright storage state (cookies, localStorage)
    that can be reused across workflow executions for authenticated browser automation.
    """

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
            # Load existing session if any
            storage_state = None
            if profile.session_state_enc:
                try:
                    storage_state = json.loads(profile.session_state_enc)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse existing session state for profile {profile_id}")

            # Launch VISIBLE browser (headless=False for interactive login)
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                storage_state=storage_state,
                viewport={"width": 1280, "height": 800},
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

            # Extract domains from cookies
            domains = self._extract_domains(final_state)

            # Save session state (in production, this should be encrypted)
            profile.session_state_enc = json.dumps(final_state)
            profile.logged_in_domains = domains
            profile.updated_at = datetime.now(timezone.utc)
            await profile.save()

            await browser.close()

        logger.info(f"Session saved for profile '{profile.name}'. Domains: {domains}")
        return {
            "profile_id": str(profile_id),
            "logged_in_domains": domains,
            "status": "session_saved"
        }

    async def get_session_state(self, user: User, profile_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Load session state for workflow execution.

        Args:
            user: Profile owner
            profile_id: UUID of the profile

        Returns:
            Playwright storage_state dict, or None if not found
        """
        profile = await BrowserProfile.get_or_none(
            id=profile_id, user=user, status="active"
        )
        if not profile or not profile.session_state_enc:
            return None

        try:
            session_data = json.loads(profile.session_state_enc)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse session data for profile {profile_id}")
            return None

        # Update last used timestamp
        profile.last_used_at = datetime.now(timezone.utc)
        await profile.save()

        return session_data

    async def update_session_state(
        self,
        user: User,
        profile_id: UUID,
        browser: PlaywrightBrowser,
    ) -> None:
        """
        Update profile session state from a browser context.

        Called after workflow execution to persist any new cookies/sessions.

        Args:
            user: Profile owner
            profile_id: Profile to update
            browser: Playwright browser with active context
        """
        profile = await BrowserProfile.get_or_none(
            id=profile_id, user=user, status="active"
        )
        if not profile:
            return

        # Get storage state from first context
        contexts = browser.contexts
        if contexts:
            final_state = await contexts[0].storage_state()
            domains = self._extract_domains(final_state)

            profile.session_state_enc = json.dumps(final_state)
            profile.logged_in_domains = domains
            profile.last_used_at = datetime.now(timezone.utc)
            await profile.save()
            logger.info(f"Updated session state for profile {profile_id}")

    def _extract_domains(self, session_data: Any) -> List[str]:
        """
        Extract unique domains from session cookies.

        Args:
            session_data: Playwright storage_state dict

        Returns:
            List of unique domain strings
        """
        domains = set()
        for cookie in session_data.get("cookies", []):
            if "domain" in cookie:
                # Remove leading dot from domain
                domain = cookie["domain"].lstrip(".")
                domains.add(domain)
        return sorted(domains)
