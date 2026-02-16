# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Pool management requires flexible exception handling and dynamic logging
"""
Browser pool manager for concurrent session management.

Provides a singleton pool that limits the number of concurrent browser
sessions (each ~100-150MB RAM) via asyncio.Semaphore and tracks active
sessions with automatic reaping of expired ones.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from browser_use import BrowserProfile as BrowserUseProfile
from browser_use import BrowserSession

from seer.config import config
from seer.services.browser.stealth_config import get_stealth_profile_kwargs

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
# Reason: ManagedSession is a cohesive data container; all attributes describe one session
@dataclass
class ManagedSession:
    """Tracks a browser session managed by the pool."""

    id: str
    session: BrowserSession
    user_id: str
    profile_id: Optional[str]
    session_type: str  # "workflow" or "interactive"
    created_at: float = field(default_factory=time.monotonic)
    timeout: int = 300
    recording_id: Optional[str] = None  # RecordingService recording ID for save_recording()
    start_url: Optional[str] = None  # URL session started on (for recording metadata)

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.timeout


class BrowserPoolManager:
    """
    Async singleton managing a pool of browser sessions.

    Limits concurrency via semaphore to prevent memory exhaustion.
    Runs a background reaper task to clean up expired sessions.
    """

    _instance: Optional["BrowserPoolManager"] = None
    _instance_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, max_concurrent: Optional[int] = None) -> None:
        self._max_concurrent = max_concurrent or config.browser_pool_max_concurrent
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._sessions: Dict[str, ManagedSession] = {}
        self._reaper_task: Optional[asyncio.Task] = None
        self._reaper_interval = config.browser_pool_reaper_interval_seconds
        self._shutdown = False

    @classmethod
    async def get_instance(cls) -> "BrowserPoolManager":
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    instance = cls()
                    instance._start_reaper()
                    cls._instance = instance
        return cls._instance

    @classmethod
    async def shutdown_instance(cls) -> None:
        """Shutdown the singleton instance, closing all sessions."""
        async with cls._instance_lock:
            if cls._instance is not None:
                await cls._instance.shutdown()
                cls._instance = None

    def _start_reaper(self) -> None:
        """Start the background session reaper task."""
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._session_reaper())
            logger.info("Browser pool reaper started")

    async def _session_reaper(self) -> None:
        """Background task that periodically kills expired sessions."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._reaper_interval)
                expired = [
                    sid for sid, s in self._sessions.items() if s.is_expired
                ]
                for sid in expired:
                    logger.warning(f"Reaping expired browser session {sid}")
                    try:
                        await self.release_session(sid)
                    except Exception as e:
                        logger.error(f"Error reaping session {sid}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reaper error: {e}")

    def get_session(self, session_id: str) -> Optional[ManagedSession]:
        """Look up an active managed session by ID without releasing it."""
        return self._sessions.get(session_id)

    async def create_session(
        self,
        user_id: str,
        *,
        profile_id: Optional[str] = None,
        session_type: str = "workflow",
        storage_state: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        stealth_mode: bool = False,
    ) -> ManagedSession:
        """Create a new managed browser session.

        Acquires a semaphore slot, creates a browser-use BrowserSession,
        starts the Chromium process, and returns a ManagedSession handle.

        Args:
            user_id: Owning user ID
            profile_id: Optional browser profile ID
            session_type: "workflow" or "interactive"
            storage_state: Optional Playwright storage_state dict for session restore
            timeout: Session timeout in seconds (defaults to config value)
            stealth_mode: Enable anti-detection features (for auth flows like Google sign-in)

        Returns:
            ManagedSession with started browser session
        """
        await self._semaphore.acquire()

        session_id = str(uuid4())
        effective_timeout = timeout or config.browser_pool_default_timeout_seconds

        try:
            # Build profile kwargs based on stealth mode
            if stealth_mode:
                # Use new headless mode with stealth (works on cloud, undetectable)
                profile_kwargs = get_stealth_profile_kwargs()
                profile_kwargs["storage_state"] = storage_state
                profile_kwargs["keep_alive"] = True
                logger.info("Stealth mode enabled with --headless=new")
            else:
                # Standard headless mode for workflows (existing behavior)
                profile_kwargs = {
                    "headless": True,
                    "storage_state": storage_state,
                    "keep_alive": True,
                }

            # Log cookie count for debugging persistence issues
            cookie_count = len(storage_state.get("cookies", [])) if storage_state else 0
            if storage_state:
                logger.info(f"Creating session with {cookie_count} cookies from storage_state")

            browser_profile = BrowserUseProfile(**profile_kwargs)
            browser_session = BrowserSession(browser_profile=browser_profile)
            await browser_session.start()

            # Apply cookies directly via CDP after session start
            # browser-use's StorageStateWatchdog only works with file paths, not dict storage_state
            if storage_state and storage_state.get("cookies"):
                cookies = storage_state["cookies"]
                try:
                    # pylint: disable=protected-access
                    # Reason: browser-use doesn't expose public method to set cookies from dict
                    await browser_session._cdp_set_cookies(cookies)
                    logger.info(f"Applied {len(cookies)} cookies to browser context via CDP")
                except Exception as e:
                    logger.warning(f"Failed to apply cookies via CDP: {e}")

            managed = ManagedSession(
                id=session_id,
                session=browser_session,
                user_id=user_id,
                profile_id=profile_id,
                session_type=session_type,
                timeout=effective_timeout,
            )
            self._sessions[session_id] = managed

            logger.info(
                f"Created browser session {session_id} "
                f"(active: {len(self._sessions)}/{self._max_concurrent})"
            )
            return managed

        except Exception:
            self._semaphore.release()
            raise

    async def release_session(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Release a managed session, exporting state and killing the browser.

        Args:
            session_id: ID of the session to release

        Returns:
            Final storage_state dict from the session, or None on failure
        """
        managed = self._sessions.pop(session_id, None)
        if managed is None:
            logger.warning(f"Session {session_id} not found in pool")
            return None

        storage_state = None
        try:
            storage_state = await managed.session.export_storage_state()
        except Exception as e:
            logger.warning(f"Failed to export storage state for session {session_id}: {e}")

        try:
            await managed.session.stop()
        except Exception as e:
            logger.warning(f"Failed to stop session {session_id}: {e}")

        self._semaphore.release()
        logger.info(
            f"Released browser session {session_id} "
            f"(active: {len(self._sessions)}/{self._max_concurrent})"
        )
        return storage_state

    async def shutdown(self) -> None:
        """Shutdown the pool: cancel reaper and close all sessions."""
        self._shutdown = True

        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass

        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                await self.release_session(sid)
            except Exception as e:
                logger.error(f"Error shutting down session {sid}: {e}")

        logger.info("Browser pool manager shut down")

    def health_status(self) -> Dict[str, Any]:
        """Return pool health status."""
        return {
            "max_concurrent": self._max_concurrent,
            "active_sessions": len(self._sessions),
            "available_slots": self._max_concurrent - len(self._sessions),
            "sessions": [
                {
                    "id": s.id,
                    "user_id": s.user_id,
                    "profile_id": s.profile_id,
                    "session_type": s.session_type,
                    "age_seconds": round(time.monotonic() - s.created_at, 1),
                    "timeout": s.timeout,
                    "is_expired": s.is_expired,
                }
                for s in self._sessions.values()
            ],
        }
