# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Pool management requires flexible exception handling and dynamic logging
"""
Browser pool manager for concurrent session management.

Provides a singleton pool that limits the number of concurrent browser
sessions via asyncio.Semaphore, connecting to a remote Browserless
service over CDP. Tracks active sessions with automatic reaping of
expired ones.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from browser_use import BrowserProfile as BrowserUseProfile
from browser_use import BrowserSession

from seer.config import config
from seer.logger import get_logger
from seer.services.browser.stealth_config import (
    get_platform_user_agent,
    get_remote_profile_kwargs,
    get_stealth_scripts_combined,
)

logger = get_logger(__name__)


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
    hitl_paused: bool = False  # Set True during HITL wait to prevent reaping

    @property
    def is_expired(self) -> bool:
        if self.hitl_paused:
            return False
        return (time.monotonic() - self.created_at) > self.timeout


class BrowserPoolManager:
    """
    Async singleton managing a pool of browser sessions.

    Connects to a remote Browserless service via CDP WebSocket.
    Limits concurrency via semaphore to prevent resource exhaustion.
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

    async def _inject_stealth_scripts(self, browser_session: BrowserSession) -> None:
        """Inject anti-detection stealth scripts and UA override into the browser session.

        - Forces user-agent via CDP Emulation.setUserAgentOverride (Browserless
          ignores BrowserProfile.user_agent for remote CDP connections).
        - Uses Page.addScriptToEvaluateOnNewDocument so JS stealth scripts run
          before any page JS on every navigation.
        """
        try:
            page = await browser_session.must_get_current_page()
            # pylint: disable=protected-access
            # Reason: browser-use stores target_id as private _target_id
            cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)
            cdp_client = browser_session.cdp_client
            sid = cdp_session.session_id

            # Override User-Agent at the CDP level — Browserless defaults to
            # "HeadlessChrome/..." which is an instant detection signal.
            ua = get_platform_user_agent()
            await cdp_client.send.Emulation.setUserAgentOverride(
                params={
                    "userAgent": ua,
                    "acceptLanguage": "en-US,en;q=0.9",
                    "platform": "Linux x86_64",
                },
                session_id=sid,
            )

            stealth_js = get_stealth_scripts_combined()

            # Inject into all future navigations
            await cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
                params={"source": stealth_js},
                session_id=sid,
            )

            # Inject into current page
            await cdp_client.send.Runtime.evaluate(
                params={"expression": stealth_js, "awaitPromise": False},
                session_id=sid,
            )

            logger.info("Stealth scripts injected into browser session")
        except Exception as e:
            logger.warning(f"Failed to inject stealth scripts: {e}")

    async def _apply_cookies_via_cdp(self, browser_session: BrowserSession, cookies: list) -> None:
        """Apply cookies to browser session via CDP.

        Primary method for restoring saved session cookies into a
        remote Browserless browser connected over CDP.
        """
        try:
            # pylint: disable=protected-access
            # Reason: browser-use doesn't expose public method to set cookies from dict
            await browser_session._cdp_set_cookies(cookies)
            logger.info(f"Applied {len(cookies)} cookies to browser context via CDP")
        except Exception as e:
            logger.warning(f"Failed to apply cookies via CDP: {e}")

    async def _export_cookies_via_cdp(self, browser_session: BrowserSession) -> Optional[Dict[str, Any]]:
        """Export cookies from browser session via CDP.

        Used when export_storage_state() fails on remote Browserless
        sessions where Playwright context methods may not be available.

        Tries two approaches:
        1. Direct CDP client (works for CDP-only remote sessions)
        2. Playwright BrowserContext fallback (works when context exists)
        """
        # Approach 1: Use the existing CDP client directly.
        # This works for remote Browserless sessions that were connected
        # via cdp_url and never have a Playwright BrowserContext.
        try:
            cdp_client = browser_session.cdp_client
            if cdp_client is not None:
                # Get the target session ID for the current page
                page = await browser_session.must_get_current_page()
                # pylint: disable=protected-access
                # Reason: browser-use stores target_id as private _target_id
                cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)
                result = await cdp_client.send_raw(
                    "Network.getAllCookies",
                    {},
                    session_id=cdp_session.session_id,
                )
                cookies = result.get("cookies", [])
                logger.info(f"Exported {len(cookies)} cookies via CDP client fallback")
                return {"cookies": cookies, "origins": []}
        except Exception as e:
            logger.warning(f"CDP client cookie export failed: {e}")

        # Approach 2: Playwright BrowserContext (if available)
        try:
            context = browser_session.browser_context
            if context is not None:
                pages = context.pages
                if pages:
                    cdp = await pages[0].context.new_cdp_session(pages[0])
                    result = await cdp.send("Network.getAllCookies")
                    cookies = result.get("cookies", [])
                    await cdp.detach()
                    logger.info(f"Exported {len(cookies)} cookies via Playwright CDP fallback")
                    return {"cookies": cookies, "origins": []}
        except Exception as e:
            logger.warning(f"Playwright CDP cookie export failed: {e}")

        return None

    async def create_session(
        self,
        user_id: str,
        *,
        profile_id: Optional[str] = None,
        session_type: str = "workflow",
        storage_state: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ManagedSession:
        """Create a new managed browser session via remote Browserless.

        Acquires a semaphore slot, connects to Browserless over CDP,
        applies cookies if provided, and returns a ManagedSession handle.

        Args:
            user_id: Owning user ID
            profile_id: Optional browser profile ID
            session_type: "workflow" or "interactive"
            storage_state: Optional storage_state dict with cookies to restore
            timeout: Session timeout in seconds (defaults to config value)

        Returns:
            ManagedSession with started browser session
        """
        await self._semaphore.acquire()

        session_id = str(uuid4())
        effective_timeout = timeout or config.browser_pool_default_timeout_seconds
        cdp_url = config.browserless_cdp_url

        try:
            profile_kwargs = get_remote_profile_kwargs()
            profile_kwargs["keep_alive"] = True

            if storage_state:
                cookie_count = len(storage_state.get("cookies", []))
                logger.info(f"Creating session with {cookie_count} cookies from storage_state")

            browser_profile = BrowserUseProfile(**profile_kwargs)

            logger.info(f"Connecting to remote Browserless at {config.browserless_url}")
            browser_session = BrowserSession(
                cdp_url=cdp_url,
                browser_profile=browser_profile,
            )
            await browser_session.start()

            # Inject stealth scripts before any page interaction
            if config.browser_stealth_enabled:
                await self._inject_stealth_scripts(browser_session)

            # Apply cookies via CDP after session start
            if storage_state and storage_state.get("cookies"):
                await self._apply_cookies_via_cdp(browser_session, storage_state["cookies"])

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
            # Fallback: extract cookies via CDP (works for remote Browserless sessions)
            try:
                storage_state = await self._export_cookies_via_cdp(managed.session)
            except Exception as cdp_err:
                logger.warning(f"CDP cookie export fallback also failed for {session_id}: {cdp_err}")

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
