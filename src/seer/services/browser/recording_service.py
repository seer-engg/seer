# pylint: disable=broad-exception-caught,logging-fstring-interpolation,too-many-lines
# Reason: Recording requires flexible exception handling and dynamic logging; module is intentionally large due to chunked recording complexity
"""
rrweb session recording service for browser observability and replay.

Injects rrweb into browser pages via CDP, stores events in JavaScript,
and retrieves them via polling when saving recordings.

ARCHITECTURE OVERVIEW
=====================

1. INLINE RRWEB LIBRARY (CSP Bypass)
   ---------------------------------
   Enterprise apps (Gmail, Salesforce, etc.) use Content Security Policy (CSP)
   to block loading external scripts from CDNs. Instead of:

       <script src="https://cdn.jsdelivr.net/npm/rrweb/...">  # Blocked by CSP!

   We inline the entire rrweb library (~68KB) directly in the injection script
   and execute it via CDP's Runtime.evaluate. This bypasses CSP because:
   - We're not creating a <script> tag with an external src
   - Runtime.evaluate executes code in the page context directly
   - The browser treats it as trusted code from the extension/automation

2. JAVASCRIPT-SIDE EVENT STORAGE
   ------------------------------
   Events are stored in window.__seer_events array in JavaScript.
   On save_recording(), we poll this array via Runtime.evaluate.

   Why not use CDP bindings (Runtime.bindingCalled)?
   - CDP event handlers can be overwritten by other code (like browser_use
     Agent's DOMService) using the same event registry
   - Timing issues with binding registration after navigation
   - JS polling is simpler and more reliable

3. RECORDING FLOW
   ---------------
   start_recording() -> Injects rrweb script into page via CDP
                     -> Script stores events in window.__seer_events
   save_recording()  -> Polls window.__seer_events via Runtime.evaluate
                     -> Compresses events with gzip
                     -> Saves to SessionRecording table

UPDATING RRWEB VERSION
======================
To update the bundled rrweb library:

1. Download new version:
   curl -o src/seer/services/browser/rrweb-record.min.js \\
     "https://cdn.jsdelivr.net/npm/rrweb@VERSION/dist/record/rrweb-record.min.js"

2. Verify the export name is still 'rrwebRecord':
   head -c 100 src/seer/services/browser/rrweb-record.min.js
   # Should show: var rrwebRecord=function()...

3. Run tests:
   uv run pytest tests/unit/services/browser/test_recording_service.py -v
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from tortoise.functions import Sum

from seer.config import config
from seer.database import User
from seer.database.models_browser_recording import SessionRecording, SessionRecordingChunk

if TYPE_CHECKING:
    from browser_use import BrowserSession

logger = logging.getLogger(__name__)

# Flush interval for chunked recordings (seconds)
FLUSH_INTERVAL_SECONDS = 45


@dataclass
class ActiveRecording:  # pylint: disable=too-many-instance-attributes  # Reason: Chunked recording requires tracking IDs, browser session, timing, user, and flush state
    """Tracks state for an active recording session with periodic flushing."""

    recording_id: str
    session_id: str
    browser_session: "BrowserSession"
    start_time: float
    user: User
    profile_id: Optional[str] = None
    workflow_run_id: Optional[str] = None
    session_type: str = "workflow"
    start_url: Optional[str] = None

    # Chunked recording state
    db_recording_id: Optional[str] = None  # Set after first DB record created
    chunk_sequence: int = 0
    total_flushed_events: int = 0
    flush_task: Optional[asyncio.Task[None]] = field(default=None, repr=False)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

# Load inlined rrweb library at module level (once)
# This bypasses CSP restrictions that block loading external scripts from CDN
_RRWEB_LIBRARY_CODE: Optional[str] = None


def _get_rrweb_library() -> str:
    """Load the bundled rrweb-record.min.js library.

    The library is inlined to bypass Content Security Policy (CSP) restrictions
    that block loading external scripts from CDN on sites like Gmail.

    Returns:
        Minified rrweb recording library JavaScript code
    """
    global _RRWEB_LIBRARY_CODE  # pylint: disable=global-statement  # Reason: Module-level caching for performance
    if _RRWEB_LIBRARY_CODE is None:
        _RRWEB_LIBRARY_CODE = resources.files("seer.services.browser").joinpath(
            "rrweb-record.min.js"
        ).read_text(encoding="utf-8")
    return _RRWEB_LIBRARY_CODE


# rrweb injection script template with INLINED library code.
#
# ARCHITECTURE: JavaScript-side event storage with polling
# =========================================================
# Events are stored in window.__seer_events array in JavaScript.
# On save_recording(), we poll this array via Runtime.evaluate.
#
# Why inline instead of CDN:
# - Gmail, Salesforce, and other enterprise apps have strict CSP policies
# - CSP blocks loading external scripts from CDN (script.src fails)
# - Inline code executed via Runtime.evaluate bypasses CSP restrictions
#
# Why this storage approach:
# - CDP event handlers (Runtime.bindingCalled) can be overwritten by other
#   code (like browser_use Agent's DOMService) using the same event registry
# - Timing issues with binding registration after navigation
# - This approach is simpler and more reliable: store in JS, poll on save
#
# NOTE: __seer_rrweb_loaded is set ONLY after rrweb successfully starts.
#
_RRWEB_INJECT_SCRIPT = """
(function() {
    // Skip if already loading or loaded
    if (window.__seer_rrweb_loading || window.__seer_rrweb_loaded) return;
    window.__seer_rrweb_loading = true;

    // Initialize events array for storing rrweb events
    window.__seer_events = window.__seer_events || [];

    // Flush function: returns events AND clears the array (for periodic flushing)
    window.__seer_flush_events = function() {
        var events = window.__seer_events.slice();
        window.__seer_events = [];
        return JSON.stringify(events);
    };

    // Get events without clearing (for final collection)
    window.__seer_get_events = function() {
        return JSON.stringify(window.__seer_events || []);
    };

    try {
        // Inline rrweb library (bypasses CSP - no external script load)
        ${RRWEB_INLINE_CODE}

        // Start recording
        rrwebRecord({
            emit(event) {
                // Store events in JS array (limit to prevent memory issues)
                if (window.__seer_events && window.__seer_events.length < 50000) {
                    window.__seer_events.push(event);
                }
            },
            sampling: { mousemove: false, mouseInteraction: true, scroll: 150, input: 'last' },
        });

        // Only set loaded flag AFTER rrweb successfully starts
        window.__seer_rrweb_loaded = true;
        console.debug('[seer] rrweb recording started (inline)');
    } catch (e) {
        console.error('[seer] rrweb init failed:', e);
        window.__seer_rrweb_loading = false;
    }
})();
"""


class RecordingService:
    """Manages rrweb session recording via CDP injection and JS polling.

    Singleton service to ensure browser session references are available
    between start_recording() and save_recording() calls.

    For long-running sessions (>45s), events are flushed periodically to
    SessionRecordingChunk records to prevent data loss on CDP target detachment.
    """

    _instance: Optional["RecordingService"] = None
    _instance_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        # Legacy tracking (kept for backwards compatibility during transition)
        self._recording_ids: Dict[str, str] = {}   # session_id -> recording_id
        self._start_times: Dict[str, float] = {}   # session_id -> start timestamp
        self._browser_sessions: Dict[str, "BrowserSession"] = {}  # session_id -> BrowserSession
        # New chunked recording tracking
        self._active_recordings: Dict[str, ActiveRecording] = {}  # session_id -> ActiveRecording

    @classmethod
    async def get_instance(cls) -> "RecordingService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def start_recording(  # pylint: disable=too-many-arguments  # Reason: Recording setup requires session ID, browser session, URL, user, profile, run ID, and type
        self,
        session_id: str,
        browser_session: "BrowserSession",
        start_url: Optional[str] = None,
        *,
        user: Optional[User] = None,
        profile_id: Optional[str] = None,
        workflow_run_id: Optional[str] = None,
        session_type: str = "workflow",
    ) -> str:
        """Inject rrweb into the page and prepare for event collection.

        Args:
            session_id: Pool session ID
            browser_session: Active browser-use BrowserSession
            start_url: Optional URL for recording metadata
            user: Recording owner (required for chunked recording support)
            profile_id: Optional browser profile ID
            workflow_run_id: Optional workflow run ID
            session_type: Recording type (default: "workflow")

        Returns:
            Recording ID (UUID string)
        """
        recording_id = str(uuid4())

        # Legacy tracking (for backwards compatibility)
        self._recording_ids[session_id] = recording_id
        self._start_times[session_id] = time.monotonic()
        self._browser_sessions[session_id] = browser_session

        # New chunked recording tracking (only if user provided)
        if user:
            active = ActiveRecording(
                recording_id=recording_id,
                session_id=session_id,
                browser_session=browser_session,
                start_time=time.monotonic(),
                user=user,
                profile_id=profile_id,
                workflow_run_id=workflow_run_id,
                session_type=session_type,
                start_url=start_url,
            )
            self._active_recordings[session_id] = active

            # Start background flush task
            active.flush_task = asyncio.create_task(
                self._flush_loop(session_id),
                name=f"recording-flush-{session_id[:8]}",
            )
            logger.debug(f"Started flush loop for session {session_id}")

        page = await browser_session.must_get_current_page()
        # browser-use stores target_id as private _target_id (no public property exposed)
        cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)  # pylint: disable=protected-access
        cdp_client = browser_session.cdp_client
        sid = cdp_session.session_id

        # Build injection script with inlined rrweb library
        inject_script = _RRWEB_INJECT_SCRIPT.replace(
            "${RRWEB_INLINE_CODE}", _get_rrweb_library()
        )

        # Inject into future navigations (runs on each new document)
        await cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
            params={"source": inject_script},
            session_id=sid,
        )

        # Inject into current page
        await cdp_client.send.Runtime.evaluate(
            params={"expression": inject_script, "awaitPromise": True},
            session_id=sid,
        )

        logger.info(f"Recording started for session {session_id} (recording_id={recording_id})")
        return recording_id

    async def _flush_loop(self, session_id: str) -> None:
        """Background task that periodically flushes events to DB chunks.

        Runs until stop_event is set or the session is removed.
        """
        active = self._active_recordings.get(session_id)
        if not active:
            return

        while not active.stop_event.is_set():
            try:
                # Wait for flush interval or stop signal
                await asyncio.wait_for(
                    active.stop_event.wait(),
                    timeout=FLUSH_INTERVAL_SECONDS,
                )
                # If we get here, stop_event was set
                break
            except asyncio.TimeoutError:
                # Timeout means we should flush
                await self._flush_events_to_chunk(session_id)

        logger.debug(f"Flush loop stopped for session {session_id}")

    async def _flush_events_to_chunk(self, session_id: str) -> bool:
        """Flush current events to a DB chunk, clearing the JS array.

        Returns:
            True if flush succeeded or no events, False on error
        """
        active = self._active_recordings.get(session_id)
        if not active:
            return False

        try:
            events = await self._collect_and_clear_events(session_id)
            if not events:
                return True  # No events to flush, not an error

            # Create parent recording on first flush
            if not active.db_recording_id:
                await self._create_recording_record(active)

            # Create chunk
            chunk_data = self.compress_events(events)
            chunk_id = str(uuid4())

            await SessionRecordingChunk.create(
                id=chunk_id,
                recording_id=active.db_recording_id,
                sequence_number=active.chunk_sequence,
                events_compressed=chunk_data,
                event_count=len(events),
                compressed_size_bytes=len(chunk_data),
            )

            active.chunk_sequence += 1
            active.total_flushed_events += len(events)

            # Update parent recording metadata
            await SessionRecording.filter(id=active.db_recording_id).update(
                chunk_count=active.chunk_sequence,
                last_flush_at=datetime.now(timezone.utc),
            )

            logger.debug(
                f"Flushed chunk {active.chunk_sequence} for {session_id} "
                f"({len(events)} events, {len(chunk_data)} bytes)"
            )
            return True

        except Exception as e:
            # Log but don't crash - next flush may succeed
            logger.warning(f"Failed to flush events for {session_id}: {e}")
            return False

    async def _collect_and_clear_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Collect events AND clear the JS array atomically (for flush)."""
        active = self._active_recordings.get(session_id)
        if not active:
            return []

        try:
            # pylint: disable=protected-access
            page = await active.browser_session.must_get_current_page()
            cdp_session = await active.browser_session.get_or_create_cdp_session(page._target_id)
            cdp_client = active.browser_session.cdp_client
            sid = cdp_session.session_id

            # Use flush function that clears array after read
            result = await cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": "window.__seer_flush_events ? window.__seer_flush_events() : JSON.stringify([])",
                    "returnByValue": True,
                },
                session_id=sid,
            )

            events_json = result.get("result", {}).get("value", "[]")
            return json.loads(events_json)

        except Exception as e:
            logger.warning(f"Failed to collect/clear events for {session_id}: {e}")
            return []

    async def _create_recording_record(self, active: ActiveRecording) -> None:
        """Create the parent SessionRecording record for chunked recording."""
        recording = await SessionRecording.create(
            id=active.recording_id,
            user=active.user,
            browser_profile_id=active.profile_id,
            workflow_run_id=active.workflow_run_id,
            session_type=active.session_type,
            start_url=active.start_url,
            status="recording",
            is_chunked=True,
            events_compressed=None,  # Chunks store the data
        )
        active.db_recording_id = str(recording.id)
        logger.debug(f"Created chunked recording record {active.db_recording_id}")

    async def _collect_events_from_page(self, session_id: str) -> List[Dict]:
        """Retrieve rrweb events from page JavaScript.

        Waits for rrweb to be loaded (up to 2 seconds), then polls
        window.__seer_events array via Runtime.evaluate.

        Args:
            session_id: Pool session ID

        Returns:
            List of rrweb event dicts
        """
        browser_session = self._browser_sessions.get(session_id)
        if not browser_session:
            logger.warning(f"No browser session found for {session_id}")
            return []

        try:
            # pylint: disable=protected-access
            # Reason: browser-use stores target_id as private _target_id
            page = await browser_session.must_get_current_page()
            cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)
            cdp_client = browser_session.cdp_client
            sid = cdp_session.session_id

            # Debug: Get current URL
            url_result = await cdp_client.send_raw(
                "Runtime.evaluate",
                {"expression": "window.location.href", "returnByValue": True},
                session_id=sid,
            )
            current_url = url_result.get("result", {}).get("value", "unknown")
            logger.debug(f"Collecting events from page: {current_url}")

            # Wait for rrweb to be loaded (up to 2 seconds)
            rrweb_loaded = False
            for attempt in range(20):
                loaded_result = await cdp_client.send_raw(
                    "Runtime.evaluate",
                    {"expression": "window.__seer_rrweb_loaded === true", "returnByValue": True},
                    session_id=sid,
                )
                if loaded_result.get("result", {}).get("value") is True:
                    logger.debug(f"rrweb loaded after {attempt * 100}ms on {current_url}")
                    rrweb_loaded = True
                    break
                await asyncio.sleep(0.1)

            if not rrweb_loaded:
                logger.warning(f"rrweb not loaded on {current_url} after 2s")

            # Collect events
            result = await cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": "JSON.stringify(window.__seer_events || [])",
                    "returnByValue": True,
                },
                session_id=sid,
            )

            events_json = result.get("result", {}).get("value", "[]")
            events = json.loads(events_json)
            logger.info(f"Collected {len(events)} events from {current_url}")
            return events
        except Exception as e:
            logger.warning(f"Failed to collect events from page for {session_id}: {e}")
            return []

    async def stop_recording(self, session_id: str) -> None:
        """Stop recording for a session.

        Cleans up session references. Events are not lost - they remain
        in the browser's window.__seer_events until collected.

        Args:
            session_id: Pool session ID
        """
        if session_id in self._recording_ids:
            events_count = 0
            browser_session = self._browser_sessions.get(session_id)
            if browser_session:
                try:
                    events = await self._collect_events_from_page(session_id)
                    events_count = len(events)
                except Exception:
                    pass

            logger.info(
                f"Recording stopped for session {session_id} "
                f"(events: {events_count})"
            )

    async def save_recording(
        self,
        session_id: str,
        user: User,
        *,
        profile_id: Optional[str] = None,
        workflow_run_id: Optional[str] = None,
        session_type: str = "workflow",
        start_url: Optional[str] = None,
    ) -> Optional[str]:
        """Collect events from page, compress, and create SessionRecording.

        For chunked recordings (with flush loop), this:
        1. Stops the flush loop
        2. Collects any remaining events as a final chunk
        3. Updates the parent recording to completed status

        For non-chunked recordings, creates a single SessionRecording with all events.

        Args:
            session_id: Pool session ID
            user: Recording owner
            profile_id: Optional browser profile ID
            workflow_run_id: Optional workflow run ID
            session_type: Recording type (default: "workflow")
            start_url: URL the session started on

        Returns:
            Recording ID string, or None if no events were recorded
        """
        # Check if this is a chunked recording
        active = self._active_recordings.pop(session_id, None)
        if active:
            return await self._finalize_chunked_recording(active, session_id)

        # Legacy path: non-chunked recording (backwards compatibility)
        return await self._save_non_chunked_recording(
            session_id, user, profile_id, workflow_run_id, session_type, start_url
        )

    async def _finalize_chunked_recording(
        self,
        active: ActiveRecording,
        session_id: str,
    ) -> Optional[str]:
        """Finalize a chunked recording: stop flush loop, save final chunk, update status."""
        # Stop the flush loop
        active.stop_event.set()
        if active.flush_task:
            try:
                await asyncio.wait_for(active.flush_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(f"Flush task did not stop cleanly for {session_id}")

        # Collect any remaining events BEFORE cleanup (don't clear - this is final collection)
        remaining_events = await self._collect_events_from_page(session_id)

        # Clean up legacy tracking AFTER event collection
        self._recording_ids.pop(session_id, None)
        self._start_times.pop(session_id, None)
        self._browser_sessions.pop(session_id, None)

        duration_ms = int((time.monotonic() - active.start_time) * 1000)
        total_events = active.total_flushed_events + len(remaining_events)

        # If we have chunks already saved
        if active.db_recording_id:
            # Add final chunk if there are remaining events
            if remaining_events:
                chunk_data = self.compress_events(remaining_events)
                await SessionRecordingChunk.create(
                    id=str(uuid4()),
                    recording_id=active.db_recording_id,
                    sequence_number=active.chunk_sequence,
                    events_compressed=chunk_data,
                    event_count=len(remaining_events),
                    compressed_size_bytes=len(chunk_data),
                )
                active.chunk_sequence += 1

            # Update parent recording to completed
            total_compressed_size = await self._calculate_total_chunk_size(active.db_recording_id)
            await SessionRecording.filter(id=active.db_recording_id).update(
                status="completed",
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                event_count=total_events,
                chunk_count=active.chunk_sequence,
                compressed_size_bytes=total_compressed_size,
            )

            logger.info(
                f"Saved chunked recording {active.db_recording_id} for session {session_id} "
                f"({total_events} events in {active.chunk_sequence} chunks)"
            )
            return active.db_recording_id

        # No chunks created yet (short session) - save as traditional recording
        if not remaining_events:
            logger.info(f"No events to save for session {session_id}")
            return None

        compressed = self.compress_events(remaining_events)
        max_size = config.browser_recording_max_size_mb * 1024 * 1024
        if len(compressed) > max_size:
            truncated = remaining_events[: int(len(remaining_events) * 0.75)]
            compressed = self.compress_events(truncated)
            remaining_events = truncated

        try:
            await SessionRecording.create(
                id=active.recording_id,
                user=active.user,
                browser_profile_id=active.profile_id,
                workflow_run_id=active.workflow_run_id,
                session_type=active.session_type,
                events_compressed=compressed,
                event_count=len(remaining_events),
                duration_ms=duration_ms,
                compressed_size_bytes=len(compressed),
                start_url=active.start_url,
                status="completed",
                completed_at=datetime.now(timezone.utc),
                is_chunked=False,
            )
            logger.info(
                f"Saved recording {active.recording_id} for session {session_id} "
                f"(events={len(remaining_events)}, size={len(compressed)}B)"
            )
            return active.recording_id
        except Exception as e:
            logger.error(f"Failed to save recording for session {session_id}: {e}")
            return None

    async def _save_non_chunked_recording(  # pylint: disable=too-many-positional-arguments  # Reason: Legacy path requires all recording metadata as positional args for backwards compatibility
        self,
        session_id: str,
        user: User,
        profile_id: Optional[str],
        workflow_run_id: Optional[str],
        session_type: str,
        start_url: Optional[str],
    ) -> Optional[str]:
        """Legacy path: save non-chunked recording (backwards compatibility)."""
        # Collect events from the browser page
        events = await self._collect_events_from_page(session_id)

        recording_id = self._recording_ids.pop(session_id, None)
        start_time = self._start_times.pop(session_id, None)
        self._browser_sessions.pop(session_id, None)

        if not events or not recording_id:
            logger.info(f"No events to save for session {session_id}")
            return None

        compressed = self.compress_events(events)
        max_size = config.browser_recording_max_size_mb * 1024 * 1024
        if len(compressed) > max_size:
            logger.warning(
                f"Recording for session {session_id} exceeds max size "
                f"({len(compressed)} > {max_size}), truncating events"
            )
            # Truncate to ~75% of events and re-compress
            truncated = events[: int(len(events) * 0.75)]
            compressed = self.compress_events(truncated)
            events = truncated

        duration_ms = int((time.monotonic() - start_time) * 1000) if start_time else 0

        try:
            await SessionRecording.create(
                id=recording_id,
                user=user,
                browser_profile_id=profile_id,
                workflow_run_id=workflow_run_id,
                session_type=session_type,
                events_compressed=compressed,
                event_count=len(events),
                duration_ms=duration_ms,
                compressed_size_bytes=len(compressed),
                start_url=start_url,
                status="completed",
                completed_at=datetime.now(timezone.utc),
            )
            logger.info(
                f"Saved recording {recording_id} for session {session_id} "
                f"(events={len(events)}, size={len(compressed)}B)"
            )
            return recording_id
        except Exception as e:
            logger.error(f"Failed to save recording for session {session_id}: {e}")
            return None

    async def _calculate_total_chunk_size(self, recording_id: str) -> int:
        """Calculate total compressed size across all chunks."""
        result = await SessionRecordingChunk.filter(recording_id=recording_id).annotate(
            total=Sum("compressed_size_bytes")
        ).values("total")
        return result[0]["total"] or 0 if result else 0

    @staticmethod
    def compress_events(events: List[Dict[str, Any]]) -> bytes:
        """Compress rrweb events to gzip bytes."""
        return gzip.compress(json.dumps(events).encode("utf-8"))

    @staticmethod
    def decompress_events(data: bytes) -> List[Dict[str, Any]]:
        """Decompress gzip bytes to rrweb event list."""
        return json.loads(gzip.decompress(data))

    @staticmethod
    async def get_recording_events(recording_id: str) -> List[Dict[str, Any]]:
        """Get all events for a recording, reassembling chunks if needed.

        Handles both chunked and non-chunked recordings transparently.

        Args:
            recording_id: UUID string of the recording

        Returns:
            List of rrweb event dicts in chronological order
        """
        recording = await SessionRecording.get_or_none(id=recording_id)
        if not recording:
            return []

        if not recording.is_chunked:
            # Traditional single-blob recording
            if recording.events_compressed:
                return RecordingService.decompress_events(recording.events_compressed)
            return []

        # Chunked recording - reassemble from chunks in sequence order
        chunks = await SessionRecordingChunk.filter(
            recording_id=recording_id
        ).order_by("sequence_number").all()

        all_events: List[Dict[str, Any]] = []
        for chunk in chunks:
            events = RecordingService.decompress_events(chunk.events_compressed)
            all_events.extend(events)

        return all_events
