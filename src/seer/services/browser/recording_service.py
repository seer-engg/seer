# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: Recording requires flexible exception handling and dynamic logging
"""
rrweb session recording service for browser observability and replay.

Injects rrweb into browser pages via CDP, collects events via JS binding,
and stores compressed recordings in the database.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from browser_use import BrowserSession

from seer.config import config
from seer.database import User
from seer.database.models_browser_recording import SessionRecording

logger = logging.getLogger(__name__)

# rrweb injection script template. ${RRWEB_CDN_URL} is replaced at runtime.
_RRWEB_INJECT_SCRIPT = """
(async () => {
    if (window.__seer_rrweb_loaded) return;
    window.__seer_rrweb_loaded = true;
    const script = document.createElement('script');
    script.src = '${RRWEB_CDN_URL}';
    script.onload = () => {
        rrwebRecord({
            emit(event) {
                window.__seer_rrweb_event(JSON.stringify(event));
            },
            sampling: { mousemove: false, mouseInteraction: true, scroll: 150, input: 'last' },
        });
    };
    document.head.appendChild(script);
})();
"""


class RecordingService:
    """Manages rrweb session recording via CDP injection and event collection."""

    def __init__(self) -> None:
        self._events: Dict[str, List[Dict]] = {}  # session_id -> events
        self._recording_ids: Dict[str, str] = {}   # session_id -> recording_id
        self._start_times: Dict[str, float] = {}   # session_id -> start timestamp

    async def start_recording(
        self,
        session_id: str,
        browser_session: BrowserSession,
        start_url: Optional[str] = None,  # pylint: disable=unused-argument  # Reason: Reserved for future recording metadata
    ) -> str:
        """Inject rrweb into the page and start collecting events.

        Args:
            session_id: Pool session ID
            browser_session: Active browser-use BrowserSession
            start_url: Optional URL for recording metadata

        Returns:
            Recording ID (UUID string)
        """
        recording_id = str(uuid4())
        self._events[session_id] = []
        self._recording_ids[session_id] = recording_id
        self._start_times[session_id] = time.monotonic()

        page = await browser_session.must_get_current_page()
        # browser-use stores target_id as private _target_id (no public property exposed)
        cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)  # pylint: disable=protected-access
        cdp_client = browser_session.cdp_client
        sid = cdp_session.session_id

        # Create JS binding for rrweb events
        await cdp_client.send_raw(
            "Runtime.addBinding",
            {"name": "__seer_rrweb_event"},
            session_id=sid,
        )

        # Register handler for binding calls
        def on_binding_called(params: dict, session_id_param: str = None) -> None:  # pylint: disable=unused-argument  # Reason: CDP callback signature requires session_id parameter
            if params.get("name") != "__seer_rrweb_event":
                return
            payload = params.get("payload", "")
            try:
                event = json.loads(payload)
                events = self._events.get(session_id)
                if events is not None and len(events) < config.browser_recording_max_events:
                    events.append(event)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse rrweb event: {e}")

        # browser-use cdp_client exposes _event_registry for event registration (no public API)
        cdp_client._event_registry.register(  # pylint: disable=protected-access
            "Runtime.bindingCalled", on_binding_called
        )

        # Build injection script with CDN URL
        inject_script = _RRWEB_INJECT_SCRIPT.replace(
            "${RRWEB_CDN_URL}", config.browser_recording_rrweb_cdn_url
        )

        # Inject into future navigations
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

    async def stop_recording(self, session_id: str) -> None:
        """Stop collecting events for a session.

        Args:
            session_id: Pool session ID
        """
        if session_id in self._recording_ids:
            logger.info(
                f"Recording stopped for session {session_id} "
                f"(events: {len(self._events.get(session_id, []))})"
            )

    async def save_recording(
        self,
        session_id: str,
        user: User,
        *,
        profile_id: Optional[str] = None,
        workflow_run_id: Optional[str] = None,
        session_type: str = "interactive",
        start_url: Optional[str] = None,
    ) -> Optional[str]:
        """Compress events and create SessionRecording in the database.

        Args:
            session_id: Pool session ID
            user: Recording owner
            profile_id: Optional browser profile ID
            workflow_run_id: Optional workflow run ID
            session_type: "interactive" or "workflow"
            start_url: URL the session started on

        Returns:
            Recording ID string, or None if no events were recorded
        """
        events = self._events.pop(session_id, [])
        recording_id = self._recording_ids.pop(session_id, None)
        start_time = self._start_times.pop(session_id, None)

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

    @staticmethod
    def compress_events(events: List[Dict]) -> bytes:
        """Compress rrweb events to gzip bytes."""
        return gzip.compress(json.dumps(events).encode("utf-8"))

    @staticmethod
    def decompress_events(data: bytes) -> List[Dict]:
        """Decompress gzip bytes to rrweb event list."""
        return json.loads(gzip.decompress(data))
