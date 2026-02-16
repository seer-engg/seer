"""Tests for RecordingService - rrweb injection, JS polling, DB storage.

The recording service uses a JS storage + polling approach:
- Events are stored in window.__seer_events in JavaScript
- On save_recording(), events are retrieved via Runtime.evaluate
- This avoids CDP binding handler overwriting issues with browser_use Agent

The rrweb library is inlined (not loaded from CDN) to bypass CSP restrictions
on enterprise apps like Gmail.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.browser.recording_service import RecordingService


@pytest.fixture(autouse=True)
def mock_rrweb_library():
    """Mock the inlined rrweb library loader for all tests.

    The real rrweb library is ~68KB of minified JavaScript bundled in
    src/seer/services/browser/rrweb-record.min.js. We mock it to avoid
    loading the actual file in tests and to keep test output clean.
    """
    with patch(
        "seer.services.browser.recording_service._get_rrweb_library",
        return_value="// mock rrweb library code",
    ):
        yield


@pytest.fixture
def mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()

    page = MagicMock()
    page._target_id = "target-123"  # browser-use uses _target_id (private)
    session.must_get_current_page = AsyncMock(return_value=page)

    cdp_session = MagicMock()
    cdp_session.session_id = "cdp-session-456"
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

    cdp_client = MagicMock()
    cdp_client.send_raw = AsyncMock(return_value={
        "result": {"value": "[]"}
    })
    cdp_client.send = MagicMock()
    cdp_client.send.Page = MagicMock()
    cdp_client.send.Page.addScriptToEvaluateOnNewDocument = AsyncMock()
    cdp_client.send.Runtime = MagicMock()
    cdp_client.send.Runtime.evaluate = AsyncMock()

    session.cdp_client = cdp_client
    return session


@pytest.fixture
def mock_user():
    """Create a mock User object."""
    user = MagicMock()
    user.user_id = "user-123"
    user.id = "user-123"
    return user


@pytest.mark.asyncio
@pytest.mark.unit
class TestStartRecording:
    """Test recording initialization."""

    async def test_start_injects_rrweb_script(self, mock_browser_session):
        """Test that start_recording injects rrweb and stores browser_session."""
        recorder = RecordingService()
        recording_id = await recorder.start_recording(
            "session-1", mock_browser_session, start_url="https://example.com"
        )

        assert recording_id is not None
        assert "session-1" in recorder._browser_sessions
        assert recorder._browser_sessions["session-1"] is mock_browser_session
        assert "session-1" in recorder._recording_ids
        assert "session-1" in recorder._start_times

        # Should inject script via addScriptToEvaluateOnNewDocument (for future navigations)
        mock_browser_session.cdp_client.send.Page.addScriptToEvaluateOnNewDocument.assert_called_once()

        # Should also inject into current page via Runtime.evaluate
        mock_browser_session.cdp_client.send.Runtime.evaluate.assert_called_once()

    async def test_start_recording_stores_session_reference(self, mock_browser_session):
        """Test that browser_session reference is stored for later polling."""
        recorder = RecordingService()
        await recorder.start_recording("session-ref", mock_browser_session)

        assert recorder._browser_sessions.get("session-ref") is mock_browser_session


@pytest.mark.asyncio
@pytest.mark.unit
class TestEventCollection:
    """Test event polling from JavaScript."""

    async def test_collect_events_polls_javascript(self, mock_browser_session):
        """Test that _collect_events_from_page polls window.__seer_events."""
        # Configure mock to return events
        events = [{"type": 1, "timestamp": 1000}, {"type": 2, "timestamp": 2000}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        collected = await recorder._collect_events_from_page("session-1")

        assert len(collected) == 2
        assert collected[0] == events[0]
        assert collected[1] == events[1]

        # Verify the correct Runtime.evaluate call was made
        mock_browser_session.cdp_client.send_raw.assert_called_with(
            "Runtime.evaluate",
            {
                "expression": "JSON.stringify(window.__seer_events || [])",
                "returnByValue": True,
            },
            session_id="cdp-session-456",
        )

    async def test_collect_events_no_browser_session(self):
        """Test _collect_events_from_page returns empty when no browser_session."""
        recorder = RecordingService()
        # Don't start recording, so no browser_session reference

        collected = await recorder._collect_events_from_page("nonexistent")

        assert collected == []

    async def test_collect_events_handles_exception(self, mock_browser_session):
        """Test _collect_events_from_page handles CDP errors gracefully."""
        mock_browser_session.cdp_client.send_raw = AsyncMock(
            side_effect=RuntimeError("CDP connection lost")
        )

        recorder = RecordingService()
        await recorder.start_recording("session-err", mock_browser_session)

        collected = await recorder._collect_events_from_page("session-err")

        assert collected == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestCompression:
    """Test compress/decompress roundtrip."""

    def test_compress_decompress_roundtrip(self):
        events = [
            {"type": 1, "data": {"source": 0}, "timestamp": 1000},
            {"type": 2, "data": {"node": {"id": 1}}, "timestamp": 2000},
        ]

        compressed = RecordingService.compress_events(events)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decompressed = RecordingService.decompress_events(compressed)
        assert decompressed == events


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveRecording:
    """Test recording persistence."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_save_polls_and_creates_db_record(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test save_recording polls JS, compresses, and creates DB record."""
        mock_config.browser_recording_max_size_mb = 50
        mock_recording_model.create = AsyncMock()

        # Configure mock to return events from JS
        events = [{"type": 1, "timestamp": 1000}, {"type": 2, "timestamp": 2000}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        result = await recorder.save_recording(
            "session-1",
            mock_user,
            profile_id="profile-abc",
            session_type="interactive",
            start_url="https://example.com",
        )

        assert result is not None
        mock_recording_model.create.assert_called_once()
        call_kwargs = mock_recording_model.create.call_args[1]
        assert call_kwargs["event_count"] == 2
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["session_type"] == "interactive"
        assert call_kwargs["start_url"] == "https://example.com"

        # Verify compressed data is valid
        decompressed = RecordingService.decompress_events(call_kwargs["events_compressed"])
        assert len(decompressed) == 2

    async def test_empty_recording_returns_none(self, mock_user, mock_browser_session):
        """Test save_recording returns None when no events collected."""
        # Configure mock to return empty events
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": "[]"}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-empty", mock_browser_session)

        result = await recorder.save_recording("session-empty", mock_user)

        assert result is None

    async def test_save_nonexistent_session_returns_none(self, mock_user):
        """Test save_recording returns None for unknown session."""
        recorder = RecordingService()
        result = await recorder.save_recording("nonexistent", mock_user)
        assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestSingleton:
    """Test singleton pattern."""

    async def test_get_instance_returns_same_instance(self):
        """Test that get_instance returns the same singleton instance."""
        # Reset singleton for test isolation
        RecordingService._instance = None

        instance1 = await RecordingService.get_instance()
        instance2 = await RecordingService.get_instance()

        assert instance1 is instance2

        # Cleanup
        RecordingService._instance = None

    async def test_singleton_preserves_browser_sessions(self, mock_browser_session):
        """Test that browser_sessions are preserved across get_instance calls."""
        # Reset singleton for test isolation
        RecordingService._instance = None

        # Start recording via one get_instance call
        recorder1 = await RecordingService.get_instance()
        await recorder1.start_recording("session-singleton", mock_browser_session)

        # Get instance again and verify reference is there
        recorder2 = await RecordingService.get_instance()
        assert recorder2 is recorder1
        assert "session-singleton" in recorder2._browser_sessions

        # Cleanup
        RecordingService._instance = None


@pytest.mark.asyncio
@pytest.mark.unit
class TestStopRecording:
    """Test stop_recording functionality."""

    async def test_stop_recording_logs_event_count(self, mock_browser_session):
        """Test stop_recording collects and logs event count."""
        events = [{"type": 1}, {"type": 2}, {"type": 3}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-stop", mock_browser_session)

        # stop_recording should not raise
        await recorder.stop_recording("session-stop")

    async def test_stop_nonexistent_session(self):
        """Test stopping a nonexistent session is a no-op."""
        recorder = RecordingService()

        # Should not raise
        await recorder.stop_recording("nonexistent-session")


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveRecordingTruncation:
    """Test save_recording truncation for oversized data."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_truncates_oversized_recording(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test that oversized recordings are truncated."""
        mock_config.browser_recording_max_size_mb = 0.0001  # Very small limit
        mock_recording_model.create = AsyncMock()

        # Generate many events that exceed the size limit
        events = [{"type": i, "data": "x" * 100, "timestamp": i * 1000} for i in range(100)]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-truncate", mock_browser_session)

        result = await recorder.save_recording(
            "session-truncate",
            mock_user,
            session_type="interactive",
        )

        # Should still save something
        assert result is not None
        mock_recording_model.create.assert_called_once()

        # Event count in DB should be less than original (truncated to ~75%)
        call_kwargs = mock_recording_model.create.call_args[1]
        assert call_kwargs["event_count"] < 100
        assert call_kwargs["event_count"] == 75  # 75% of 100


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveRecordingDbErrors:
    """Test save_recording database error handling."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_db_error_returns_none(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test that database errors cause None return."""
        mock_config.browser_recording_max_size_mb = 50
        mock_recording_model.create = AsyncMock(
            side_effect=RuntimeError("Database connection failed")
        )

        events = [{"type": 1, "timestamp": 1000}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-db-err", mock_browser_session)

        # Should return None on DB error
        result = await recorder.save_recording(
            "session-db-err",
            mock_user,
            session_type="interactive",
        )

        assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveRecordingNoRecordingId:
    """Test save_recording when recording_id is missing."""

    async def test_no_recording_id_returns_none(self, mock_user, mock_browser_session):
        """Test that save_recording returns None when no recording_id (no start)."""
        recorder = RecordingService()
        # Store browser_session but no recording_id
        recorder._browser_sessions["session-no-id"] = mock_browser_session

        # Configure mock to return events
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps([{"type": 1}])}
        })

        result = await recorder.save_recording("session-no-id", mock_user)

        # Returns None because recording_id is missing
        assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveRecordingCleanup:
    """Test that save_recording cleans up session references."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_save_cleans_up_session_refs(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test save_recording removes session references after saving."""
        mock_config.browser_recording_max_size_mb = 50
        mock_recording_model.create = AsyncMock()

        events = [{"type": 1}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording("session-cleanup", mock_browser_session)

        # Verify references exist before save
        assert "session-cleanup" in recorder._browser_sessions
        assert "session-cleanup" in recorder._recording_ids
        assert "session-cleanup" in recorder._start_times

        await recorder.save_recording("session-cleanup", mock_user)

        # References should be cleaned up after save
        assert "session-cleanup" not in recorder._browser_sessions
        assert "session-cleanup" not in recorder._recording_ids
        assert "session-cleanup" not in recorder._start_times
