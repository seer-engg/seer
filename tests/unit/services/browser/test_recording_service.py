"""Tests for RecordingService - rrweb injection, JS polling, DB storage.

The recording service uses a JS storage + polling approach:
- Events are stored in window.__seer_events in JavaScript
- On save_recording(), events are retrieved via Runtime.evaluate
- This avoids CDP binding handler overwriting issues with browser_use Agent

The rrweb library is inlined (not loaded from CDN) to bypass CSP restrictions
on enterprise apps like Gmail.
"""
import asyncio
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


# ==============================================================================
# CHUNKED RECORDING TESTS - Recording Durability Feature
# ==============================================================================
#
# These tests verify the periodic flushing mechanism that prevents data loss
# when CDP target detaches during long-running browser sessions.
#
# Architecture:
# - Events stored in JS (window.__seer_events) are flushed every 45 seconds
# - Each flush creates a SessionRecordingChunk in the database
# - On finalization, remaining events become the final chunk
# - get_recording_events() reassembles chunks in sequence order


@pytest.mark.asyncio
@pytest.mark.unit
class TestFlushLoop:
    """Test the periodic event flushing mechanism."""

    @patch("seer.services.browser.recording_service.SessionRecordingChunk")
    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_flush_creates_chunk_in_db(
        self, mock_recording_model, mock_chunk_model, mock_browser_session, mock_user
    ):
        """Test that flushing events creates a chunk in the database."""
        mock_recording_model.create = AsyncMock()
        mock_recording_model.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
        mock_chunk_model.create = AsyncMock()

        # Configure mock to return events via flush function
        events = [{"type": 1, "timestamp": 1000}, {"type": 2, "timestamp": 2000}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording(
            "session-flush", mock_browser_session, user=mock_user
        )

        # Manually trigger a flush (simulate timer expiry)
        result = await recorder._flush_events_to_chunk("session-flush")

        assert result is True
        # First flush creates parent recording
        mock_recording_model.create.assert_called_once()
        # Then creates the chunk
        mock_chunk_model.create.assert_called_once()

        # Verify chunk data
        chunk_call = mock_chunk_model.create.call_args[1]
        assert chunk_call["event_count"] == 2
        assert chunk_call["sequence_number"] == 0

        # Stop flush loop before test cleanup
        active = recorder._active_recordings.get("session-flush")
        if active:
            active.stop_event.set()

    @patch("seer.services.browser.recording_service.SessionRecordingChunk")
    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_flush_increments_sequence_number(
        self, mock_recording_model, mock_chunk_model, mock_browser_session, mock_user
    ):
        """Test that successive flushes increment chunk sequence number."""
        mock_recording_model.create = AsyncMock()
        mock_recording_model.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
        mock_chunk_model.create = AsyncMock()

        events = [{"type": 1}]
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": json.dumps(events)}
        })

        recorder = RecordingService()
        await recorder.start_recording(
            "session-seq", mock_browser_session, user=mock_user
        )

        # Multiple flushes
        await recorder._flush_events_to_chunk("session-seq")
        await recorder._flush_events_to_chunk("session-seq")
        await recorder._flush_events_to_chunk("session-seq")

        # Check sequence numbers
        calls = mock_chunk_model.create.call_args_list
        assert calls[0][1]["sequence_number"] == 0
        assert calls[1][1]["sequence_number"] == 1
        assert calls[2][1]["sequence_number"] == 2

        # Cleanup
        active = recorder._active_recordings.get("session-seq")
        if active:
            active.stop_event.set()

    async def test_flush_with_empty_events_returns_true(self, mock_browser_session, mock_user):
        """Test that flushing with no events returns True (not an error)."""
        # Configure mock to return empty events
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": "[]"}
        })

        recorder = RecordingService()
        await recorder.start_recording(
            "session-empty-flush", mock_browser_session, user=mock_user
        )

        result = await recorder._flush_events_to_chunk("session-empty-flush")

        assert result is True  # Empty is not an error

        # Cleanup
        active = recorder._active_recordings.get("session-empty-flush")
        if active:
            active.stop_event.set()

    async def test_flush_continues_on_cdp_error(self, mock_browser_session, mock_user):
        """Test that flush handles CDP errors gracefully and doesn't crash.

        When CDP collection fails, _collect_and_clear_events catches the exception
        and returns an empty list. _flush_events_to_chunk then sees "no events"
        and returns True (success - nothing to flush is fine). This is by design:
        the flush loop should continue running and try again next interval.
        """
        # Simulate CDP error
        mock_browser_session.cdp_client.send_raw = AsyncMock(
            side_effect=RuntimeError("CDP connection lost")
        )

        recorder = RecordingService()
        await recorder.start_recording(
            "session-cdp-err", mock_browser_session, user=mock_user
        )

        # Should not raise - CDP errors are caught and treated as "no events"
        result = await recorder._flush_events_to_chunk("session-cdp-err")

        # Returns True because empty events = success (will retry next interval)
        assert result is True

        # Cleanup
        active = recorder._active_recordings.get("session-cdp-err")
        if active:
            active.stop_event.set()

    async def test_stop_event_terminates_flush_loop(self, mock_browser_session, mock_user):
        """Test that setting stop_event terminates the flush loop."""
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": "[]"}
        })

        recorder = RecordingService()
        await recorder.start_recording(
            "session-stop-loop", mock_browser_session, user=mock_user
        )

        active = recorder._active_recordings.get("session-stop-loop")
        assert active is not None
        assert active.flush_task is not None
        assert not active.stop_event.is_set()

        # Set stop event
        active.stop_event.set()

        # Wait for task to complete
        await asyncio.wait_for(active.flush_task, timeout=1.0)

        assert active.flush_task.done()


@pytest.mark.asyncio
@pytest.mark.unit
class TestChunkedRecordingFinalization:
    """Test finalization of chunked recordings."""

    @patch("seer.services.browser.recording_service.SessionRecordingChunk")
    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_finalize_saves_remaining_events_as_final_chunk(
        self, mock_recording_model, mock_chunk_model, mock_browser_session, mock_user
    ):
        """Test that finalization saves remaining events as a final chunk."""
        mock_recording_model.create = AsyncMock()
        mock_recording_model.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
        mock_chunk_model.create = AsyncMock()
        mock_chunk_model.filter = MagicMock(return_value=MagicMock(
            annotate=MagicMock(return_value=MagicMock(
                values=AsyncMock(return_value=[{"total": 1000}])
            ))
        ))

        # Events for flush and final collection
        events_batch1 = [{"type": 1}]
        events_batch2 = [{"type": 2}, {"type": 3}]

        # Track call count to return different values
        call_count = 0

        async def mock_send_raw(method, params, **kwargs):
            nonlocal call_count
            call_count += 1
            # Flush uses __seer_flush_events
            if "__seer_flush_events" in params.get("expression", ""):
                return {"result": {"value": json.dumps(events_batch1)}}
            # URL check
            if "location.href" in params.get("expression", ""):
                return {"result": {"value": "https://example.com"}}
            # rrweb loaded check
            if "__seer_rrweb_loaded" in params.get("expression", ""):
                return {"result": {"value": True}}
            # Final event collection (uses __seer_events)
            if "__seer_events" in params.get("expression", ""):
                return {"result": {"value": json.dumps(events_batch2)}}
            return {"result": {"value": "[]"}}

        mock_browser_session.cdp_client.send_raw = mock_send_raw

        recorder = RecordingService()
        await recorder.start_recording(
            "session-final", mock_browser_session, user=mock_user
        )

        # Flush once to create a chunk
        await recorder._flush_events_to_chunk("session-final")

        # Now finalize (save_recording)
        result = await recorder.save_recording("session-final", mock_user)

        assert result is not None
        # Should have 2 chunks: initial flush + final
        assert mock_chunk_model.create.call_count == 2

    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_short_session_creates_non_chunked_recording(
        self, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test that short sessions (no flushes) create traditional non-chunked recordings."""
        mock_recording_model.create = AsyncMock()

        # Only events at final collection (no prior flushes)
        events = [{"type": 1, "timestamp": 1000}]

        async def mock_send_raw(method, params, **kwargs):
            # URL check
            if "location.href" in params.get("expression", ""):
                return {"result": {"value": "https://example.com"}}
            # rrweb loaded check
            if "__seer_rrweb_loaded" in params.get("expression", ""):
                return {"result": {"value": True}}
            # Event collection
            if "__seer_events" in params.get("expression", ""):
                return {"result": {"value": json.dumps(events)}}
            return {"result": {"value": "[]"}}

        mock_browser_session.cdp_client.send_raw = mock_send_raw

        recorder = RecordingService()
        await recorder.start_recording(
            "session-short", mock_browser_session, user=mock_user
        )

        # Immediately finalize (no flush triggered)
        result = await recorder.save_recording("session-short", mock_user)

        assert result is not None
        # Should create a single non-chunked recording
        mock_recording_model.create.assert_called_once()
        call_kwargs = mock_recording_model.create.call_args[1]
        assert call_kwargs["is_chunked"] is False
        assert call_kwargs["events_compressed"] is not None


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetRecordingEvents:
    """Test the static get_recording_events method for reassembly."""

    @patch("seer.services.browser.recording_service.SessionRecordingChunk")
    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_get_chunked_recording_reassembles_in_order(
        self, mock_recording_model, mock_chunk_model
    ):
        """Test that chunked recordings are reassembled in sequence order."""
        # Mock parent recording
        mock_recording = MagicMock()
        mock_recording.is_chunked = True
        mock_recording_model.get_or_none = AsyncMock(return_value=mock_recording)

        # Create mock chunks with events
        events1 = [{"type": 1, "timestamp": 1000}]
        events2 = [{"type": 2, "timestamp": 2000}]
        events3 = [{"type": 3, "timestamp": 3000}]

        chunk1 = MagicMock()
        chunk1.events_compressed = RecordingService.compress_events(events1)
        chunk2 = MagicMock()
        chunk2.events_compressed = RecordingService.compress_events(events2)
        chunk3 = MagicMock()
        chunk3.events_compressed = RecordingService.compress_events(events3)

        # Mock chunk query (already ordered by sequence_number)
        mock_chunk_model.filter = MagicMock(return_value=MagicMock(
            order_by=MagicMock(return_value=MagicMock(
                all=AsyncMock(return_value=[chunk1, chunk2, chunk3])
            ))
        ))

        events = await RecordingService.get_recording_events("recording-123")

        assert len(events) == 3
        assert events[0]["type"] == 1
        assert events[1]["type"] == 2
        assert events[2]["type"] == 3

    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_get_non_chunked_recording_returns_blob(
        self, mock_recording_model
    ):
        """Test that non-chunked recordings return events from blob."""
        events = [{"type": 1}, {"type": 2}]

        mock_recording = MagicMock()
        mock_recording.is_chunked = False
        mock_recording.events_compressed = RecordingService.compress_events(events)
        mock_recording_model.get_or_none = AsyncMock(return_value=mock_recording)

        result = await RecordingService.get_recording_events("recording-456")

        assert len(result) == 2
        assert result == events

    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_get_missing_recording_returns_empty_list(
        self, mock_recording_model
    ):
        """Test that missing recordings return empty list."""
        mock_recording_model.get_or_none = AsyncMock(return_value=None)

        result = await RecordingService.get_recording_events("nonexistent")

        assert result == []

    @patch("seer.services.browser.recording_service.SessionRecording")
    async def test_get_non_chunked_with_no_blob_returns_empty(
        self, mock_recording_model
    ):
        """Test non-chunked recording with null events_compressed."""
        mock_recording = MagicMock()
        mock_recording.is_chunked = False
        mock_recording.events_compressed = None
        mock_recording_model.get_or_none = AsyncMock(return_value=mock_recording)

        result = await RecordingService.get_recording_events("recording-empty")

        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestActiveRecordingTracking:
    """Test ActiveRecording dataclass and tracking."""

    async def test_start_recording_with_user_creates_active_recording(
        self, mock_browser_session, mock_user
    ):
        """Test that start_recording with user param creates ActiveRecording."""
        recorder = RecordingService()

        await recorder.start_recording(
            "session-active",
            mock_browser_session,
            user=mock_user,
            profile_id="profile-123",
            workflow_run_id="run-456",
        )

        assert "session-active" in recorder._active_recordings
        active = recorder._active_recordings["session-active"]
        assert active.user is mock_user
        assert active.profile_id == "profile-123"
        assert active.workflow_run_id == "run-456"
        assert active.flush_task is not None

        # Cleanup
        active.stop_event.set()

    async def test_start_recording_without_user_uses_legacy_path(
        self, mock_browser_session
    ):
        """Test that start_recording without user uses legacy tracking only."""
        recorder = RecordingService()

        await recorder.start_recording(
            "session-legacy",
            mock_browser_session,
        )

        # Should be in legacy tracking
        assert "session-legacy" in recorder._browser_sessions
        assert "session-legacy" in recorder._recording_ids
        # But NOT in active recordings
        assert "session-legacy" not in recorder._active_recordings

    async def test_save_recording_removes_active_recording(
        self, mock_browser_session, mock_user
    ):
        """Test that save_recording removes the ActiveRecording entry."""
        mock_browser_session.cdp_client.send_raw = AsyncMock(return_value={
            "result": {"value": "[]"}
        })

        recorder = RecordingService()
        await recorder.start_recording(
            "session-remove",
            mock_browser_session,
            user=mock_user,
        )

        assert "session-remove" in recorder._active_recordings

        await recorder.save_recording("session-remove", mock_user)

        # Should be removed after save
        assert "session-remove" not in recorder._active_recordings
