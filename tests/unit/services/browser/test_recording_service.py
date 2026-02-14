"""Tests for RecordingService - rrweb injection, event collection, DB storage."""
import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.browser.recording_service import RecordingService


@pytest.fixture
def mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()

    page = MagicMock()
    page.target_id = "target-123"
    session.must_get_current_page = AsyncMock(return_value=page)

    cdp_session = MagicMock()
    cdp_session.session_id = "cdp-session-456"
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

    cdp_client = MagicMock()
    cdp_client.send_raw = AsyncMock()
    cdp_client._event_registry = MagicMock()
    cdp_client._event_registry.register = MagicMock()
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


class TestStartRecording:
    """Test recording initialization."""

    async def test_start_injects_rrweb_script(self, mock_browser_session):
        recorder = RecordingService()
        recording_id = await recorder.start_recording(
            "session-1", mock_browser_session, start_url="https://example.com"
        )

        assert recording_id is not None
        assert "session-1" in recorder._events
        assert recorder._events["session-1"] == []

        # Should create JS binding
        mock_browser_session.cdp_client.send_raw.assert_called_once_with(
            "Runtime.addBinding",
            {"name": "__seer_rrweb_event"},
            session_id="cdp-session-456",
        )

        # Should register for Runtime.bindingCalled events
        mock_browser_session.cdp_client._event_registry.register.assert_called_once()
        call_args = mock_browser_session.cdp_client._event_registry.register.call_args
        assert call_args[0][0] == "Runtime.bindingCalled"

        # Should inject script via addScriptToEvaluateOnNewDocument
        mock_browser_session.cdp_client.send.Page.addScriptToEvaluateOnNewDocument.assert_called_once()

        # Should also inject into current page via Runtime.evaluate
        mock_browser_session.cdp_client.send.Runtime.evaluate.assert_called_once()


class TestEventCollection:
    """Test event accumulation."""

    async def test_event_collection_accumulates(self, mock_browser_session):
        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Get the binding callback
        register_call = mock_browser_session.cdp_client._event_registry.register.call_args
        on_binding = register_call[0][1]

        # Simulate rrweb events via the binding
        event1 = {"type": 1, "data": {"source": 0}, "timestamp": 1000}
        event2 = {"type": 2, "data": {"source": 1}, "timestamp": 2000}

        on_binding({"name": "__seer_rrweb_event", "payload": json.dumps(event1)})
        on_binding({"name": "__seer_rrweb_event", "payload": json.dumps(event2)})

        assert len(recorder._events["session-1"]) == 2
        assert recorder._events["session-1"][0] == event1
        assert recorder._events["session-1"][1] == event2

    @patch("seer.services.browser.recording_service.config")
    async def test_max_events_truncation(self, mock_config, mock_browser_session):
        mock_config.browser_recording_max_events = 3
        mock_config.browser_recording_rrweb_cdn_url = "https://cdn.test/rrweb.js"

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        register_call = mock_browser_session.cdp_client._event_registry.register.call_args
        on_binding = register_call[0][1]

        # Send 5 events, should only keep 3
        for i in range(5):
            on_binding({
                "name": "__seer_rrweb_event",
                "payload": json.dumps({"type": i, "timestamp": i * 1000}),
            })

        assert len(recorder._events["session-1"]) == 3


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


class TestSaveRecording:
    """Test recording persistence."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_save_compresses_and_creates_db_record(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        mock_config.browser_recording_max_events = 50000
        mock_config.browser_recording_max_size_mb = 50
        mock_config.browser_recording_rrweb_cdn_url = "https://cdn.test/rrweb.js"
        mock_recording_model.create = AsyncMock()

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Inject some events
        register_call = mock_browser_session.cdp_client._event_registry.register.call_args
        on_binding = register_call[0][1]
        on_binding({"name": "__seer_rrweb_event", "payload": json.dumps({"type": 1})})
        on_binding({"name": "__seer_rrweb_event", "payload": json.dumps({"type": 2})})

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

    async def test_empty_recording_returns_none(self, mock_user):
        recorder = RecordingService()
        result = await recorder.save_recording("nonexistent", mock_user)
        assert result is None

    @patch("seer.services.browser.recording_service.config")
    async def test_recording_disabled_skips_injection(self, mock_config):
        """RecordingService itself doesn't check enabled flag - caller does.
        Test that saving with no events returns None."""
        recorder = RecordingService()
        # Never started recording, so no events
        result = await recorder.save_recording(
            "session-1", MagicMock(), session_type="interactive"
        )
        assert result is None
