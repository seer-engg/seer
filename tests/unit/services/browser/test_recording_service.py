"""Tests for RecordingService - rrweb injection, event collection, DB storage."""
import asyncio
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
    cdp_client._event_registry.unregister = MagicMock()
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

        # Should register for Page.frameNavigated and Runtime.bindingCalled events
        assert mock_browser_session.cdp_client._event_registry.register.call_count == 2
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        event_names = [call[0][0] for call in register_calls]
        assert "Page.frameNavigated" in event_names
        assert "Runtime.bindingCalled" in event_names

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


class TestSingleton:
    """Test singleton pattern."""

    async def test_get_instance_returns_same_instance(self):
        """Test that get_instance returns the same singleton instance."""
        # Reset singleton for test isolation
        RecordingService._instance = None

        instance1 = await RecordingService.get_instance()
        instance2 = await RecordingService.get_instance()

        assert instance1 is instance2

    async def test_singleton_preserves_events(self, mock_browser_session):
        """Test that events are preserved across get_instance calls."""
        # Reset singleton for test isolation
        RecordingService._instance = None

        # Start recording via one get_instance call
        recorder1 = await RecordingService.get_instance()
        await recorder1.start_recording("session-singleton", mock_browser_session)

        # Get the binding callback and inject events
        register_call = mock_browser_session.cdp_client._event_registry.register.call_args
        on_binding = register_call[0][1]
        on_binding({
            "name": "__seer_rrweb_event",
            "payload": '{"type": 1, "timestamp": 1000}',
        })

        # Get instance again and verify events are there
        recorder2 = await RecordingService.get_instance()
        assert recorder2 is recorder1
        assert len(recorder2._events.get("session-singleton", [])) == 1

        # Cleanup
        RecordingService._instance = None


class TestNavigationHandling:
    """Test recording across page navigations."""

    async def test_registers_navigation_handler(self, mock_browser_session):
        """Verify Page.frameNavigated handler is registered."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        # Should register both Page.frameNavigated and Runtime.bindingCalled handlers
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        event_names = [call[0][0] for call in register_calls]

        assert "Page.frameNavigated" in event_names
        assert "Runtime.bindingCalled" in event_names

    async def test_reregisters_binding_on_navigation(self, mock_browser_session):
        """Verify binding is re-registered after main frame navigation."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        # Initial binding registration call
        initial_call_count = mock_browser_session.cdp_client.send_raw.call_count
        assert initial_call_count == 1

        # Get the navigation handler (now sync)
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_frame_navigated = None
        for call in register_calls:
            if call[0][0] == "Page.frameNavigated":
                on_frame_navigated = call[0][1]
                break
        assert on_frame_navigated is not None

        # Simulate main frame navigation - handler is sync but fires async task
        on_frame_navigated({
            "frame": {
                "id": "main-frame-123",
                "url": "https://google.com",
            }
        })

        # Allow the fire-and-forget task to run
        await asyncio.sleep(0)

        # Binding should be re-registered
        assert mock_browser_session.cdp_client.send_raw.call_count == 2
        second_call = mock_browser_session.cdp_client.send_raw.call_args_list[1]
        assert second_call[0][0] == "Runtime.addBinding"
        assert second_call[0][1] == {"name": "__seer_rrweb_event"}

    async def test_ignores_iframe_navigation(self, mock_browser_session):
        """Verify iframe navigations don't trigger re-registration."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        initial_call_count = mock_browser_session.cdp_client.send_raw.call_count

        # Get the navigation handler (now sync)
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_frame_navigated = None
        for call in register_calls:
            if call[0][0] == "Page.frameNavigated":
                on_frame_navigated = call[0][1]
                break

        # Simulate iframe navigation (has parentId) - early return, no task created
        on_frame_navigated({
            "frame": {
                "id": "iframe-456",
                "parentId": "main-frame-123",
                "url": "https://ads.example.com",
            }
        })

        # No new binding registration
        assert mock_browser_session.cdp_client.send_raw.call_count == initial_call_count

    async def test_ignores_non_http_navigation(self, mock_browser_session):
        """Verify about:blank and chrome:// navigations don't trigger re-registration."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        initial_call_count = mock_browser_session.cdp_client.send_raw.call_count

        # Get the navigation handler (now sync)
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_frame_navigated = None
        for call in register_calls:
            if call[0][0] == "Page.frameNavigated":
                on_frame_navigated = call[0][1]
                break

        # Simulate about:blank navigation - early return, no task created
        on_frame_navigated({
            "frame": {
                "id": "main-frame-123",
                "url": "about:blank",
            }
        })

        # No new binding registration
        assert mock_browser_session.cdp_client.send_raw.call_count == initial_call_count

    async def test_events_captured_after_navigation(self, mock_browser_session):
        """Verify events accumulate across multiple navigations."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        # Get both handlers (on_frame_navigated is now sync)
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_frame_navigated = None
        on_binding_called = None
        for call in register_calls:
            if call[0][0] == "Page.frameNavigated":
                on_frame_navigated = call[0][1]
            elif call[0][0] == "Runtime.bindingCalled":
                on_binding_called = call[0][1]

        # Event before navigation
        on_binding_called({
            "name": "__seer_rrweb_event",
            "payload": json.dumps({"type": 1, "timestamp": 1000}),
        })

        # Navigate (sync handler with fire-and-forget)
        on_frame_navigated({
            "frame": {"id": "main", "url": "https://google.com"}
        })
        await asyncio.sleep(0)  # Let task run

        # Event after navigation
        on_binding_called({
            "name": "__seer_rrweb_event",
            "payload": json.dumps({"type": 2, "timestamp": 2000}),
        })

        # Navigate again
        on_frame_navigated({
            "frame": {"id": "main", "url": "https://github.com"}
        })
        await asyncio.sleep(0)  # Let task run

        # Event after second navigation
        on_binding_called({
            "name": "__seer_rrweb_event",
            "payload": json.dumps({"type": 3, "timestamp": 3000}),
        })

        # All events should be accumulated
        assert len(recorder._events["session-nav"]) == 3

    async def test_stop_recording_unregisters_handlers(self, mock_browser_session):
        """Verify cleanup removes both handlers."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        await recorder.stop_recording("session-nav")

        # Should call unregister for both handlers
        unregister_calls = mock_browser_session.cdp_client._event_registry.unregister.call_args_list
        unregistered_events = [call[0][0] for call in unregister_calls]

        assert "Page.frameNavigated" in unregistered_events
        assert "Runtime.bindingCalled" in unregistered_events

        # CDP references should be cleaned up
        assert "session-nav" not in recorder._cdp_sessions
        assert "session-nav" not in recorder._cdp_clients

    async def test_stores_cdp_references(self, mock_browser_session):
        """Verify CDP session and client references are stored."""
        recorder = RecordingService()
        await recorder.start_recording("session-nav", mock_browser_session)

        assert "session-nav" in recorder._cdp_sessions
        assert recorder._cdp_sessions["session-nav"] == "cdp-session-456"
        assert "session-nav" in recorder._cdp_clients
        assert recorder._cdp_clients["session-nav"] == mock_browser_session.cdp_client


class TestRegisterBindingErrors:
    """Test _register_binding error handling."""

    async def test_register_binding_no_cdp_client(self):
        """Test _register_binding returns early when no CDP client."""
        recorder = RecordingService()
        # Don't start recording, so no CDP client

        # Should not raise
        await recorder._register_binding("nonexistent-session")

    async def test_register_binding_no_cdp_session(self, mock_browser_session):
        """Test _register_binding returns early when no CDP session ID."""
        recorder = RecordingService()
        recorder._cdp_clients["session-1"] = mock_browser_session.cdp_client
        # Don't set _cdp_sessions

        # Should not raise
        await recorder._register_binding("session-1")

    async def test_register_binding_exception_handled(self, mock_browser_session):
        """Test _register_binding handles CDP exceptions gracefully."""
        mock_browser_session.cdp_client.send_raw = AsyncMock(
            side_effect=RuntimeError("CDP error")
        )

        recorder = RecordingService()
        recorder._cdp_clients["session-1"] = mock_browser_session.cdp_client
        recorder._cdp_sessions["session-1"] = "cdp-sess-123"

        # Should not raise
        await recorder._register_binding("session-1")


class TestBindingCallbackJsonErrors:
    """Test JSON parse errors in binding callback."""

    async def test_invalid_json_payload(self, mock_browser_session):
        """Test that invalid JSON payloads are handled gracefully."""
        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Get the binding callback (second registered handler)
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_binding_called = None
        for call in register_calls:
            if call[0][0] == "Runtime.bindingCalled":
                on_binding_called = call[0][1]
                break

        # Send invalid JSON
        on_binding_called({
            "name": "__seer_rrweb_event",
            "payload": "not valid json {{{",
        })

        # Should not crash, events list should be empty
        assert len(recorder._events.get("session-1", [])) == 0

    async def test_wrong_binding_name_ignored(self, mock_browser_session):
        """Test that callbacks with wrong binding name are ignored."""
        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_binding_called = None
        for call in register_calls:
            if call[0][0] == "Runtime.bindingCalled":
                on_binding_called = call[0][1]
                break

        # Send event with wrong binding name
        on_binding_called({
            "name": "__some_other_binding",
            "payload": '{"type": 1}',
        })

        # Should be ignored
        assert len(recorder._events.get("session-1", [])) == 0


class TestStopRecordingCleanup:
    """Test stop_recording cleanup error handling."""

    async def test_stop_cleanup_errors_ignored(self, mock_browser_session):
        """Test that cleanup errors during stop are ignored."""
        mock_browser_session.cdp_client._event_registry.unregister = MagicMock(
            side_effect=RuntimeError("Already closed")
        )

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Should not raise
        await recorder.stop_recording("session-1")

        # CDP references should still be cleaned up
        assert "session-1" not in recorder._cdp_sessions
        assert "session-1" not in recorder._cdp_clients

    async def test_stop_nonexistent_session(self):
        """Test stopping a nonexistent session is a no-op."""
        recorder = RecordingService()

        # Should not raise
        await recorder.stop_recording("nonexistent-session")


class TestSaveRecordingTruncation:
    """Test save_recording truncation for oversized data."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_truncates_oversized_recording(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test that oversized recordings are truncated."""
        mock_config.browser_recording_max_events = 1000
        mock_config.browser_recording_max_size_mb = 0.0001  # Very small limit
        mock_config.browser_recording_rrweb_cdn_url = "https://cdn.test/rrweb.js"
        mock_recording_model.create = AsyncMock()

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Get binding callback
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_binding = None
        for call in register_calls:
            if call[0][0] == "Runtime.bindingCalled":
                on_binding = call[0][1]
                break

        # Add many events to exceed size limit
        for i in range(100):
            on_binding({
                "name": "__seer_rrweb_event",
                "payload": json.dumps({
                    "type": i,
                    "data": "x" * 100,  # Some padding
                    "timestamp": i * 1000,
                }),
            })

        original_count = len(recorder._events["session-1"])

        result = await recorder.save_recording(
            "session-1",
            mock_user,
            session_type="interactive",
        )

        # Should still save something
        assert result is not None
        mock_recording_model.create.assert_called_once()

        # Event count in DB should be less than original (truncated)
        call_kwargs = mock_recording_model.create.call_args[1]
        assert call_kwargs["event_count"] < original_count


class TestSaveRecordingDbErrors:
    """Test save_recording database error handling."""

    @patch("seer.services.browser.recording_service.SessionRecording")
    @patch("seer.services.browser.recording_service.config")
    async def test_db_error_returns_none(
        self, mock_config, mock_recording_model, mock_browser_session, mock_user
    ):
        """Test that database errors cause None return."""
        mock_config.browser_recording_max_events = 50000
        mock_config.browser_recording_max_size_mb = 50
        mock_config.browser_recording_rrweb_cdn_url = "https://cdn.test/rrweb.js"
        mock_recording_model.create = AsyncMock(
            side_effect=RuntimeError("Database connection failed")
        )

        recorder = RecordingService()
        await recorder.start_recording("session-1", mock_browser_session)

        # Add an event
        register_calls = mock_browser_session.cdp_client._event_registry.register.call_args_list
        on_binding = None
        for call in register_calls:
            if call[0][0] == "Runtime.bindingCalled":
                on_binding = call[0][1]
                break

        on_binding({
            "name": "__seer_rrweb_event",
            "payload": '{"type": 1}',
        })

        # Should return None on DB error
        result = await recorder.save_recording(
            "session-1",
            mock_user,
            session_type="interactive",
        )

        assert result is None


class TestSaveRecordingNoRecordingId:
    """Test save_recording when recording_id is missing."""

    async def test_no_recording_id_returns_none(self, mock_user):
        """Test that save_recording returns None when no recording_id."""
        recorder = RecordingService()
        # Add events directly without start_recording (no recording_id)
        recorder._events["session-1"] = [{"type": 1}]
        # Don't set _recording_ids["session-1"]

        result = await recorder.save_recording("session-1", mock_user)

        assert result is None
