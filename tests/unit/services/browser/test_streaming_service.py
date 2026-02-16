"""Tests for StreamingService - CDP screencast and input dispatch."""
import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from seer.services.browser.streaming_service import StreamingService


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG image (10x10 red) for testing.

    The streaming service parses JPEG frames to extract dimensions,
    so we need actual valid JPEG data, not fake bytes.
    """
    img = Image.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()

    # Mock page
    page = MagicMock()
    page.target_id = "target-123"
    session.must_get_current_page = AsyncMock(return_value=page)

    # Mock CDP session
    cdp_session = MagicMock()
    cdp_session.session_id = "cdp-session-456"
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

    # Mock CDP client
    cdp_client = MagicMock()
    cdp_client.send_raw = AsyncMock()
    cdp_client._event_registry = MagicMock()
    cdp_client._event_registry.register = MagicMock()

    # Mock typed CDP commands
    cdp_client.send = MagicMock()
    cdp_client.send.Input = MagicMock()
    cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
    cdp_client.send.Input.dispatchKeyEvent = AsyncMock()

    session.cdp_client = cdp_client
    return session


class TestStart:
    """Test screencast start."""

    async def test_start_registers_cdp_handler_and_sends_screencast(self, mock_browser_session):
        streamer = StreamingService(quality=80, max_width=1920, max_height=1080)
        await streamer.start(mock_browser_session)

        # Should register for Page.screencastFrame events
        mock_browser_session.cdp_client._event_registry.register.assert_called_once_with(
            "Page.screencastFrame", streamer._on_frame
        )

        # Should send Page.startScreencast via raw CDP
        mock_browser_session.cdp_client.send_raw.assert_called_once_with(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 80,
                "maxWidth": 1920,
                "maxHeight": 1080,
                "everyNthFrame": 1,
            },
            session_id="cdp-session-456",
        )

        assert streamer.is_running is True
        assert streamer.frame_count == 0


class TestFrameCallback:
    """Test frame handling."""

    async def test_frame_callback_queues_decoded_frames(self, mock_browser_session, valid_jpeg_bytes):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Simulate a screencast frame with valid JPEG data
        b64_data = base64.b64encode(valid_jpeg_bytes).decode("ascii")

        streamer._on_frame({"data": b64_data, "sessionId": 1})

        assert streamer.frame_count == 1
        frame = await asyncio.wait_for(streamer._frame_queue.get(), timeout=1.0)
        assert frame == valid_jpeg_bytes

        # Verify JPEG dimensions were extracted (10x10 from fixture)
        assert streamer._actual_screencast_width == 10.0
        assert streamer._actual_screencast_height == 10.0

    async def test_frame_ack_sent_on_receive(self, mock_browser_session, valid_jpeg_bytes):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Reset send_raw call count after start
        mock_browser_session.cdp_client.send_raw.reset_mock()

        b64_data = base64.b64encode(valid_jpeg_bytes).decode("ascii")
        streamer._on_frame({"data": b64_data, "sessionId": 42})

        # Give the ack task a moment to run
        await asyncio.sleep(0.05)

        # Should have sent a Page.screencastFrameAck
        mock_browser_session.cdp_client.send_raw.assert_called_with(
            "Page.screencastFrameAck",
            {"sessionId": 42},
            session_id="cdp-session-456",
        )

    async def test_get_frame_timeout_returns_none(self, mock_browser_session):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        result = await streamer.get_frame(timeout=0.05)
        assert result is None


class TestCoordinateScaling:
    """Test coordinate scaling from screencast to viewport space."""

    def test_scale_coordinates_uses_jpeg_dimensions(self):
        """Verify scaling uses actual JPEG frame dimensions, not CDP metadata.

        This test validates the fix for the bug where CDP metadata.deviceWidth/deviceHeight
        reported viewport size (e.g., 1920x1080) instead of actual frame size (e.g., 1280x720),
        causing clicks to land at wrong positions.
        """
        streamer = StreamingService(max_width=1280, max_height=800)

        # Simulate what happens after receiving a frame:
        # - Viewport is 1920x1080 (browser window size)
        # - Screencast max is 1280x800
        # - Actual JPEG frame is 1280x720 (scaled to fit, maintaining 16:9 aspect)
        streamer._viewport_width = 1920.0
        streamer._viewport_height = 1080.0
        streamer._actual_screencast_width = 1280.0  # From JPEG parsing
        streamer._actual_screencast_height = 720.0  # From JPEG parsing

        # Click at (640, 360) in screencast space (center of 1280x720 frame)
        # Should scale to (960, 540) in viewport space (center of 1920x1080)
        scaled_x, scaled_y = streamer._scale_coordinates(640, 360)

        # Scale factors: 1920/1280 = 1.5, 1080/720 = 1.5
        assert scaled_x == 960.0  # 640 * 1.5
        assert scaled_y == 540.0  # 360 * 1.5

    def test_scale_coordinates_without_frame_uses_max_bounds(self):
        """Before receiving any frame, use max bounds as fallback."""
        streamer = StreamingService(max_width=1280, max_height=800)
        streamer._viewport_width = 1920.0
        streamer._viewport_height = 1200.0
        # _actual_screencast_width/height are 0 (no frame received yet)

        scaled_x, scaled_y = streamer._scale_coordinates(640, 400)

        # Fallback to max bounds: 1920/1280 = 1.5, 1200/800 = 1.5
        assert scaled_x == 960.0
        assert scaled_y == 600.0


class TestInputDispatch:
    """Test input event dispatch."""

    async def test_dispatch_mouse_event_calls_cdp(self, mock_browser_session):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        await streamer.dispatch_mouse_event("mousePressed", 100, 200, button="left", click_count=1)

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_called_once_with(
            params={
                "type": "mousePressed",
                "x": 100,
                "y": 200,
                "button": "left",
                "clickCount": 1,
            },
            session_id="cdp-session-456",
        )

    async def test_dispatch_key_event_calls_cdp(self, mock_browser_session):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        await streamer.dispatch_key_event("keyDown", "Enter", code="Enter", text="\r", modifiers=0)

        mock_browser_session.cdp_client.send.Input.dispatchKeyEvent.assert_called_once_with(
            params={
                "type": "keyDown",
                "key": "Enter",
                "modifiers": 0,
                "code": "Enter",
                "text": "\r",
            },
            session_id="cdp-session-456",
        )

    async def test_dispatch_scroll_event_calls_cdp(self, mock_browser_session):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        await streamer.dispatch_scroll_event(640, 400, 0, -120)

        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent.assert_called_once_with(
            params={
                "type": "mouseWheel",
                "x": 640,
                "y": 400,
                "deltaX": 0,
                "deltaY": -120,
            },
            session_id="cdp-session-456",
        )


class TestStop:
    """Test screencast stop."""

    async def test_stop_sends_stop_screencast(self, mock_browser_session):
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        mock_browser_session.cdp_client.send_raw.reset_mock()
        await streamer.stop()

        mock_browser_session.cdp_client.send_raw.assert_called_once_with(
            "Page.stopScreencast",
            {},
            session_id="cdp-session-456",
        )
        assert streamer.is_running is False

    async def test_stop_when_not_running_is_noop(self):
        streamer = StreamingService()
        await streamer.stop()  # Should not raise
        assert streamer.is_running is False

    async def test_stop_exception_handling(self, mock_browser_session):
        """Test that stop handles CDP exceptions gracefully."""
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Make stop screencast fail
        mock_browser_session.cdp_client.send_raw.side_effect = RuntimeError("CDP disconnected")

        # Should not raise
        await streamer.stop()
        assert streamer.is_running is False


class TestViewportParsing:
    """Test viewport JSON string parsing edge case."""

    async def test_viewport_json_string_parsing(self, mock_browser_session):
        """Test that viewport returned as JSON string is handled."""
        page = mock_browser_session.must_get_current_page.return_value

        # Return viewport as JSON string instead of dict
        page.evaluate = AsyncMock(return_value='{"width": 1920, "height": 1080}')

        streamer = StreamingService(max_width=1280, max_height=800)
        await streamer.start(mock_browser_session)

        assert streamer._viewport_width == 1920.0
        assert streamer._viewport_height == 1080.0

    async def test_viewport_dict_parsing(self, mock_browser_session):
        """Test normal viewport dict handling."""
        page = mock_browser_session.must_get_current_page.return_value
        page.evaluate = AsyncMock(return_value={"width": 1600, "height": 900})

        streamer = StreamingService(max_width=1280, max_height=800)
        await streamer.start(mock_browser_session)

        assert streamer._viewport_width == 1600.0
        assert streamer._viewport_height == 900.0

    async def test_viewport_evaluation_exception(self, mock_browser_session):
        """Test that viewport evaluation exception uses fallback values."""
        page = mock_browser_session.must_get_current_page.return_value
        page.evaluate = AsyncMock(side_effect=RuntimeError("JS eval failed"))

        streamer = StreamingService(max_width=1280, max_height=800)
        await streamer.start(mock_browser_session)

        # Should fallback to max values
        assert streamer._viewport_width == 1280.0
        assert streamer._viewport_height == 800.0


class TestDispatchClickJs:
    """Test JavaScript click dispatch method."""

    async def test_dispatch_click_js_success(self, mock_browser_session):
        """Test successful JS click dispatch."""
        mock_browser_session.cdp_client.send.Runtime = MagicMock()
        mock_browser_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            "result": {"value": {"success": True, "tagName": "BUTTON", "id": "submit-btn"}}
        })

        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        await streamer.dispatch_click_js(100, 200, "left")

        mock_browser_session.cdp_client.send.Runtime.evaluate.assert_called_once()
        call_args = mock_browser_session.cdp_client.send.Runtime.evaluate.call_args
        assert "elementFromPoint" in call_args.kwargs["params"]["expression"]

    async def test_dispatch_click_js_no_cdp_client(self):
        """Test click dispatch without CDP client returns early."""
        streamer = StreamingService()
        # Don't start, so no CDP client

        # Should not raise
        await streamer.dispatch_click_js(100, 200)

    async def test_dispatch_click_js_with_scaling(self, mock_browser_session):
        """Test that click coordinates are scaled correctly."""
        mock_browser_session.cdp_client.send.Runtime = MagicMock()
        mock_browser_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={})

        streamer = StreamingService(max_width=1280, max_height=800)
        await streamer.start(mock_browser_session)

        # Set up scaling scenario
        streamer._viewport_width = 1920.0
        streamer._viewport_height = 1080.0
        streamer._actual_screencast_width = 1280.0
        streamer._actual_screencast_height = 720.0

        await streamer.dispatch_click_js(640, 360)

        call_args = mock_browser_session.cdp_client.send.Runtime.evaluate.call_args
        js_code = call_args.kwargs["params"]["expression"]
        # Scaled coordinates should be 960, 540 (center of 1920x1080)
        assert "960" in js_code
        assert "540" in js_code

    async def test_dispatch_click_js_exception_handling(self, mock_browser_session):
        """Test that JS click handles exceptions gracefully."""
        mock_browser_session.cdp_client.send.Runtime = MagicMock()
        mock_browser_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=RuntimeError("JS execution failed")
        )

        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Should not raise
        await streamer.dispatch_click_js(100, 200)


class TestDispatchWithoutCDP:
    """Test dispatch methods without CDP client initialized."""

    async def test_dispatch_mouse_event_without_cdp(self):
        """Test mouse event dispatch without CDP returns early."""
        streamer = StreamingService()
        # Don't start, so no CDP client

        # Should not raise
        await streamer.dispatch_mouse_event("mousePressed", 100, 200)

    async def test_dispatch_key_event_without_cdp(self):
        """Test key event dispatch without CDP returns early."""
        streamer = StreamingService()
        # Don't start, so no CDP client

        # Should not raise
        await streamer.dispatch_key_event("keyDown", "Enter")

    async def test_dispatch_scroll_event_without_cdp(self):
        """Test scroll event dispatch without CDP returns early."""
        streamer = StreamingService()
        # Don't start, so no CDP client

        # Should not raise
        await streamer.dispatch_scroll_event(100, 200, 0, -120)


class TestFrameQueueFull:
    """Test frame queue overflow handling."""

    async def test_frame_dropped_when_queue_full(self, mock_browser_session, valid_jpeg_bytes):
        """Test that frames are dropped when queue is full."""
        # Create streamer with small queue (maxsize=5)
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        b64_data = base64.b64encode(valid_jpeg_bytes).decode("ascii")

        # Fill the queue beyond capacity
        for i in range(10):
            streamer._on_frame({"data": b64_data, "sessionId": i})

        # Queue should be at max capacity (5)
        assert streamer._frame_queue.qsize() == 5
        # But all frames should be counted
        assert streamer.frame_count == 10


class TestFrameProcessingError:
    """Test frame processing error handling."""

    async def test_invalid_base64_handled(self, mock_browser_session):
        """Test that invalid base64 data is handled gracefully."""
        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Send invalid base64 data
        streamer._on_frame({"data": "not-valid-base64!!!", "sessionId": 1})

        # Should not crash, frame count should increment
        # (frame is counted but not queued due to decode error)
        assert streamer.frame_count == 1
        assert streamer._frame_queue.qsize() == 0

    async def test_not_running_skips_frame(self, mock_browser_session, valid_jpeg_bytes):
        """Test that frames are skipped when not running."""
        streamer = StreamingService()
        await streamer.start(mock_browser_session)
        await streamer.stop()  # Stop the streamer

        b64_data = base64.b64encode(valid_jpeg_bytes).decode("ascii")
        streamer._on_frame({"data": b64_data, "sessionId": 1})

        # Frame should be skipped
        assert streamer.frame_count == 0


class TestCoordinateScalingEdgeCases:
    """Test coordinate scaling edge cases."""

    def test_scale_coordinates_zero_viewport(self):
        """Test scaling with zero viewport dimensions returns original coords."""
        streamer = StreamingService()
        streamer._viewport_width = 0
        streamer._viewport_height = 0

        x, y = streamer._scale_coordinates(100, 200)

        assert x == 100
        assert y == 200

    def test_scale_coordinates_negative_viewport(self):
        """Test scaling with negative viewport dimensions returns original coords."""
        streamer = StreamingService()
        streamer._viewport_width = -100
        streamer._viewport_height = 0

        x, y = streamer._scale_coordinates(100, 200)

        assert x == 100
        assert y == 200


class TestDispatchMouseEventException:
    """Test mouse event dispatch exception handling."""

    async def test_dispatch_mouse_event_cdp_exception(self, mock_browser_session):
        """Test that CDP mouse dispatch exceptions are handled."""
        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(
            side_effect=RuntimeError("CDP error")
        )

        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Should not raise
        await streamer.dispatch_mouse_event("mousePressed", 100, 200)


class TestDispatchScrollEventException:
    """Test scroll event dispatch exception handling."""

    async def test_dispatch_scroll_event_cdp_exception(self, mock_browser_session):
        """Test that CDP scroll dispatch exceptions are handled."""
        mock_browser_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(
            side_effect=RuntimeError("CDP error")
        )

        streamer = StreamingService()
        await streamer.start(mock_browser_session)

        # Should not raise
        await streamer.dispatch_scroll_event(100, 200, 0, -120)
