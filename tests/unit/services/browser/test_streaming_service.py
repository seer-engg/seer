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
