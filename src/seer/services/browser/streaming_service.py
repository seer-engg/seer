# pylint: disable=broad-exception-caught,logging-fstring-interpolation,too-many-instance-attributes
# Reason: CDP operations require flexible exception handling and dynamic logging.
# Streaming state requires multiple related attributes (viewport, frame, queue, CDP session).
"""
CDP screencast streaming service for live browser viewing.

Manages Chrome DevTools Protocol screencast for a single browser session,
providing frame capture and input dispatch decoupled from transport (WebSocket).
"""
from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from typing import Optional

from PIL import Image
from websockets.exceptions import ConnectionClosed

from browser_use import BrowserSession

from seer.logger import get_logger

logger = get_logger(__name__)

# WebSocket close codes / error fragments that indicate the CDP connection is dead
_CDP_DEAD_INDICATORS = (
    "reserved bits must be 0",
    "no close frame received",
    "websocket connection closed",
    "connection is closed",
    "client is not started",
)


class StreamingService:
    """Manages CDP screencast streaming and input dispatch for a browser session."""

    def __init__(
        self,
        quality: int = 60,
        max_width: int = 1280,
        max_height: int = 800,
        every_nth_frame: int = 1,
    ) -> None:
        self._quality = quality
        self._max_width = max_width
        self._max_height = max_height
        self._every_nth_frame = every_nth_frame
        self._frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        self._cdp_client = None
        self._owns_cdp_client = False  # True when we created the CDP client (cross-process mode)
        self._target_session_id: Optional[str] = None
        self._running = False
        self._frame_count = 0
        # Signalled when the CDP WebSocket is detected as dead
        self._cdp_dead_event: asyncio.Event = asyncio.Event()
        # Viewport dimensions for coordinate scaling
        self._viewport_width: float = 0
        self._viewport_height: float = 0
        # Actual screencast frame dimensions (parsed from JPEG data, not CDP metadata)
        self._actual_screencast_width: float = 0
        self._actual_screencast_height: float = 0

    async def start(self, browser_session: BrowserSession) -> None:
        """Start screencast on the current page via CDP.

        Args:
            browser_session: Active browser-use BrowserSession with started browser
        """
        page = await browser_session.must_get_current_page()
        # NOTE: browser-use stores target_id as private _target_id (no public property)
        # pylint: disable-next=protected-access
        cdp_session = await browser_session.get_or_create_cdp_session(page._target_id)
        self._cdp_client = browser_session.cdp_client
        self._target_session_id = cdp_session.session_id

        # Capture actual viewport size for coordinate scaling
        try:
            viewport = await page.evaluate(
                "() => ({ width: window.innerWidth, height: window.innerHeight })"
            )
            # Handle both dict and JSON string returns from evaluate
            if isinstance(viewport, str):
                viewport = json.loads(viewport)
            self._viewport_width = float(viewport.get("width", self._max_width))
            self._viewport_height = float(viewport.get("height", self._max_height))
            logger.info(
                f"Browser viewport: {self._viewport_width:.0f}x{self._viewport_height:.0f}, "
                f"Screencast max: {self._max_width}x{self._max_height}"
            )
        except Exception as e:
            logger.warning(f"Could not get viewport size, using screencast max: {e}")
            self._viewport_width = float(self._max_width)
            self._viewport_height = float(self._max_height)

        await self._start_screencast()

    async def start_from_target(self, target_id: str) -> None:
        """Start screencast by connecting to an existing browser page via its target_id.

        Used for cross-process streaming: the browser session lives in the worker
        process, but the API process needs to stream from the same page.
        Both connect to the same Browserless container via CDP.

        Args:
            target_id: CDP target ID of the page to stream from
        """
        from cdp_use import CDPClient  # pylint: disable=import-outside-toplevel  # Reason: Only needed for cross-process streaming
        from seer.config import config as app_config  # pylint: disable=import-outside-toplevel  # Reason: Avoid shadowing

        cdp_url = app_config.browserless_cdp_url
        logger.info(f"Connecting to Browserless for HITL streaming: target={target_id}")

        cdp_client = CDPClient(cdp_url)
        await cdp_client.start()
        self._cdp_client = cdp_client
        self._owns_cdp_client = True  # Track that we need to close this

        # Attach to the existing page target
        result = await cdp_client.send.Target.attachToTarget(
            params={"targetId": target_id, "flatten": True}
        )
        self._target_session_id = result["sessionId"]

        # Use screencast max dimensions as viewport fallback (can't call page.evaluate without Playwright)
        self._viewport_width = float(self._max_width)
        self._viewport_height = float(self._max_height)

        await self._start_screencast()
        logger.info(f"HITL screencast started for target={target_id}")

    async def _start_screencast(self) -> None:
        """Register frame handler and start CDP screencast."""
        # Register event handler for screencast frames
        # NOTE: cdp_use library exposes _event_registry as the only way to register handlers
        # pylint: disable-next=protected-access
        self._cdp_client._event_registry.register(
            "Page.screencastFrame", self._on_frame
        )

        # Start screencast via raw CDP (not in cdp_use typed lib)
        await self._cdp_client.send_raw(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": self._quality,
                "maxWidth": self._max_width,
                "maxHeight": self._max_height,
                "everyNthFrame": self._every_nth_frame,
            },
            session_id=self._target_session_id,
        )
        self._running = True
        self._frame_count = 0
        logger.info(f"Screencast started (quality={self._quality}, session={self._target_session_id})")

    async def stop(self) -> None:
        """Stop screencast and unregister handler."""
        if not self._running:
            return

        self._running = False

        if self._cdp_client and self._target_session_id:
            try:
                await self._cdp_client.send_raw(
                    "Page.stopScreencast",
                    {},
                    session_id=self._target_session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to stop screencast: {e}")

        # Close CDP client if we created it (cross-process mode)
        if self._owns_cdp_client and self._cdp_client:
            try:
                await self._cdp_client.stop()
            except Exception as e:
                logger.warning(f"Failed to close CDP client: {e}")

        logger.info(f"Screencast stopped (total frames: {self._frame_count})")

    async def get_frame(self, timeout: float = 5.0) -> Optional[bytes]:
        """Get next base64-decoded JPEG frame from queue.

        Args:
            timeout: Max seconds to wait for a frame

        Returns:
            JPEG bytes or None on timeout
        """
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def _scale_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """Scale coordinates from screencast space to actual viewport space.

        CDP screencast scales the browser to fit within maxWidth/maxHeight while
        maintaining aspect ratio. We use the actual JPEG frame dimensions (parsed
        from frame data) rather than max bounds for accurate scaling.

        Args:
            x: X coordinate in screencast space
            y: Y coordinate in screencast space

        Returns:
            Tuple of (scaled_x, scaled_y) in viewport space
        """
        if self._viewport_width <= 0 or self._viewport_height <= 0:
            return x, y

        # Use actual screencast dimensions if available, fallback to max bounds
        width = self._actual_screencast_width if self._actual_screencast_width > 0 else self._max_width
        height = self._actual_screencast_height if self._actual_screencast_height > 0 else self._max_height

        scale_x = self._viewport_width / width
        scale_y = self._viewport_height / height
        return x * scale_x, y * scale_y

    async def dispatch_mouse_event(
        self,
        event_type: str,
        x: float,
        y: float,
        *,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        """Dispatch a mouse event via CDP.

        Args:
            event_type: CDP mouse event type (mousePressed, mouseReleased, mouseMoved)
            x: X coordinate in screencast space
            y: Y coordinate in screencast space
            button: Mouse button (left, middle, right)
            click_count: Number of clicks
        """
        if not self._cdp_client or not self._target_session_id:
            logger.warning(
                f"Cannot dispatch mouse event: cdp_client={bool(self._cdp_client)}, "
                f"session_id={bool(self._target_session_id)}"
            )
            return

        # Scale coordinates from screencast space to actual viewport space
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        logger.info(
            f"Mouse {event_type}: input=({x:.0f},{y:.0f}) -> scaled=({scaled_x:.0f},{scaled_y:.0f})"
        )

        try:
            await self._cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": event_type,
                    "x": scaled_x,
                    "y": scaled_y,
                    "button": button,
                    "clickCount": click_count,
                },
                session_id=self._target_session_id,
            )
        except Exception as e:
            logger.error(f"CDP dispatchMouseEvent failed: {e}")
            if self._is_connection_error(e):
                self._mark_cdp_dead(str(e))

    async def dispatch_key_event(
        self,
        event_type: str,
        key: str,
        *,
        code: str = "",
        text: str = "",
        modifiers: int = 0,
    ) -> None:
        """Dispatch a keyboard event via CDP.

        Args:
            event_type: CDP key event type (keyDown, keyUp, char)
            key: Key value (e.g., "Enter", "a")
            code: Physical key code (e.g., "Enter", "KeyA")
            text: Text generated by the key
            modifiers: Bit field for modifier keys (Alt=1, Ctrl=2, Meta=4, Shift=8)
        """
        if not self._cdp_client or not self._target_session_id:
            return
        params = {
            "type": event_type,
            "key": key,
            "modifiers": modifiers,
        }
        if code:
            params["code"] = code
        if text:
            params["text"] = text
        await self._cdp_client.send.Input.dispatchKeyEvent(
            params=params,
            session_id=self._target_session_id,
        )

    async def dispatch_click_js(self, x: float, y: float, button: str = "left") -> None:
        """Dispatch a click using CDP mousePressed + mouseReleased sequence.

        Uses real CDP input events instead of JavaScript injection so that:
        1. Clicks propagate into cross-origin iframes (e.g. reCAPTCHA)
        2. Events have isTrusted=true at the browser compositor level
        3. Bot-detection scripts see authentic input events

        Args:
            x: X coordinate in screencast space (will be scaled)
            y: Y coordinate in screencast space (will be scaled)
            button: Mouse button (left, middle, right)
        """
        if not self._cdp_client or not self._target_session_id:
            logger.warning("Cannot dispatch click: CDP not initialized")
            return

        # Scale coordinates from screencast space to viewport space
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        logger.info(f"CDP Click: input=({x:.0f},{y:.0f}) -> viewport=({scaled_x:.0f},{scaled_y:.0f})")

        try:
            # Move mouse to target first (some sites track mouse movement)
            await self._cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mouseMoved",
                    "x": scaled_x,
                    "y": scaled_y,
                },
                session_id=self._target_session_id,
            )
            # mousePressed + mouseReleased = full click at compositor level
            # This is how Playwright implements .click() under the hood
            await self._cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mousePressed",
                    "x": scaled_x,
                    "y": scaled_y,
                    "button": button,
                    "clickCount": 1,
                },
                session_id=self._target_session_id,
            )
            await self._cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mouseReleased",
                    "x": scaled_x,
                    "y": scaled_y,
                    "button": button,
                    "clickCount": 1,
                },
                session_id=self._target_session_id,
            )
            logger.info(f"CDP Click completed at viewport=({scaled_x:.0f},{scaled_y:.0f})")
        except Exception as e:
            logger.error(f"CDP Click failed: {e}")
            if self._is_connection_error(e):
                self._mark_cdp_dead(str(e))

    async def dispatch_scroll_event(
        self, x: float, y: float, delta_x: float = 0, delta_y: float = -120
    ) -> None:
        """Dispatch a scroll (mouseWheel) event via CDP.

        Args:
            x: X coordinate of scroll position in screencast space
            y: Y coordinate of scroll position in screencast space
            delta_x: Horizontal scroll delta
            delta_y: Vertical scroll delta (negative = scroll down)
        """
        if not self._cdp_client or not self._target_session_id:
            logger.warning("Cannot dispatch scroll event: CDP not initialized")
            return

        # Scale coordinates from screencast space to actual viewport space
        scaled_x, scaled_y = self._scale_coordinates(x, y)

        try:
            await self._cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mouseWheel",
                    "x": scaled_x,
                    "y": scaled_y,
                    "deltaX": delta_x,
                    "deltaY": delta_y,
                },
                session_id=self._target_session_id,
            )
        except Exception as e:
            logger.error(f"CDP scroll event failed: {e}")
            if self._is_connection_error(e):
                self._mark_cdp_dead(str(e))

    def _on_frame(self, params: dict, session_id: str = None) -> None:  # pylint: disable=unused-argument  # Reason: CDP callback signature requires session_id parameter
        """CDP callback for screencast frames. Decodes and queues frame data."""
        if not self._running:
            return

        self._frame_count += 1
        frame_data = params.get("data", "")
        session_id_val = params.get("sessionId", 0)

        try:
            jpeg_bytes = base64.b64decode(frame_data)

            # Extract actual JPEG frame dimensions (only on first frame)
            # NOTE: CDP metadata.deviceWidth/deviceHeight report VIEWPORT size, not frame size.
            # The screencast scales frames to fit within maxWidth/maxHeight, so we must
            # parse the actual JPEG to get true frame dimensions for coordinate scaling.
            if self._actual_screencast_width == 0:
                img = Image.open(BytesIO(jpeg_bytes))
                self._actual_screencast_width = float(img.width)
                self._actual_screencast_height = float(img.height)
                logger.info(
                    f"Actual JPEG frame dimensions: {self._actual_screencast_width:.0f}x{self._actual_screencast_height:.0f}"
                )
                img.close()

            # Drop frame if queue is full (non-blocking)
            try:
                self._frame_queue.put_nowait(jpeg_bytes)
            except asyncio.QueueFull:
                pass  # Drop frame rather than block

            # Acknowledge the frame to CDP
            if self._cdp_client and self._target_session_id:
                asyncio.get_event_loop().create_task(
                    self._cdp_client.send_raw(
                        "Page.screencastFrameAck",
                        {"sessionId": session_id_val},
                        session_id=self._target_session_id,
                    )
                )
        except Exception as e:
            logger.warning(f"Error processing screencast frame: {e}")
            if self._is_connection_error(e):
                self._mark_cdp_dead(str(e))

    @property
    def is_running(self) -> bool:
        """Whether screencast is currently active."""
        return self._running

    @property
    def is_cdp_dead(self) -> bool:
        """Whether the CDP WebSocket connection has been detected as dead."""
        return self._cdp_dead_event.is_set()

    async def wait_until_dead(self) -> None:
        """Block until the CDP connection is detected as dead.

        Used by the streaming loop to know when to stop sending frames
        and notify the client.
        """
        await self._cdp_dead_event.wait()

    def _mark_cdp_dead(self, reason: str) -> None:
        """Mark the CDP connection as dead and stop the screencast.

        Called when a CDP send operation raises an error that indicates the
        underlying WebSocket is no longer usable (close-code 1002, connection
        closed, etc.).  Sets the event so the streaming loop can break out
        and notify the frontend.
        """
        if self._cdp_dead_event.is_set():
            return  # Already flagged
        logger.error(f"CDP connection lost: {reason}")
        self._running = False
        self._cdp_dead_event.set()

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Return True if the exception indicates a dead CDP WebSocket."""
        msg = str(exc).lower()
        if any(indicator in msg for indicator in _CDP_DEAD_INDICATORS):
            return True
        if isinstance(exc, (ConnectionError, ConnectionClosed)):
            return True
        return False

    @property
    def frame_count(self) -> int:
        """Total number of frames received."""
        return self._frame_count
