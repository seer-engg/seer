"""
Patch cdp_use.CDPClient to disable WebSocket compression.

The ``websockets`` library (v15) defaults to ``compression="deflate"``
(``permessage-deflate``).  When a TLS-terminating reverse proxy sits
between the API container and Browserless (e.g. ALB, Nginx, Cloudflare),
the proxy can strip or mishandle the ``Sec-WebSocket-Extensions`` header
during the upgrade handshake while the server still sends compressed
frames with RSV1=1.  The client then rejects them with:

    sent 1002 (protocol error) reserved bits must be 0

Disabling compression at the WebSocket layer is safe because:
- CDP messages are small JSON; compression savings are negligible.
- It eliminates an entire class of proxy-incompatibility failures.
- Chrome/Browserless itself does not require ``permessage-deflate``.

This module is imported once from ``seer.services.browser.__init__`` so
the patch is in place before any ``CDPClient`` is instantiated.
"""
from __future__ import annotations

import asyncio
import logging

import websockets

from seer.logger import get_logger

logger = get_logger(__name__)

_PATCHED = False


def _patch_cdp_client() -> None:
    """Replace ``CDPClient.start`` with a version that disables compression."""
    global _PATCHED  # pylint: disable=global-statement  # Reason: one-time idempotent guard
    if _PATCHED:
        return

    from cdp_use.client import CDPClient  # pylint: disable=import-outside-toplevel  # Reason: avoid import cycle

    async def _start_no_compression(self: CDPClient) -> None:
        """CDPClient.start with compression disabled and ping tuned."""
        if self.ws is not None:
            raise RuntimeError("Client is already started")

        cdp_logger = logging.getLogger("cdp_use.client")
        cdp_logger.info(
            "Connecting to %s (max frame size: %.0fMB, compression: off)",
            self.url,
            self.max_ws_frame_size / 1024 / 1024,
        )
        connect_kwargs: dict = {
            "max_size": self.max_ws_frame_size,
            "compression": None,      # Disable permessage-deflate
            "ping_interval": 20,      # Send ping every 20s (websockets default)
            "ping_timeout": 30,       # Wait 30s for pong (generous for cross-AZ)
        }
        if self.additional_headers:
            connect_kwargs["additional_headers"] = self.additional_headers
        self.ws = await websockets.connect(self.url, **connect_kwargs)
        # pylint: disable=protected-access
        # Reason: monkey-patching CDPClient.start — must replicate original internals
        self._message_handler_task = asyncio.create_task(self._handle_messages())

    CDPClient.start = _start_no_compression  # type: ignore[assignment]
    _PATCHED = True
    logger.info("Patched CDPClient.start: WebSocket compression disabled")


# Apply on import
_patch_cdp_client()
