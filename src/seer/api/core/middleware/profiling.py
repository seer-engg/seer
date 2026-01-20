"""
PyInstrument middleware to profile API requests and persist HTML reports.

Usage:
    Set REQUEST_PROFILING_ENABLED=true to enable middleware and
    REQUEST_PROFILING_OUTPUT_DIR to control the output location.
"""
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from fastapi import Request, Response
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from seer.logger import get_logger

logger = get_logger("api.middleware.profiling")


class PyInstrumentMiddleware(BaseHTTPMiddleware):
    """Middleware to profile requests with pyinstrument and save HTML reports."""

    def __init__(self, app: ASGIApp, enabled: bool, output_dir: str):
        super().__init__(app)
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip noisy endpoints like health checks to reduce clutter
        if request.url.path == "/health":
            return await call_next(request)

        profiler = Profiler(async_mode="enabled")
        profiler.start()

        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            profiler.stop()
            profile_path = self._save_profile(profiler, request)
            if response is not None and profile_path:
                response.headers["X-Profiler-Report-Path"] = str(profile_path)

    def _save_profile(self, profiler: Profiler, request: Request) -> Optional[Path]:
        """Persist profiler output to HTML and return the path if successful."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{self._build_filename(request)}.html"
            output_path.write_text(profiler.output_html(), encoding="utf-8")
            logger.info("Saved pyinstrument profile for %s to %s", request.url.path, output_path)
            return output_path
        except Exception as exc:  # pylint: disable=broad-except # Reason: Avoid failing requests due to profiling issues
            logger.warning("Failed to save pyinstrument profile for %s: %s", request.url.path, exc, exc_info=True)
            return None

    def _build_filename(self, request: Request) -> str:
        """Generate a safe, descriptive filename for the profile report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path_fragment = re.sub(r"[^a-zA-Z0-9_-]+", "_", request.url.path.strip("/")) or "root"
        correlation_id = getattr(request.state, "correlation_id", None)
        suffix = f"-{correlation_id}" if correlation_id else ""
        return f"{timestamp}-{request.method.lower()}-{path_fragment}{suffix}"
