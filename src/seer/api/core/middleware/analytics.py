"""
Analytics middleware for automatic PostHog event flushing.

Based on PostHog best practices for FastAPI:
https://www.eliehamouche.com/blog/posthog-with-fastapi
"""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from seer.analytics import analytics
from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.middleware.analytics")


class PostHogMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API requests and flush PostHog events.

    Functionality:
    - Tracks all API requests with timing and metadata
    - Flushes PostHog event queue after each request (prevents data loss)
    - Identifies users from request.state.user (set by auth middleware)
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        if not config.is_posthog_configured:
            return await call_next(request)

        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000

        # Track API request event (skip health checks)
        if request.url.path != "/health":
            distinct_id = self._get_distinct_id(request)

            analytics.capture(
                distinct_id=distinct_id,
                event="api_request",
                properties={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "user_agent": request.headers.get("user-agent"),
                    "deployment_mode": config.seer_mode,
                },
            )

        # Flush events with timeout to ensure they're sent before container terminates
        # In serverless/container environments, consumer threads may not finish before shutdown
        analytics.flush()

        return response

    def _get_distinct_id(self, request: Request) -> str:
        """
        Extract distinct_id from authenticated user or use session/IP.

        Priority:
        1. Authenticated user ID (from Clerk)
        2. Session ID (if available)
        3. IP address as fallback
        """
        # Try to get user from request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return request.state.user.user_id

        # Fallback to IP address for anonymous requests
        client_ip = request.client.host if request.client else "unknown"
        return f"anonymous_{client_ip}"
