"""
PostHog analytics middleware for API request tracking.

Tracks API requests with:
- Request path and method
- Response status and latency
- User identification (when authenticated)
- Correlation ID for request tracing

All tracking is non-blocking using schedule_async_task pattern.
"""
import time
from typing import Callable, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from seer.config import config
from seer.logger import get_logger
from seer.observability.posthog_client import capture_event, identify_user

logger = get_logger(__name__)

# Paths to exclude from tracking (health checks, internal endpoints)
EXCLUDED_PATHS: Set[str] = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/.well-known/oauth-protected-resource",
}

# Prefixes to exclude from tracking (endpoints with their own tracking)
# MCP endpoints have dedicated tracking via @track_mcp_tool() in src/seer/mcp/tracking.py
EXCLUDED_PREFIXES: tuple[str, ...] = (
    "/mcp",  # MCP HTTP transport
    "/sse",  # MCP SSE transport
)


class PostHogMiddleware(BaseHTTPMiddleware):
    """
    Non-blocking PostHog analytics middleware.

    Captures API request events after the response is sent using
    the schedule_async_task pattern for zero-latency impact.

    Events tracked:
    - "api_request": Every API request with method, path, status, latency
    - User identification on authenticated requests
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip tracking for excluded paths and OPTIONS requests
        path = request.url.path
        if self._should_skip(request, path):
            return await call_next(request)

        # Record start time
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Track the request (non-blocking)
        self._track_request(request, response, latency_ms)

        return response

    def _should_skip(self, request: Request, path: str) -> bool:
        """Check if request should be excluded from tracking."""
        # Skip OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return True

        # Skip excluded paths
        if path in EXCLUDED_PATHS:
            return True

        # Skip MCP endpoints (they have their own tracking via @track_mcp_tool)
        for prefix in EXCLUDED_PREFIXES:
            if path == prefix or path.startswith(f"{prefix}/"):
                return True

        return False

    def _track_request(self, request: Request, response: Response, latency_ms: float) -> None:
        """Track API request event in PostHog (non-blocking)."""
        # Get user info if authenticated
        user = getattr(request.state, "user", None)
        distinct_id = user.user_id if user else "anonymous"

        # Get correlation ID
        correlation_id = getattr(request.state, "correlation_id", None)

        # Build properties
        properties = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "$current_url": str(request.url),
            "seer_mode": config.seer_mode,
        }

        if correlation_id:
            properties["correlation_id"] = correlation_id

        # Add user context if authenticated
        if user:
            properties["user_email"] = user.email
            properties["authenticated"] = True

            # Identify user (will be deduped by PostHog)
            identify_user(
                distinct_id=user.user_id,
                properties={
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                }
            )
        else:
            properties["authenticated"] = False

        # Capture the event (non-blocking)
        capture_event(
            distinct_id=distinct_id,
            event="api_request",
            properties=properties,
        )
