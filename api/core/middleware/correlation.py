"""
Request correlation middleware for tracing requests across services.

Generates or accepts X-Request-ID headers and attaches to request state.
"""
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to attach correlation IDs to all requests.

    - Accepts X-Request-ID from client (if provided)
    - Generates new UUID if not provided
    - Attaches to request.state.correlation_id
    - Returns X-Request-ID in response headers
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Attach to request state for access in handlers
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = correlation_id

        return response
