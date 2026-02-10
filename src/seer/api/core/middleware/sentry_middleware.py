"""
Sentry context enrichment middleware for FastAPI.

Enriches Sentry error context with:
- Correlation ID for request tracing
- User information (when authenticated)
- Request metadata (path, method, etc.)
- Seer mode tag for environment filtering

All operations are non-blocking and fail silently.
"""
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from seer.config import config
from seer.logger import get_logger
from seer.observability.sentry_client import (
    set_context,
    set_tag,
    set_user_context,
)

logger = get_logger(__name__)


class SentryContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enrich Sentry scope with request context.

    Should be added AFTER CorrelationMiddleware so correlation_id is available.
    All operations are non-blocking and fail silently to avoid impacting requests.

    Context added:
    - Tags: correlation_id, seer_mode, request_method, request_path
    - User: user_id, email (if authenticated)
    - Context: full request metadata dict
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Set correlation ID tag (from CorrelationMiddleware)
        correlation_id = getattr(request.state, "correlation_id", None)
        if correlation_id:
            set_tag("correlation_id", correlation_id)

        # Set seer_mode tag for filtering in Sentry
        set_tag("seer_mode", config.seer_mode)

        # Set basic request tags
        set_tag("request_method", request.method)
        set_tag("request_path", request.url.path)

        # Set user context if authenticated
        user = getattr(request.state, "user", None)
        if user:
            set_user_context(
                user_id=user.user_id,
                email=getattr(user, "email", None),
                username=f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip() or None,
            )
            # Set user tags for indexed searching in Sentry
            set_tag("user_id", user.user_id)
            if getattr(user, "email", None):
                set_tag("user_email", user.email)

        # Set rich request context (not indexed, but visible in error details)
        request_context = {
            "url": str(request.url),
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.query_params) if request.query_params else None,
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        if correlation_id:
            request_context["correlation_id"] = correlation_id
        if user:
            request_context["user_id"] = user.user_id

        set_context("seer_request", request_context)

        # Process request
        response = await call_next(request)

        # Optionally set response tags (useful for filtering by status code)
        set_tag("response_status", str(response.status_code))

        return response
