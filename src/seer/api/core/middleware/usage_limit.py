"""
Centralized usage limit enforcement middleware.

This middleware enforces subscription-based usage limits BEFORE route handlers execute,
providing a single source of truth for all limit checks.
"""
import json
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from seer.api.core.middleware.path_allowlist import is_public_path
from seer.database import User
from seer.logger import get_logger
from seer.observability import (
    ChatDisabledError,
    MessageLimitExceeded,
    RunLimitExceeded,
    WorkflowLimitExceeded,
    get_limits_for_user,
    get_monthly_run_count,
    get_total_chat_message_count,
    get_workflow_count,
    resolve_user_tier,
)

logger = get_logger(__name__)


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """
    Centralized usage limit enforcement for all subscription tiers.

    Checks limits before route handlers execute based on:
    - Request path pattern matching
    - Current user usage counters
    - User's subscription tier

    Returns HTTP 402 Payment Required with upgrade prompt if limit exceeded.

    Enforcement Points:
    1. Workflow creation: POST /api/v1/workflows
    2. Workflow runs: POST /api/v1/workflows/{id}/run
    3. Chat messages: POST /api/workflow-agent/{id}/chat
    4. Polling intervals: POST /api/v1/trigger-subscriptions (soft enforcement)
    """

    async def dispatch(self, request: Request, call_next):
        """
        Main middleware dispatch method.

        Checks usage limits based on request path and method before proceeding to handler.
        """
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)

        if is_public_path(path, include_docs=True):
            return await call_next(request)

        user: Optional[User] = getattr(request.state, "db_user", None)
        if user is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        # Get user's tier limits
        limits = await get_limits_for_user(user)

        # Check limits based on request path
        method = request.method

        # 1. Workflow Creation Limit
        if method == "POST" and path == "/api/v1/workflows":
            if not limits.has_unlimited_workflows:
                logger.info("Checking workflow creation limit for user %s", user.id)
                current = await get_workflow_count(user)
                if current >= limits.workflows:
                    tier = await resolve_user_tier(user)
                    error = WorkflowLimitExceeded(limits.workflows, current, tier)
                    logger.warning(
                        "Workflow creation limit exceeded for user %s (tier=%s, current=%d, limit=%d)",
                        user.id,
                        tier.value,
                        current,
                        limits.workflows,
                    )
                    return JSONResponse(status_code=402, content=error.to_dict())

        # 2. Workflow Run Limit
        elif method == "POST" and "/run" in path and "/api/v1/workflows/" in path:
            if not limits.has_unlimited_runs:
                current = await get_monthly_run_count(user)
                if current >= limits.runs_monthly:
                    tier = await resolve_user_tier(user)
                    error = RunLimitExceeded(limits.runs_monthly, current, tier)
                    logger.warning(
                        "Workflow run limit exceeded for user %s (tier=%s, current=%d, limit=%d)",
                        user.id,
                        tier.value,
                        current,
                        limits.runs_monthly,
                    )
                    return JSONResponse(status_code=402, content=error.to_dict())

        # 3. Chat Message Limit (PER USER, not per workflow)
        elif method == "POST" and "/chat" in path and "/workflow-agent/" in path:
            # Check if chat is enabled
            if limits.is_chat_disabled:
                error = ChatDisabledError()
                logger.info("Chat access denied for user %s (self-hosted mode)", user.id)
                return JSONResponse(status_code=403, content=error.to_dict())

            if not limits.has_unlimited_chat:
                # Count ALL chat messages for this user (across all workflows)
                current = await get_total_chat_message_count(user)
                if current >= limits.chat_messages_total:
                    tier = await resolve_user_tier(user)
                    error = MessageLimitExceeded(
                        limit=limits.chat_messages_total,
                        current=current,
                        tier=tier,
                    )
                    logger.warning(
                        "Chat message limit exceeded for user %s (tier=%s, current=%d, limit=%d)",
                        user.id,
                        tier.value,
                        current,
                        limits.chat_messages_total,
                    )
                    return JSONResponse(status_code=402, content=error.to_dict())

        # 4. Polling Frequency Validation (reads request body for soft enforcement)
        elif method == "POST" and "trigger-subscriptions" in path:
            await self._validate_polling_interval(request, user, limits)

        # All checks passed - proceed to handler
        return await call_next(request)

    async def _validate_polling_interval(self, request: Request, user: User, limits) -> None:
        """
        Soft enforcement of polling interval limits.

        Reads request body, checks poll_interval_seconds, and logs warning if too fast.
        Does not block the request, but logs for monitoring.
        """
        try:
            body = await request.body()
            # Must reset body for downstream handlers
            request._body = body

            data = json.loads(body.decode("utf-8"))
            requested_interval = data.get("poll_interval_seconds")

            if requested_interval is not None:
                min_interval = limits.poll_min_interval_seconds
                if requested_interval < min_interval:
                    tier = await resolve_user_tier(user)
                    logger.warning(
                        "Poll interval too fast for user %s (tier=%s, requested=%ds, min=%ds)",
                        user.id,
                        tier.value,
                        requested_interval,
                        min_interval,
                        extra={
                            "user_id": user.id,
                            "tier": tier.value,
                            "requested_interval": requested_interval,
                            "min_interval": min_interval,
                        },
                    )
                    # Note: We log but don't block. Service layer will clamp the value.
        except Exception as e:
            # If body parsing fails, let it through (validation will catch it downstream)
            logger.debug("Failed to parse request body for polling interval check: %s", e)
