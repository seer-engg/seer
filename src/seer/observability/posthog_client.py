"""
PostHog analytics client singleton and tracking utilities.

Provides:
- Lazy initialization of PostHog client
- Non-blocking event capture via schedule_async_task
- Graceful degradation when PostHog is not configured

Usage:
    from seer.observability.posthog_client import capture_event, identify_user

    # Track an event (non-blocking)
    capture_event(
        distinct_id=user.user_id,
        event="api_request",
        properties={"path": "/api/v1/workflows", "method": "GET"}
    )

    # Identify a user (non-blocking)
    identify_user(
        distinct_id=user.user_id,
        properties={"email": user.email, "tier": "pro"}
    )
"""
from typing import Any, Dict, Optional

import posthog

from seer.config import config
from seer.core.event_loop import schedule_async_task
from seer.logger import get_logger

logger = get_logger(__name__)

# Module-level client reference (lazy initialized)
POSTHOG_INITIALIZED = False


def _ensure_initialized() -> bool:
    """
    Initialize PostHog client if not already done.

    Returns:
        bool: True if PostHog is configured and initialized, False otherwise
    """
    global POSTHOG_INITIALIZED  # pylint: disable=global-statement  # Reason: application singleton pattern

    if POSTHOG_INITIALIZED:
        return True

    if not config.is_posthog_configured:
        logger.debug("PostHog not configured, analytics disabled")
        return False

    posthog.project_api_key = config.posthog_api_key
    posthog.host = config.posthog_host
    posthog.disabled = False
    posthog.debug = config.env == "dev"

    POSTHOG_INITIALIZED = True
    logger.info("PostHog analytics initialized (host=%s)", config.posthog_host)
    return True


def capture_event(
    distinct_id: str,
    event: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Capture a PostHog event in a non-blocking manner.

    Uses schedule_async_task to ensure the API call doesn't block the response.
    Silently no-ops if PostHog is not configured.

    Args:
        distinct_id: User identifier (user_id from Clerk)
        event: Event name (e.g., "api_request", "mcp_tool_call")
        properties: Additional event properties
    """
    if not _ensure_initialized():
        return

    async def do_capture():
        try:
            posthog.capture(
                distinct_id=distinct_id,
                event=event,
                properties=properties or {},
            )
            logger.debug("PostHog event captured: %s for user %s", event, distinct_id)
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: PostHog failures should never break the application
            logger.warning("PostHog capture failed: %s", e)

    schedule_async_task(
        coro=do_capture(),
        logger=logger,
        error_message=f"Failed to capture PostHog event: {event}",
    )


def identify_user(
    distinct_id: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Identify a user in PostHog (non-blocking).

    Args:
        distinct_id: User identifier (user_id from Clerk)
        properties: User properties (email, name, tier, etc.)
    """
    if not _ensure_initialized():
        return

    async def do_identify():
        try:
            posthog.identify(
                distinct_id=distinct_id,
                properties=properties or {},
            )
            logger.debug("PostHog user identified: %s", distinct_id)
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: PostHog failures should never break the application
            logger.warning("PostHog identify failed: %s", e)

    schedule_async_task(
        coro=do_identify(),
        logger=logger,
        error_message=f"Failed to identify user: {distinct_id}",
    )


def shutdown() -> None:
    """Flush pending events and shutdown PostHog client."""
    global POSTHOG_INITIALIZED  # pylint: disable=global-statement  # Reason: application singleton pattern

    if POSTHOG_INITIALIZED:
        try:
            posthog.flush()
            posthog.shutdown()
            logger.info("PostHog client shutdown complete")
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: shutdown failures should be logged but not raised
            logger.warning("PostHog shutdown error: %s", e)
        finally:
            POSTHOG_INITIALIZED = False
