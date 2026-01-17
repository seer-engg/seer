"""
PostHog analytics service for server-side event tracking.

Provides centralized analytics tracking for:
- API requests and responses
- Workflow executions
- Agent runs and completions
- User actions and system events
- Error tracking and bug reporting
"""
import random
import traceback as tb
from typing import Any, Dict, Optional

import posthog

from seer.config import config
from seer.logger import get_logger

logger = get_logger("shared.analytics")


class AnalyticsService:
    """Singleton service for PostHog analytics."""

    _initialized = False

    @classmethod
    def initialize(cls):
        """Initialize PostHog client with configuration."""
        if cls._initialized:
            return

        if not config.is_posthog_configured:
            logger.info("PostHog analytics disabled or not configured")
            return

        try:
            posthog.api_key = config.posthog_api_key
            posthog.host = config.posthog_host
            # DO NOT set sync_mode - let PostHog use async batching (default)
            # sync_mode=True uses blocking HTTP (requests.post()) which breaks in async FastAPI contexts
            # Background consumer threads will send events asynchronously
            cls._initialized = True
            logger.info("PostHog analytics initialized (host: %s)", config.posthog_host)
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Analytics errors should never break app initialization
            logger.error("Failed to initialize PostHog: %s", e, exc_info=True)

    @classmethod
    def capture(
        cls,
        distinct_id: str,
        event: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Capture an analytics event.

        Args:
            distinct_id: User identifier (from Clerk or session)
            event: Event name (e.g., "workflow_executed", "api_request")
            properties: Additional event properties
        """
        if not config.is_posthog_configured:
            return

        try:
            posthog.capture(
                distinct_id=distinct_id,
                event=event,
                properties=properties or {},
            )
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Never let analytics errors break the application
            logger.error("PostHog capture failed: %s", e, exc_info=True)

    @classmethod
    def identify(
        cls,
        distinct_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Identify a user with properties.

        Args:
            distinct_id: User identifier (from Clerk)
            properties: User properties (email, name, etc.)
        """
        if not config.is_posthog_configured:
            return

        try:
            posthog.identify(
                distinct_id=distinct_id,
                properties=properties or {},
            )
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Analytics errors should not break the application
            logger.error("PostHog identify failed: %s", e, exc_info=True)

    @classmethod
    def flush(cls):
        """
        Flush pending events to PostHog.

        For serverless/container environments, we wait for the consumer thread's
        queue to be fully processed before returning. This ensures events are sent
        before the container terminates.
        """
        if not config.is_posthog_configured:
            return

        try:
            # Wait for consumer thread to process all queued events
            # This is critical in containerized environments where the process may terminate quickly
            if posthog.default_client and posthog.default_client.consumers:
                posthog.default_client.queue.join()
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Flush errors should not break shutdown
            logger.error("PostHog flush failed: %s", e, exc_info=True)

    @classmethod
    def shutdown(cls):
        """Shutdown PostHog client and flush remaining events."""
        if not cls._initialized:
            return

        try:
            posthog.shutdown()
            cls._initialized = False
            logger.debug("PostHog analytics shutdown")
        except TypeError as e:
            # Ignore "NoneType is not iterable" error in sync_mode
            # This is a known PostHog SDK bug when shutting down without consumers
            if "'NoneType' object is not iterable" not in str(e):
                logger.error("PostHog shutdown failed: %s", e, exc_info=True)
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Shutdown errors should be logged but not raised
            logger.error("PostHog shutdown failed: %s", e, exc_info=True)

    @classmethod
    def capture_error(  # pylint: disable=too-many-positional-arguments # Reason: All parameters are essential for error context
        cls,
        distinct_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        error_location: str = "unknown",
        truncate_stacktrace: int = 2000,
    ) -> None:
        """
        Capture an error event to PostHog with context.

        Args:
            distinct_id: User identifier
            error: Exception instance
            context: Additional context (request data, etc.)
            error_location: Location where error occurred (e.g., "tool_execution", "workflow_run")
            truncate_stacktrace: Max length of stack trace to include
        """
        if not config.is_posthog_configured:
            return

        # Sample errors based on configuration
        if random.random() > config.posthog_error_sampling_rate:
            return

        try:
            # Extract stack trace
            stack_trace = "".join(tb.format_exception(type(error), error, error.__traceback__))
            if len(stack_trace) > truncate_stacktrace:
                stack_trace = stack_trace[:truncate_stacktrace] + "... (truncated)"

            # Build error properties
            properties = {
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],  # Truncate message
                "error_location": error_location,
                "stack_trace": stack_trace,
                "deployment_mode": config.seer_mode,
            }

            # Add filtered context
            if context:
                properties["context"] = PrivacyFilter.filter_dict(context)

            cls.capture(
                distinct_id=distinct_id,
                event="error_occurred",
                properties=properties,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Error tracking failures must not break the application
            logger.error("Failed to capture error event: %s", e, exc_info=True)

    @classmethod
    def capture_tool_error(
        cls,
        distinct_id: str,
        tool_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture tool execution error."""
        cls.capture_error(
            distinct_id=distinct_id,
            error=error,
            context={**(context or {}), "tool_name": tool_name},
            error_location="tool_execution",
        )


class PrivacyFilter:
    """Filters sensitive data from event properties."""

    SENSITIVE_KEYS = {
        'password', 'secret', 'token', 'key', 'api_key', 'access_token',
        'refresh_token', 'bearer', 'authorization', 'credential', 'auth',
        'ssn', 'credit_card', 'email', 'phone', 'address'
    }

    @classmethod
    def filter_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive keys from dictionary."""
        if not config.posthog_filter_sensitive_data:
            return data

        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            # Check if key contains sensitive terms
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_KEYS):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = cls.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [cls.filter_dict(v) if isinstance(v, dict) else v for v in value]
            else:
                filtered[key] = value
        return filtered


# Convenience exports
analytics = AnalyticsService  # pylint: disable=invalid-name # Reason: Lowercase name for convenience import pattern
