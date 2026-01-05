"""
PostHog analytics service for server-side event tracking.

Provides centralized analytics tracking for:
- API requests and responses
- Workflow executions
- Agent runs and completions
- User actions and system events
"""
from typing import Optional, Dict, Any
import posthog
from shared.config import config
from shared.logger import get_logger

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
            # Disable sync mode for better performance
            # Events are batched and flushed via middleware
            posthog.sync_mode = False
            cls._initialized = True
            logger.info(f"✅ PostHog analytics initialized (host: {config.posthog_host})")
        except Exception as e:
            logger.error(f"Failed to initialize PostHog: {e}", exc_info=True)

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
        except Exception as e:
            # Never let analytics errors break the application
            logger.error(f"PostHog capture failed: {e}", exc_info=True)

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
        except Exception as e:
            logger.error(f"PostHog identify failed: {e}", exc_info=True)

    @classmethod
    def flush(cls):
        """Flush pending events to PostHog."""
        if not config.is_posthog_configured:
            return

        try:
            posthog.flush()
        except Exception as e:
            logger.error(f"PostHog flush failed: {e}", exc_info=True)

    @classmethod
    def shutdown(cls):
        """Shutdown PostHog client and flush remaining events."""
        if not cls._initialized:
            return

        try:
            posthog.shutdown()
            cls._initialized = False
            logger.info("PostHog analytics shutdown")
        except Exception as e:
            logger.error(f"PostHog shutdown failed: {e}", exc_info=True)


# Convenience exports
analytics = AnalyticsService
