"""
Sentry error monitoring client singleton and tracking utilities.

Provides:
- Lazy initialization of Sentry SDK
- Error capture with context enrichment
- Performance monitoring integration
- Graceful degradation when Sentry is not configured

Usage:
    from seer.observability.sentry_client import init_sentry, capture_exception

    # Initialize at app startup (before FastAPI app creation)
    init_sentry()

    # Capture an exception with context
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e)
"""
from typing import Any, Dict, Optional

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

# Module-level state (lazy initialized)
SENTRY_INITIALIZED = False

# Expected exceptions that are business logic, not bugs
# These will be captured but marked as non-error
EXPECTED_EXCEPTION_TYPES = (
    "UsageLimitError",
    "ChatDisabledError",
    "WorkflowLimitExceeded",
    "RunLimitExceeded",
    "MessageLimitExceeded",
    "TrialExpiredError",
    "CreditLimitExceeded",
    "PollingIntervalTooFast",
    "RunCostCapExceeded",
)

# Paths to skip performance tracing (reduce noise)
SKIP_TRANSACTION_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/.well-known/oauth-protected-resource",
}

# Headers to scrub from Sentry events
SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}


def _scrub_headers(headers: Dict[str, Any]) -> None:
    """
    Scrub sensitive headers in-place from a headers dict.

    Args:
        headers: Headers dictionary to modify
    """
    for key in SENSITIVE_HEADERS:
        if key in headers:
            headers[key] = "[Filtered]"
    # Also check case-insensitive
    for header_key in list(headers.keys()):
        if header_key.lower() in SENSITIVE_HEADERS:
            headers[header_key] = "[Filtered]"


def _before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter and enrich events before sending to Sentry.

    - Filters expected business logic exceptions (UsageLimitError, etc.)
    - Scrubs sensitive headers from request data
    - Adds fingerprinting for better error grouping

    Args:
        event: The Sentry event dict
        hint: Additional context including original exception

    Returns:
        Modified event dict, or None to drop the event
    """
    # Check if this is an expected business exception
    if "exc_info" in hint:
        exc_type = hint["exc_info"][0]
        exc_name = exc_type.__name__ if exc_type else None

        if exc_name in EXPECTED_EXCEPTION_TYPES:
            # Mark as handled/expected - will show as "info" level
            event["level"] = "info"
            event["tags"] = event.get("tags", {})
            event["tags"]["expected_error"] = "true"
            event["tags"]["error_type"] = exc_name

    # Scrub sensitive headers from request data
    if "request" in event:
        request_data = event["request"]
        if "headers" in request_data and isinstance(request_data["headers"], dict):
            _scrub_headers(request_data["headers"])

    return event


def _traces_sampler(sampling_context: Dict[str, Any]) -> float:
    """
    Dynamic trace sampler that skips health checks and high-frequency endpoints.

    Args:
        sampling_context: Context from Sentry with transaction info

    Returns:
        Sample rate (0.0 to 1.0)
    """
    # Get transaction name (usually the URL path)
    transaction_name = sampling_context.get("transaction_context", {}).get("name", "")

    # Skip health check and documentation endpoints
    if any(skip_path in transaction_name for skip_path in SKIP_TRANSACTION_PATHS):
        return 0.0

    # Use configured sample rate for all other transactions
    return config.sentry_traces_sample_rate


def init_sentry() -> bool:
    """
    Initialize Sentry SDK with configuration from environment.

    Must be called BEFORE FastAPI app creation for proper ASGI integration.
    Safe to call multiple times - will only initialize once.

    Returns:
        bool: True if Sentry was initialized, False if not configured
    """
    global SENTRY_INITIALIZED  # pylint: disable=global-statement  # Reason: application singleton pattern

    if SENTRY_INITIALIZED:
        return True

    if not config.is_sentry_configured:
        logger.debug("Sentry not configured (SENTRY_DSN not set), error monitoring disabled")
        return False

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel  # Reason: lazy import to avoid load if not configured
        from sentry_sdk.integrations.fastapi import FastApiIntegration  # pylint: disable=import-outside-toplevel
        from sentry_sdk.integrations.starlette import StarletteIntegration  # pylint: disable=import-outside-toplevel

        sentry_sdk.init(
            dsn=config.sentry_dsn,
            environment=config.sentry_environment or config.env,
            traces_sample_rate=config.sentry_traces_sample_rate,
            profiles_sample_rate=config.sentry_profiles_sample_rate,
            before_send=_before_send,
            traces_sampler=_traces_sampler,
            integrations=[
                StarletteIntegration(transaction_style="url"),
                FastApiIntegration(transaction_style="url"),
            ],
            # Send default PII (email, user id) - we filter sensitive headers in before_send
            send_default_pii=True,
            # Attach stack traces to all messages
            attach_stacktrace=True,
            # Include request bodies (up to 10KB)
            max_request_body_size="medium",
        )

        SENTRY_INITIALIZED = True
        logger.info(
            "Sentry initialized (env=%s, traces=%.0f%%, profiles=%.0f%%)",
            config.sentry_environment or config.env,
            config.sentry_traces_sample_rate * 100,
            config.sentry_profiles_sample_rate * 100,
        )
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry init failures should never crash the app
        logger.warning("Failed to initialize Sentry: %s", e)
        return False


def capture_exception(exc: Optional[BaseException] = None) -> Optional[str]:
    """
    Capture an exception to Sentry.

    Can be called with an exception or without (will capture current exception).
    No-ops gracefully if Sentry is not configured.

    Args:
        exc: Exception to capture (optional, uses sys.exc_info() if not provided)

    Returns:
        Event ID string if captured, None otherwise
    """
    if not SENTRY_INITIALIZED:
        return None

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel
        return sentry_sdk.capture_exception(exc)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry errors should never propagate
        logger.warning("Failed to capture exception to Sentry: %s", e)
        return None


def set_user_context(
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    username: Optional[str] = None,
    **extras: Any,
) -> None:
    """
    Set user context for Sentry error tracking.

    Args:
        user_id: Unique user identifier (e.g., Clerk user_id)
        email: User email address
        username: Display name
        **extras: Additional user attributes
    """
    if not SENTRY_INITIALIZED:
        return

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel

        user_data: Dict[str, Any] = {}
        if user_id:
            user_data["id"] = user_id
        if email:
            user_data["email"] = email
        if username:
            user_data["username"] = username
        user_data.update(extras)

        if user_data:
            sentry_sdk.set_user(user_data)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry errors should never propagate
        logger.debug("Failed to set Sentry user context: %s", e)


def set_tag(key: str, value: str) -> None:
    """
    Set a searchable tag on the current Sentry scope.

    Tags are indexed and can be used for filtering in the Sentry UI.

    Args:
        key: Tag name (e.g., "correlation_id", "seer_mode")
        value: Tag value
    """
    if not SENTRY_INITIALIZED:
        return

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel
        sentry_sdk.set_tag(key, value)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry errors should never propagate
        logger.debug("Failed to set Sentry tag: %s", e)


def set_context(name: str, data: Dict[str, Any]) -> None:
    """
    Set structured context data on the current Sentry scope.

    Unlike tags, context is not indexed but can hold rich structured data.

    Args:
        name: Context name (e.g., "workflow", "request")
        data: Dictionary of context data
    """
    if not SENTRY_INITIALIZED:
        return

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel
        sentry_sdk.set_context(name, data)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry errors should never propagate
        logger.debug("Failed to set Sentry context: %s", e)


def flush(timeout: float = 2.0) -> None:
    """
    Flush pending Sentry events before shutdown.

    Should be called during application shutdown to ensure all events are sent.

    Args:
        timeout: Maximum seconds to wait for flush to complete
    """
    global SENTRY_INITIALIZED  # pylint: disable=global-statement  # Reason: application singleton pattern

    if not SENTRY_INITIALIZED:
        return

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel
        sentry_sdk.flush(timeout=timeout)
        logger.info("Sentry events flushed (timeout=%.1fs)", timeout)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Shutdown errors should be logged but not raised
        logger.warning("Error flushing Sentry events: %s", e)
    finally:
        SENTRY_INITIALIZED = False


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add a breadcrumb to the current Sentry scope.

    Breadcrumbs provide a trail of events leading up to an error.

    Args:
        message: Description of the breadcrumb
        category: Category for grouping (e.g., "http", "query", "custom")
        level: Severity level ("debug", "info", "warning", "error")
        data: Additional structured data
    """
    if not SENTRY_INITIALIZED:
        return

    try:
        import sentry_sdk  # pylint: disable=import-outside-toplevel
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {},
        )
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Sentry errors should never propagate
        logger.debug("Failed to add Sentry breadcrumb: %s", e)
