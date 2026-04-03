"""
FastAPI server for Seer LangGraph agents.

Provides REST API endpoints for:
- Thread management (create, get state)
- Run execution with streaming

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 2024 --reload
"""
import asyncio
import os
import signal
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from typing import cast
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from seer.api.agents.checkpointer import checkpointer_lifespan
from seer.api.core.middleware.auth import ClerkAuthMiddleware
from seer.api.core.middleware.correlation import CorrelationMiddleware
from seer.api.core.middleware.organization import OrganizationContextMiddleware
from seer.api.core.middleware.usage_limit import UsageLimitMiddleware
from seer.api.router import router
from seer.api.tools.router import router as tools_router
from seer.api.tracking.router import router as tracking_router
from seer.api.byok.router import router as byok_router
from seer.config import config
from seer.database import db_lifespan
from seer.logger import get_logger
from seer.observability.exceptions import UsageLimitError

# Initialize Sentry BEFORE app creation for proper ASGI integration
if config.is_sentry_configured:
    from seer.observability.sentry_client import init_sentry, flush as sentry_flush  # pylint: disable=ungrouped-imports  # Reason: Must init before app creation
    if init_sentry():
        get_logger("api.main").info("Sentry error monitoring initialized")

# MCP imports for combined server
if config.mcp_enabled:
    from seer.mcp.server import create_combined_mcp_app, oauth_protected_resource_metadata

# Middleware order is important:
# Think of middleware as layers wrapping your core application (the route handler).
# The first middleware you add forms the innermost layer,
# while the last one added forms the outermost layer.

logger = get_logger("api.main")


@contextmanager
def _install_shutdown_signal_handlers(shutdown_event: asyncio.Event):
    """Set an asyncio event as soon as the process receives a shutdown signal."""
    previous_handlers: dict[signal.Signals, signal.Handlers] = {}

    def _handle_signal(signum, frame):  # type: ignore[no-untyped-def]  # Reason: signal handlers use stdlib signature
        shutdown_event.set()
        previous = previous_handlers.get(signal.Signals(signum))
        if callable(previous):
            previous(signum, frame)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError, RuntimeError):
            continue

    try:
        yield
    finally:
        for sig, previous in previous_handlers.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, RuntimeError):
                continue


def _get_clerk_audience() -> list[str] | None:
    """Normalize optional Clerk audience config into a list."""
    audience = cast(str | None, getattr(config, "clerk_audience", None))
    if not audience:
        return None
    return [value.strip() for value in audience.split(",") if value.strip()]


async def open_frontend_after_startup() -> None:
    """Launch frontend pointing at local backend."""
    if not config.auto_open_browser:
        return

    frontend_url = config.frontend_url
    backend_override = os.getenv("BACKEND_API_URL", "localhost:8000")
    target_url = f"{frontend_url}?{urlencode({'backend': backend_override})}"

    # Small delay to let the server finish binding before opening the browser
    await asyncio.sleep(1)

    try:
        opened = webbrowser.open(target_url)
        if opened:
            logger.info("✅ Opened frontend at %s", target_url)
        else:
            logger.warning("⚠️  Could not auto-open browser")
            logger.info("📋 Visit http://localhost:8000 to connect")
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Browser opening is non-critical and should never crash server
        logger.warning("Failed to open browser: %s", exc)
        logger.info("📋 Visit http://localhost:8000 to connect")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("🚀 Starting Seer API server...")
    fastapi_app.state.shutdown_event = asyncio.Event()

    with _install_shutdown_signal_handlers(fastapi_app.state.shutdown_event):
        async with db_lifespan(fastapi_app):
            logger.info("✅ Database initialized")

            # Capture main event loop for cross-thread async operations
            # Import inside lifespan to ensure correct initialization order
            from seer.core.event_loop import set_main_event_loop  # pylint: disable=import-outside-toplevel  # Reason: must run during lifespan after async context is available
            set_main_event_loop(asyncio.get_running_loop())
            logger.info("✅ Main event loop captured for cross-thread scheduling")

            async with checkpointer_lifespan() as checkpointer:
                if checkpointer is not None:
                    fastapi_app.state.checkpointer = checkpointer
                logger.info("✅ Checkpointer initialized")

                trigger_status = "enabled – handled by Taskiq worker" if config.trigger_poller_enabled else "disabled via configuration"
                logger.info("Trigger poller %s", trigger_status)

                asyncio.create_task(open_frontend_after_startup())

                # Nest MCP lifespan if MCP is enabled
                # The MCP lifespan initializes the session manager for MCP transports
                if config.mcp_enabled:
                    mcp_lifespan_cm = fastapi_app.state.mcp_lifespan
                    async with mcp_lifespan_cm(fastapi_app):
                        logger.info("✅ MCP endpoints enabled at /sse and /mcp")
                        try:
                            yield
                        finally:
                            if hasattr(fastapi_app.state, "checkpointer"):
                                delattr(fastapi_app.state, "checkpointer")
                            from seer.services.browser.pool_manager import BrowserPoolManager  # pylint: disable=import-outside-toplevel  # Reason: shutdown import
                            await BrowserPoolManager.shutdown_instance()
                else:
                    try:
                        yield
                    finally:
                        if hasattr(fastapi_app.state, "checkpointer"):
                            delattr(fastapi_app.state, "checkpointer")
                        from seer.services.browser.pool_manager import BrowserPoolManager  # pylint: disable=import-outside-toplevel  # Reason: shutdown import
                        await BrowserPoolManager.shutdown_instance()

    # Shutdown PostHog client (flush pending events)
    if config.is_posthog_configured:
        from seer.observability.posthog_client import shutdown as posthog_shutdown  # pylint: disable=import-outside-toplevel  # Reason: conditional import during shutdown
        posthog_shutdown()

    # Flush Sentry events before shutdown
    if config.is_sentry_configured:
        sentry_flush(timeout=2.0)

    logger.info("👋 Seer API server shutting down...")


app = FastAPI(
    title="Seer LangGraph API",
    description="REST API for Seer multi-agent system",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(tools_router)
app.include_router(tracking_router)
app.include_router(byok_router)

# =============================================================================
# MCP Server Integration (setup - mount happens after all routes are defined)
# =============================================================================
# Create MCP app if enabled. The actual mount happens at the end of this file
# to ensure it doesn't catch requests meant for other routes.
if config.mcp_enabled:
    # Create combined MCP app that handles both SSE (/sse) and HTTP (/mcp) transports
    mcp_app, mcp_lifespan = create_combined_mcp_app()

    # Store in app.state for lifespan access
    app.state.mcp_app = mcp_app
    app.state.mcp_lifespan = mcp_lifespan
    logger.info("📡 MCP app created (will be mounted after routes)")

app.add_middleware(CorrelationMiddleware)

# Sentry context middleware - enrich errors with request context (must be after CorrelationMiddleware)
if config.is_sentry_configured:
    from seer.api.core.middleware.sentry_middleware import SentryContextMiddleware  # pylint: disable=wrong-import-position,ungrouped-imports # Reason: Conditional import after config check
    app.add_middleware(SentryContextMiddleware)
    logger.info("Sentry context middleware enabled")

# PostHog analytics middleware - track API requests (non-blocking)
if config.is_posthog_configured:
    from seer.api.core.middleware.posthog_middleware import PostHogMiddleware  # pylint: disable=wrong-import-position,ungrouped-imports # Reason: Conditional import after config check
    app.add_middleware(PostHogMiddleware)
    logger.info("📊 PostHog analytics middleware enabled")

# Usage limit middleware - enforce subscription limits centrally
# must be AFTER auth middleware to have user info
app.add_middleware(UsageLimitMiddleware)
logger.info("🔒 Usage limit middleware enabled")

# Organization context middleware - resolves org from persisted user context
# must be AFTER auth middleware to have db_user set
app.add_middleware(OrganizationContextMiddleware)
logger.info("🏢 Organization context middleware enabled")

# Authentication middleware - register BEFORE CORS to ensure user is set
logger.info("🔐 Using Clerk authentication")

app.add_middleware(
    ClerkAuthMiddleware,
    jwks_url=config.clerk_jwks_url,  # type: ignore[arg-type]  # Reason: Clerk credentials are required
    issuer=config.clerk_issuer,  # type: ignore[arg-type]  # Reason: Clerk credentials are required
    audience=_get_clerk_audience(),
)


# CORS middleware for development - must be AFTER auth middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# Session middleware - use Redis for multi-worker deployments, cookie for dev
# OAuth flows require proper cookie settings for cross-origin redirects
if config.session_backend == "redis":
    from seer.api.core.middleware.redis_sessions import RedisSessionMiddleware  # pylint: disable=ungrouped-imports # Reason: Conditional import based on config
    app.add_middleware(
        RedisSessionMiddleware,
        redis_url=config.redis_url,
        session_ttl=config.session_ttl_seconds,
        cookie_secure=config.redirect_uri_scheme == "https",
    )
    logger.info("Redis session middleware enabled (TTL=%ds)", config.session_ttl_seconds)
else:
    # Configure SessionMiddleware with OAuth-compatible settings
    # same_site="lax" allows cookies on top-level navigations (OAuth redirects)
    # max_age matches our default session TTL
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("SECRET_KEY", "dev_secret_key"),
        same_site="lax",
        max_age=config.session_ttl_seconds,
        https_only=config.redirect_uri_scheme == "https",
    )

# PyInstrument profiling middleware - writes HTML reports for inspection
if config.request_profiling_enabled:
    from seer.api.core.middleware.profiling import PyInstrumentMiddleware  # pylint: disable=ungrouped-imports # Reason: Conditional import after config check
    app.add_middleware(
        PyInstrumentMiddleware,
        enabled=config.request_profiling_enabled,
        output_dir=config.request_profiling_output_dir,
    )
    logger.info("🧪 PyInstrument profiling enabled; saving reports to %s", config.request_profiling_output_dir)

# Exception handler to ensure CORS headers on errors


@app.exception_handler(UsageLimitError)
# pylint: disable=unused-argument # Reason: FastAPI requires request parameter in exception handler signature
async def usage_limit_exception_handler(request: Request, exc: UsageLimitError):
    """
    Handle usage limit violations by returning 402 Payment Required with upgrade prompt.

    Returns structured error response with:
    - Current usage and limit values
    - User's tier
    - Upgrade URL
    - Clear error message
    """
    return JSONResponse(
        status_code=402,  # Payment Required
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are included and tracks errors."""
    error_logger = get_logger("api.main.errors")

    # Get correlation ID
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')

    # Log with correlation ID
    error_logger.error(
        "Unhandled exception: %s",
        exc,
        exc_info=True,
        extra={'correlation_id': correlation_id}
    )

    # Capture exception to Sentry with context
    if config.is_sentry_configured:
        from seer.observability.sentry_client import capture_exception, set_tag  # pylint: disable=import-outside-toplevel  # Reason: conditional import for error handling
        set_tag("correlation_id", correlation_id)
        capture_exception(exc)

    # Create error response with CORS headers
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

    # Add CORS headers manually
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response


# =============================================================================
# Health & Info Endpoints
# =============================================================================

# OAuth discovery endpoint for MCP clients (must be at well-known path)
if config.mcp_enabled:
    @app.get("/.well-known/oauth-protected-resource", tags=["OAuth"])
    async def oauth_discovery(request: Request):
        """
        OAuth Protected Resource Metadata for MCP clients.

        This endpoint tells MCP clients (like ChatGPT) where to authenticate.
        See: https://datatracker.ietf.org/doc/html/rfc9728
        """
        return await oauth_protected_resource_metadata(request)


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns server information including status, server name, and version.
    """
    return {
        "status": "ok",
        "server": "Seer API",
        "version": "1.0.0"
    }

@app.get("/sentry-debug")
async def trigger_error():
    _ = 1 / 0  # Intentionally trigger error for Sentry testing

@app.get("/", tags=["System"], include_in_schema=False)
async def root_redirect():
    """
    Root endpoint - redirects to frontend with backend URL configured.
    For API docs, visit /docs (Swagger UI) or /redoc.
    """
    # Get backend URL to pass to frontend
    backend_url = os.getenv("BACKEND_API_URL", "localhost:8000")

    # Get frontend URL from config
    frontend_url = config.frontend_url

    # Build redirect URL with backend parameter
    query_params = urlencode({"backend": backend_url})
    redirect_url = f"{frontend_url}?{query_params}"

    logger.info("Root path accessed, redirecting to: %s", redirect_url)

    return RedirectResponse(url=redirect_url, status_code=302)


# =============================================================================
# MCP Mount (must be AFTER all routes to avoid catching /health, /api/*, etc.)
# =============================================================================
# Mount MCP app at root - it has internal routes at /sse and /mcp
# This must be done last because a root mount acts as a catch-all
if config.mcp_enabled:
    app.mount("", mcp_app)
    logger.info("📡 MCP endpoints mounted at /sse and /mcp")


# =============================================================================
# Entry point for running directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
