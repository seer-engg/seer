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
import webbrowser
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from seer.api.agents.checkpointer import checkpointer_lifespan
from seer.api.router import router
from seer.api.tools.router import router as tools_router
from seer.analytics import analytics
from seer.config import config
from seer.database import db_lifespan
from seer.logger import get_logger
from seer.observability.exceptions import ChatDisabledError, UsageLimitError

# Middleware order is important:
# Think of middleware as layers wrapping your core application (the route handler).
# The first middleware you add forms the innermost layer,
# while the last one added forms the outermost layer.

logger = get_logger("api.main")


async def initialize_tool_index(app: FastAPI) -> None:
    """Initialize tool index in background if enabled."""
    if not config.tool_index_auto_generate:
        return

    try:
        from seer.tool_hub.index_manager import ensure_tool_index_exists

        async def init_tool_index():
            try:
                toolhub = await ensure_tool_index_exists(
                    auto_generate=config.tool_index_auto_generate
                )
                if toolhub:
                    from seer.tool_hub.singleton import set_toolhub_instance
                    set_toolhub_instance(toolhub)
                    logger.info("✅ Tool index initialized")
                else:
                    logger.warning("⚠️ Tool index initialization skipped or failed")
            except Exception as e:
                logger.error("Error initializing tool index: %s", e, exc_info=True)

        task = asyncio.create_task(init_tool_index())
        app.state.tool_index_init_task = task
    except Exception as e:
        logger.warning("Could not initialize tool index: %s. Tool search may not work.", e)


async def open_frontend_after_startup() -> None:
    """Launch hosted frontend pointing at local backend for convenience."""
    if config.is_cloud_mode:
        return

    frontend_url = config.FRONTEND_URL
    backend_override = "localhost:8000"
    target_url = f"{frontend_url}?{urlencode({'backend': backend_override})}"

    # Small delay to let the server finish binding before opening the browser
    await asyncio.sleep(1)

    try:
        opened = webbrowser.open(target_url)
        if opened:
            logger.info("Opened frontend at %s", target_url)
        else:
            logger.warning("Could not open frontend automatically; url=%s", target_url)
    except Exception as exc:
        logger.warning("Failed to open frontend in browser: %s (url=%s)", exc, target_url, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("🚀 Starting Seer API server...")
    analytics.initialize()

    async with db_lifespan(app):
        logger.info("✅ Database initialized")
        async with checkpointer_lifespan() as checkpointer:
            if checkpointer is not None:
                app.state.checkpointer = checkpointer
            logger.info("✅ Checkpointer initialized")

            trigger_status = "enabled – handled by Taskiq worker" if config.trigger_poller_enabled else "disabled via configuration"
            logger.info("Trigger poller %s", trigger_status)

            await initialize_tool_index(app)
            asyncio.create_task(open_frontend_after_startup())

            try:
                yield
            finally:
                if hasattr(app.state, "checkpointer"):
                    delattr(app.state, "checkpointer")

    analytics.shutdown()
    logger.info("👋 Seer API server shutting down...")


app = FastAPI(
    title="Seer LangGraph API",
    description="REST API for Seer multi-agent system",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(tools_router)

# Correlation middleware - add correlation IDs to all requests
from seer.api.core.middleware.correlation import CorrelationMiddleware  # pylint: disable=wrong-import-position,ungrouped-imports # Reason: Import after app creation
app.add_middleware(CorrelationMiddleware)

# Usage limit middleware - enforce subscription limits centrally
# must be AFTER auth middleware to have user info
from seer.api.core.middleware.usage_limit import UsageLimitMiddleware  # pylint: disable=ungrouped-imports # Reason: Import after auth middleware setup
app.add_middleware(UsageLimitMiddleware)
logger.info("🔒 Usage limit middleware enabled")

# Authentication middleware - register BEFORE CORS to ensure user is set
if config.is_cloud_mode:
    if not config.is_clerk_configured:
        raise ValueError("Cloud mode requires Clerk configuration. Set CLERK_JWKS_URL and CLERK_ISSUER environment variables.")
    logger.info("🔐 Cloud mode: Using Clerk authentication")
    from seer.api.core.middleware.auth import ClerkAuthMiddleware  # pylint: disable=ungrouped-imports # Reason: Conditional import after cloud mode check

    app.add_middleware(
        ClerkAuthMiddleware,
        jwks_url=config.clerk_jwks_url,
        issuer=config.clerk_issuer,
        audience=config.clerk_audience.split(",") if config.clerk_audience else None,
    )
else:
    from seer.api.core.middleware.auth import TokenDecodeWithoutValidationMiddleware
    app.add_middleware(TokenDecodeWithoutValidationMiddleware)
    logger.info("🔧 Self-hosted mode: Authentication disabled")


# CORS middleware for development - must be AFTER auth middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev_secret_key"))

# PostHog analytics middleware - track requests and flush events
if config.is_posthog_configured:
    from seer.api.core.middleware.analytics import PostHogMiddleware
    app.add_middleware(PostHogMiddleware)
    logger.info("📊 PostHog analytics middleware enabled")

# Exception handler to ensure CORS headers on errors


@app.exception_handler(UsageLimitError)
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


@app.exception_handler(ChatDisabledError)
async def chat_disabled_exception_handler(request: Request, exc: ChatDisabledError):
    """
    Handle chat disabled errors (self-hosted mode) with 403 Forbidden.

    This is not an upgradeable limitation, so we return 403 instead of 402.
    """
    return JSONResponse(
        status_code=403,  # Forbidden
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are included and tracks errors."""
    error_logger = get_logger("api.main.errors")

    # Get correlation ID and user
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    user = getattr(request.state, 'user', None)
    distinct_id = user.user_id if user else f"anonymous_{request.client.host if request.client else 'unknown'}"

    # Log with correlation ID
    error_logger.error(
        "Unhandled exception: %s",
        exc,
        exc_info=True,
        extra={'correlation_id': correlation_id}
    )

    # Track error to PostHog
    analytics.capture_error(
        distinct_id=distinct_id,
        error=exc,
        context={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
        },
        error_location="global_exception_handler",
    )

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

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns server information including status, server name, and version.
    """
    return {
        "status": "ok",
        "server": "Seer LangGraph API",
        "version": "1.0.0"
    }


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
