"""
FastAPI server for Seer LangGraph agents.

Provides REST API endpoints for:
- Thread management (create, get state)
- Run execution with streaming

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 2024 --reload
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from api.agents.checkpointer import checkpointer_lifespan
from api.router import router
from api.tools.router import router as tools_router
from shared.analytics import analytics
from shared.config import config
from shared.database import db_lifespan
from shared.logger import get_logger

# Import tools to register them
# Note: model_block removed - use LLM block in workflows instead

logger = get_logger("api.main")


async def initialize_tool_index(app: FastAPI) -> None:
    """Initialize tool index in background if enabled."""
    if not config.tool_index_auto_generate:
        return

    try:
        import asyncio

        from shared.tool_hub.index_manager import ensure_tool_index_exists

        async def init_tool_index():
            try:
                toolhub = await ensure_tool_index_exists(
                    auto_generate=config.tool_index_auto_generate
                )
                if toolhub:
                    from shared.tool_hub.singleton import set_toolhub_instance
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

# Authentication middleware - register BEFORE CORS to ensure user is set
if config.is_cloud_mode:
    if not config.is_clerk_configured:
        raise ValueError("Cloud mode requires Clerk configuration. Set CLERK_JWKS_URL and CLERK_ISSUER environment variables.")
    logger.info("🔐 Cloud mode: Using Clerk authentication")
    from api.core.middleware.auth import ClerkAuthMiddleware  # pylint: disable=ungrouped-imports # Reason: Conditional import after cloud mode check

    app.add_middleware(
        ClerkAuthMiddleware,
        jwks_url=config.clerk_jwks_url,
        issuer=config.clerk_issuer,
        audience=config.clerk_audience.split(",") if config.clerk_audience else None,
        allow_unauthenticated_paths=[
            "/health",
            "/api/integrations/google/callback",
            "/api/integrations/github/callback",
            "/api/integrations/asana/callback",
            "/api/v1/webhooks",
            "/api/subscriptions/webhooks/stripe"
        ],
    )
else:
    from api.core.middleware.auth import TokenDecodeWithoutValidationMiddleware
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
    from api.core.middleware.analytics import PostHogMiddleware
    app.add_middleware(PostHogMiddleware)
    logger.info("📊 PostHog analytics middleware enabled")

# Exception handler to ensure CORS headers on errors


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are included."""
    error_logger = get_logger("api.main.errors")
    error_logger.error("Unhandled exception: %s", exc, exc_info=True)

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
