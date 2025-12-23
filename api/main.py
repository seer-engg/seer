"""
FastAPI server for Seer LangGraph agents.

Provides REST API endpoints for:
- Thread management (create, get state)
- Run execution with streaming

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 2024 --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import os
from shared.logger import get_logger
from shared.config import config
from api.router import router
from api.integrations.router import router as integrations_router
from api.tools.router import router as tools_router
from api.agents.checkpointer import checkpointer_lifespan
from shared.database import db_lifespan

# Import tools to register them
from shared.tools import gmail  # noqa: F401
# Note: model_block removed - use LLM block in workflows instead

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("🚀 Starting Seer API server...")
    
    async with db_lifespan(app):
        logger.info("✅ Database initialized")
        async with checkpointer_lifespan():
            logger.info("✅ Checkpointer initialized")
            yield
    
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
    from api.middleware.auth import ClerkAuthMiddleware
    
    app.add_middleware(
        ClerkAuthMiddleware,
        jwks_url=config.clerk_jwks_url,
        issuer=config.clerk_issuer,
        audience=config.clerk_audience.split(",") if config.clerk_audience else None,
        allow_unauthenticated_paths=[
            "/health",
            "/ok",
            "/info",
            "/api/integrations/gmail/callback",
            "/api/integrations/google_drive/callback",
            "/api/integrations/github/callback",
            "/api/integrations/asana/callback",
        ],
    )
else:
    from api.middleware.auth import TokenDecodeWithoutValidationMiddleware
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

# Exception handler to ensure CORS headers on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are included."""
    error_logger = get_logger("api.main.errors")
    error_logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
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
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ok", tags=["System"])
async def ok_check():
    """
    LangGraph Server health check endpoint.
    Required by @langchain/langgraph-sdk for server compatibility.
    Returns {"ok": true} as per LangGraph Server API specification.
    """
    return {"ok": True}


@app.get("/info", tags=["System"])
async def info():
    """
    LangGraph Server info endpoint.
    Used by frontend to verify server connectivity.
    Returns basic server information.
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

