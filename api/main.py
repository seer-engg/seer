"""
FastAPI server for Seer LangGraph agents.

Provides REST API endpoints for:
- Thread management (create, get state)
- Run execution with streaming

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 2024 --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import os
from shared.logger import get_logger
from api.router import router
from api.integrations.router import router as integrations_router
from api.agents.checkpointer import checkpointer_lifespan
from shared.database import db_lifespan
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
app.include_router(integrations_router)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev_secret_key"))


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}




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

