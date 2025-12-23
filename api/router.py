"""API router for the Seer Agents API."""
from fastapi import APIRouter
from .agents.routes import router as agents_router
from .workflows.router import router as workflows_router
from .integrations.router import router as integrations_router
from .models.router import router as models_router

router = APIRouter(prefix="/api")
router.include_router(agents_router)
router.include_router(workflows_router)
router.include_router(integrations_router)
router.include_router(models_router)