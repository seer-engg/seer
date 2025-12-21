"""API router for the Seer Agents API."""
from fastapi import APIRouter
from .agents.routes import router as agents_router

router = APIRouter()
router.include_router(agents_router)