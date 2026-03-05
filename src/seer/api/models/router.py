"""API router for model information endpoints."""
from typing import List

from fastapi import APIRouter

from .schema import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """
    List available models for the Chat Agent.

    Returns Kimi series models only.
    """
    return [
        ModelInfo(id="z-ai/glm-4.7", provider="openrouter", name="GLM 4.7", available=True),
        ModelInfo(id="moonshotai/kimi-k2.5", provider="openrouter", name="Kimi K2.5", available=True),
        ModelInfo(id="moonshotai/kimi-k2-thinking", provider="openrouter", name="Kimi K2 Thinking", available=True),
    ]
