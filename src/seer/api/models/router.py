"""API router for model information endpoints."""
from typing import List

from fastapi import APIRouter

from seer.config import config

from .schema import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """
    List available models based on configured API keys.

    Returns a list of models that are available based on which API keys
    are configured in the environment.
    """
    models = []

    if config.anthropic_api_key:
        models.extend([
            ModelInfo(id="claude-sonnet-4.5", provider="anthropic", name="Claude Sonnet 4.5", available=True),
            ModelInfo(id="claude-opus-4.5", provider="anthropic", name="Claude Opus 4.5", available=True),
        ])

    if config.openrouter_api_key:
        models.extend([
            ModelInfo(id="moonshotai/kimi-k2.5", provider="openrouter", name="Kimi K2.5", available=True),
            ModelInfo(id="moonshotai/kimi-k2-thinking", provider="openrouter", name="Kimi K2 Thinking", available=True),
        ])

    return models
