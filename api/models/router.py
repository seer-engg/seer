"""API router for model information endpoints."""
from typing import List
from fastapi import APIRouter
from shared.config import config
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
    
    # Only OpenAI models - limited to gpt-5.2 and gpt-5-mini
    if config.openai_api_key:
        models.extend([
            ModelInfo(id="gpt-5.2", provider="openai", name="GPT-5.2", available=True),
            ModelInfo(id="gpt-5-mini", provider="openai", name="GPT-5 Mini", available=True),
        ])
    
    return models

