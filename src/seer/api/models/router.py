"""API router for model information endpoints."""
from typing import List

from fastapi import APIRouter

from seer.api.workflows.services.catalog import DEFAULT_MODEL_REGISTRY

from .schema import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """List available models for the Chat Agent."""
    return [
        ModelInfo(id=m.id, provider="openrouter", name=m.title, available=True)
        for m in DEFAULT_MODEL_REGISTRY
    ]


@router.get("/image", response_model=List[ModelInfo])
async def list_image_models():
    """List available models for image generation."""
    return [
        ModelInfo(id="google/gemini-3.1-flash-image-preview", provider="openrouter", name="Gemini 3.1 Flash Image", available=True),
        ModelInfo(id="sourceful/riverflow-v2-pro", provider="openrouter", name="Riverflow V2 Pro", available=True),
        ModelInfo(id="sourceful/riverflow-v2-fast", provider="openrouter", name="Riverflow V2 Fast", available=True),
        ModelInfo(id="black-forest-labs/flux.2-klein-4b", provider="openrouter", name="FLUX.2 Klein", available=True),
        ModelInfo(id="bytedance-seed/seedream-4.5", provider="openrouter", name="Seedream 4.5", available=True),
        ModelInfo(id="black-forest-labs/flux.2-flex", provider="openrouter", name="FLUX.2 Flex", available=True),
    ]
