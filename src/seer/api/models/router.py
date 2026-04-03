"""API router for model information endpoints."""
from typing import List

from fastapi import APIRouter, Request

from seer.api.workflows.services.catalog import DEFAULT_MODEL_REGISTRY
from seer.logger import get_logger

from .schema import ModelInfo

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


async def _get_byok_models(organization_id: int) -> List[ModelInfo] | None:
    """Fetch models from BYOK provider. Returns None if org is not on BYOK plan."""
    from seer.database.byok_models import LLMApiKey  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.database.subscription_models import BillingSubscription, SubscriptionTier  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.services.byok.key_vault import get_key_vault  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    org_sub = await BillingSubscription.get_or_none(organization_id=organization_id)
    if not org_sub or org_sub.tier != SubscriptionTier.BYOK:
        return None

    active_key = await LLMApiKey.get_or_none(
        organization_id=organization_id, is_active=True, status="active",
    )
    if not active_key:
        return None

    vault = get_key_vault()
    api_key = vault.decrypt(active_key.key_enc)
    if not api_key:
        return None

    import httpx  # pylint: disable=import-outside-toplevel  # Reason: Only needed for BYOK path
    base_url = active_key.base_url or "https://openrouter.ai/api/v1"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(
                    id=m.get("id", ""),
                    provider="openrouter",
                    name=m.get("name", m.get("id", "")),
                    available=True,
                )
                for m in data.get("data", [])
            ][:200]
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: BYOK model fetch failure falls back to defaults
        logger.warning("Failed to fetch BYOK models: %s", e)
        return None


@router.get("", response_model=List[ModelInfo])
async def list_models(request: Request):
    """List available models for the Chat Agent. BYOK orgs get provider models."""
    user = getattr(request.state, "db_user", None)
    if user and user.active_organization_id:
        byok_models = await _get_byok_models(user.active_organization_id)
        if byok_models is not None:
            return byok_models

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
