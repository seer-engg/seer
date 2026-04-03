"""API router for model information endpoints."""
from typing import List

from fastapi import APIRouter, Request

from seer.api.workflows.services.catalog import DEFAULT_MODEL_REGISTRY
from seer.core.registry.default_models import MODELS_BY_PROVIDER
from seer.logger import get_logger

from .schema import ModelInfo

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


def _provider_catalog_to_model_info(provider: str) -> List[ModelInfo]:
    """Get curated models for a specific BYOK provider."""
    models = MODELS_BY_PROVIDER.get(provider, ())
    return [
        ModelInfo(id=m.id, provider=provider, name=m.title, available=True)
        for m in models
    ]


async def _fetch_openai_compatible_models(api_key: str, base_url: str, provider: str) -> List[ModelInfo]:
    """Fetch models from an OpenAI-compatible /models endpoint."""
    import httpx  # pylint: disable=import-outside-toplevel  # Reason: Only needed for BYOK path

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
                    provider=provider,
                    name=m.get("name", m.get("id", "")),
                    available=True,
                )
                for m in data.get("data", [])
            ][:200]
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: BYOK model fetch failure falls back to curated catalog
        logger.warning("Failed to fetch BYOK models from %s: %s", base_url, e)
        return []


async def _get_byok_models(organization_id: int) -> List[ModelInfo] | None:
    """Build model list for BYOK orgs.

    Strategy: start with curated models for each active provider,
    then merge dynamic models from providers that support /models listing.
    Deduplicates by model id.
    """
    from seer.database.byok_models import LLMApiKey  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.services.byok.key_vault import get_key_vault  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    active_keys = await LLMApiKey.filter(
        organization_id=organization_id, is_active=True, status="active",
    )
    if not active_keys:
        return None

    vault = get_key_vault()
    seen_ids: set[str] = set()
    all_models: List[ModelInfo] = []

    def _add_unique(models: List[ModelInfo]) -> None:
        for m in models:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                all_models.append(m)

    for key in active_keys:
        api_key = vault.decrypt(key.key_enc)
        if not api_key:
            continue

        # Always include curated models for this provider
        _add_unique(_provider_catalog_to_model_info(key.provider))

        # Dynamic fetch only for openrouter/custom — known providers (openai,
        # anthropic, google) use curated lists to avoid overwhelming the user.
        if key.provider in ("openrouter", "custom"):
            base_url = key.base_url or "https://openrouter.ai/api/v1"
            dynamic = await _fetch_openai_compatible_models(api_key, base_url, key.provider)
            _add_unique(dynamic)

    return all_models if all_models else None


@router.get("", response_model=List[ModelInfo])
async def list_models(request: Request):
    """List available models.

    - BYOK orgs: curated models for their provider(s) + dynamic fetch
    - Non-BYOK: platform defaults only
    """
    user = getattr(request.state, "db_user", None)
    if user and user.active_organization_id:
        byok_models = await _get_byok_models(user.active_organization_id)
        if byok_models is not None:
            return byok_models

    # Non-BYOK: show platform defaults only
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
