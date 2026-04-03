"""BYOK-aware LLM factory. Single entry point for all LLM resolution."""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from seer.logger import get_logger

logger = get_logger(__name__)


def detect_byok_provider(model: str) -> str:
    """Infer the LLM provider from a model identifier.

    Used to match a model to the correct BYOK API key by provider.
    """
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model.startswith(("gemini-", "models/gemini-")):
        return "google"
    # Slash-prefixed models (e.g., "qwen/...", "mistralai/...") are typically OpenRouter
    if "/" in model:
        return "openrouter"
    return "openrouter"


async def resolve_llm(
    model: str,
    temperature: float = 0.2,
    organization_id: int | None = None,
) -> BaseChatModel:
    """
    BYOK-aware LLM factory. If org is on BYOK plan with active key,
    uses their credentials. Otherwise falls back to get_llm().
    """
    from seer.llm import get_llm, get_llm_for_byok  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    if organization_id:
        creds = await resolve_byok_credentials(organization_id, model=model)
        if creds:
            api_key, base_url, provider = creds
            return get_llm_for_byok(
                model=model,
                temperature=temperature,
                api_key=api_key,
                provider=provider,
                base_url=base_url,
            )

    return get_llm(model=model, temperature=temperature)


async def resolve_byok_credentials(
    organization_id: int,
    model: str | None = None,
) -> Optional[tuple[str, str, str]]:
    """
    Check if org has an active BYOK key for the given model's provider.

    Returns (api_key, base_url, provider) or None.

    Resolution order:
    1. Exact provider match (detected from model prefix)
    2. Fallback to any active key (backward compat for single-key orgs)
    """
    from seer.database.byok_models import LLMApiKey  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.services.byok.key_vault import get_key_vault  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    vault = get_key_vault()

    # Try exact provider match first
    if model:
        provider = detect_byok_provider(model)
        key = await LLMApiKey.get_or_none(
            organization_id=organization_id,
            provider=provider,
            is_active=True,
            status="active",
        )
        if key:
            decrypted = vault.decrypt(key.key_enc)
            if decrypted:
                logger.info("BYOK key resolved for org %s (provider=%s)", organization_id, provider)
                return decrypted, key.base_url, key.provider

    # Fallback: any active key (backward compat for single-key orgs)
    active_key = await LLMApiKey.get_or_none(
        organization_id=organization_id, is_active=True, status="active",
    )
    if not active_key:
        return None

    decrypted = vault.decrypt(active_key.key_enc)
    if not decrypted:
        return None

    logger.info("BYOK key resolved for org %s (fallback, provider=%s)", organization_id, active_key.provider)
    return decrypted, active_key.base_url, active_key.provider
