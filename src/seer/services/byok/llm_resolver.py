"""BYOK-aware LLM factory. Single entry point for all LLM resolution."""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from seer.logger import get_logger

logger = get_logger(__name__)


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
        creds = await resolve_byok_credentials(organization_id)
        if creds:
            api_key, base_url = creds
            return get_llm_for_byok(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )

    return get_llm(model=model, temperature=temperature)


async def resolve_byok_credentials(
    organization_id: int,
) -> Optional[tuple[str, str]]:
    """
    Check if org is on BYOK plan with active key. Returns (api_key, base_url) or None.
    """
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
    decrypted = vault.decrypt(active_key.key_enc)
    if not decrypted:
        return None

    logger.info("BYOK key resolved for org %s", organization_id)
    return decrypted, active_key.base_url
