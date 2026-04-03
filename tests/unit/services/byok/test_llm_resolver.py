"""Tests for BYOK-aware LLM resolver."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.byok.llm_resolver import resolve_byok_credentials, resolve_llm


@pytest.fixture
def mock_byok_org():
    """Set up mocks for a BYOK org with active key."""
    with (
        patch("seer.database.subscription_models.BillingSubscription.get_or_none", new_callable=AsyncMock) as mock_sub_get,
        patch("seer.database.byok_models.LLMApiKey.get_or_none", new_callable=AsyncMock) as mock_key_get,
        patch("seer.services.byok.key_vault.get_key_vault") as mock_vault_fn,
    ):
        from seer.database.subscription_models import SubscriptionTier

        mock_sub = MagicMock()
        mock_sub.tier = SubscriptionTier.BYOK
        mock_sub_get.return_value = mock_sub

        mock_key = MagicMock()
        mock_key.key_enc = "encrypted_key"
        mock_key.base_url = "https://custom.provider.ai/v1"
        mock_key_get.return_value = mock_key

        mock_vault = MagicMock()
        mock_vault.decrypt.return_value = "sk-decrypted-key"
        mock_vault_fn.return_value = mock_vault

        yield {
            "sub_get": mock_sub_get,
            "key_get": mock_key_get,
            "vault": mock_vault,
        }


async def test_resolve_byok_credentials_returns_creds(mock_byok_org):
    result = await resolve_byok_credentials(organization_id=42)
    assert result == ("sk-decrypted-key", "https://custom.provider.ai/v1")


async def test_resolve_byok_credentials_no_subscription():
    with patch("seer.database.subscription_models.BillingSubscription.get_or_none", new_callable=AsyncMock, return_value=None):
        result = await resolve_byok_credentials(organization_id=1)
    assert result is None


async def test_resolve_llm_falls_back_without_byok():
    """Non-BYOK org → get_llm()."""
    with (
        patch("seer.database.subscription_models.BillingSubscription.get_or_none", new_callable=AsyncMock, return_value=None),
        patch("seer.llm.get_llm") as mock_get_llm,
    ):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = await resolve_llm(model="qwen/qwen3", organization_id=1)
        assert result is mock_llm
        mock_get_llm.assert_called_once_with(model="qwen/qwen3", temperature=0.2)


async def test_resolve_llm_uses_byok(mock_byok_org):
    """BYOK org → get_llm_for_byok()."""
    with patch("seer.llm.get_llm_for_byok") as mock_byok:
        mock_llm = MagicMock()
        mock_byok.return_value = mock_llm

        result = await resolve_llm(model="gpt-4", organization_id=42)
        assert result is mock_llm
        mock_byok.assert_called_once_with(
            model="gpt-4",
            temperature=0.2,
            api_key="sk-decrypted-key",
            base_url="https://custom.provider.ai/v1",
        )


async def test_resolve_llm_no_org_id():
    """No org_id → straight to get_llm()."""
    with patch("seer.llm.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = await resolve_llm(model="qwen/qwen3")
        assert result is mock_llm
