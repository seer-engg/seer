"""Database models for BYOK (Bring Your Own Key) LLM API key storage."""
from __future__ import annotations

from tortoise import fields, models


class LLMApiKey(models.Model):
    """
    Encrypted LLM API key storage for BYOK users.

    Org-scoped: each org has at most one active key per provider.
    Keys are Fernet-encrypted at rest via the key_vault service.
    The raw key is NEVER returned in API responses — only masked version.
    """

    id = fields.BigIntField(primary_key=True)
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="llm_api_keys",
        on_delete=fields.CASCADE,
    )
    created_by = fields.ForeignKeyField(
        "models.User",
        related_name="created_llm_api_keys",
        on_delete=fields.CASCADE,
    )
    label = fields.CharField(max_length=100)
    base_url = fields.CharField(max_length=500, default="https://openrouter.ai/api/v1")
    key_enc = fields.TextField()
    key_suffix = fields.CharField(max_length=8)
    provider = fields.CharField(max_length=50, default="openrouter")  # openrouter, openai, anthropic, google, custom
    provider_config = fields.JSONField(default={})
    is_active = fields.BooleanField(default=True)
    status = fields.CharField(max_length=20, default="active")  # active, revoked
    last_used_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "llm_api_keys"
        indexes = (("organization", "provider", "is_active"),)

    def __str__(self) -> str:
        return f"LLMApiKey<org={self.organization_id}, label={self.label}, suffix=...{self.key_suffix}>"  # pylint: disable=no-member # tortoise FK generates _id at runtime
