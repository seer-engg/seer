from __future__ import annotations

from tortoise import fields, models


class IntegrationResource(models.Model):
    """
    Persisted resource binding that hangs off an OAuth connection.

    Examples:
        - Supabase project selected after OAuth
        - Slack workspace / channel binding
        - GitHub repository linked to an installation
    """

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="integration_resources")
    oauth_connection = fields.ForeignKeyField(
        "models.OAuthConnection",
        related_name="resources",
        null=True,
    )
    provider = fields.CharField(max_length=50)
    resource_type = fields.CharField(max_length=50)
    resource_id = fields.CharField(max_length=255)
    resource_key = fields.CharField(max_length=255, null=True)
    name = fields.CharField(max_length=255, null=True)
    resource_metadata = fields.JSONField(null=True)
    status = fields.CharField(max_length=20, default="active")
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "integration_resources"
        indexes = (("user", "provider", "resource_type"),)
        unique_together = (("oauth_connection", "resource_type", "resource_id"),)

    def __str__(self) -> str:
        return f"{self.provider}:{self.resource_type}:{self.resource_id}"


class IntegrationSecret(models.Model):
    """
    Generic vault for non-OAuth credentials tied to a connection or resource.

    Stores encrypted values (value_enc) – encryption layer handled by callers.
    """

    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="integration_secrets")
    provider = fields.CharField(max_length=50)
    oauth_connection = fields.ForeignKeyField(
        "models.OAuthConnection",
        related_name="secrets",
        null=True,
    )
    resource = fields.ForeignKeyField(
        "models.IntegrationResource",
        related_name="secrets",
        null=True,
    )
    secret_type = fields.CharField(max_length=50)
    name = fields.CharField(max_length=100)
    value_enc = fields.TextField()
    value_fingerprint = fields.CharField(max_length=64, null=True)
    metadata = fields.JSONField(null=True)
    expires_at = fields.DatetimeField(null=True)
    status = fields.CharField(max_length=20, default="active")
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "integration_secrets"
        indexes = (("user", "provider", "secret_type"),)
        unique_together = (
            ("oauth_connection", "name"),
            ("resource", "name"),
        )

    def __str__(self) -> str:
        scope = "resource" if self.id else "connection"
        return f"{self.provider}:{self.name} ({scope})"
