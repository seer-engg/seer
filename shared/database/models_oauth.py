from tortoise import fields, models
from shared.database.models import User

class OAuthConnection(models.Model):
    """
    Database model for storing OAuth connections/tokens.
    """
    id = fields.BigIntField(pk=True)
    user = fields.ForeignKeyField("models.User", related_name="oauth_connections")
    provider = fields.CharField(max_length=50)  # google, github, etc.
    provider_account_id = fields.CharField(max_length=255)  # e.g. google sub/email, github user id
    
    # Encrypted tokens - for now storing as text, but should be encrypted in practice
    # In a real app, use a Fernet field or similar encryption at app level before saving
    access_token_enc = fields.TextField()
    refresh_token_enc = fields.TextField(null=True)
    
    expires_at = fields.DatetimeField(null=True)
    scopes = fields.TextField(null=True)  # JSON or space-separated string
    token_type = fields.CharField(max_length=50, default="Bearer")
    
    # Metadata: team_id, instance_url, profile info etc.
    provider_metadata = fields.JSONField(null=True)
    
    status = fields.CharField(max_length=20, default="active")  # active, revoked, error
    
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "oauth_connections"
        unique_together = (("user", "provider", "provider_account_id"),)

    def __str__(self) -> str:
        return f"{self.provider}:{self.provider_account_id} ({self.user})"

