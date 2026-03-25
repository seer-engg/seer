"""Database models for WhatsApp integration."""
from tortoise import fields, models


class WhatsAppUserLink(models.Model):
    """Links a WhatsApp phone number to a Seer user."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="whatsapp_links")
    phone_number = fields.CharField(max_length=20, unique=True, db_index=True)
    verified = fields.BooleanField(default=False)
    verification_code = fields.CharField(max_length=10, null=True)
    verification_expires_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "whatsapp_user_links"

    def __str__(self) -> str:
        return f"WhatsAppUserLink<{self.phone_number}>"


class WhatsAppChatSession(models.Model):
    """Maps a WhatsApp phone number to a GeneralChatSession for continuity."""

    id = fields.IntField(primary_key=True)
    phone_number = fields.CharField(max_length=20, db_index=True)
    chat_session = fields.ForeignKeyField("models.GeneralChatSession", related_name="whatsapp_sessions")
    active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "whatsapp_chat_sessions"

    def __str__(self) -> str:
        return f"WhatsAppChatSession<{self.phone_number}:{self.chat_session_id}>"  # pylint: disable=no-member  # Reason: Tortoise ORM auto-creates _id attr for FK fields


class WhatsAppMessageLog(models.Model):
    """Tracks processed message IDs for idempotency."""

    id = fields.IntField(primary_key=True)
    message_id = fields.CharField(max_length=255, unique=True, db_index=True)
    processed_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "whatsapp_message_log"

    def __str__(self) -> str:
        return f"WhatsAppMessageLog<{self.message_id}>"
