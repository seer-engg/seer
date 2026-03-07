"""Database models for general chat sessions."""
from tortoise import fields, models

from seer.database.workflow_models import ChatExecutionStatus


class GeneralChatSession(models.Model):
    """General-purpose chat session (non-workflow)."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="general_chat_sessions")
    thread_id = fields.CharField(max_length=255, unique=True, db_index=True)
    title = fields.CharField(max_length=255, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    # Execution tracking (same pattern as WorkflowChatSession)
    current_execution_status = fields.CharEnumField(ChatExecutionStatus, max_length=20, null=True)
    current_execution_task_id = fields.CharField(max_length=255, null=True)
    current_execution_started_at = fields.DatetimeField(null=True)
    current_execution_finished_at = fields.DatetimeField(null=True)
    current_execution_error = fields.JSONField(null=True)

    class Meta:
        table = "general_chat_sessions"
        ordering = ("-updated_at",)

    def __str__(self) -> str:
        return f"GeneralChatSession<{self.id}:{self.thread_id}>"


class GeneralChatMessage(models.Model):
    """Message in a general chat session."""

    id = fields.IntField(primary_key=True)
    session = fields.ForeignKeyField("models.GeneralChatSession", related_name="messages")
    role = fields.CharField(max_length=20)  # 'user' or 'assistant'
    content = fields.TextField()
    model = fields.CharField(max_length=255, null=True)
    image_urls = fields.JSONField(null=True)
    thinking = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "general_chat_messages"
        ordering = ("created_at",)

    def __str__(self) -> str:
        return f"GeneralChatMessage<{self.role}:{self.content[:50]}>"
