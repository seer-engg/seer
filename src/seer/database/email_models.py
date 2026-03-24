"""Database models for email analytics tracking."""
from __future__ import annotations

from uuid import uuid4

from tortoise import fields, models


class EmailEvent(models.Model):
    """
    Tracks email send and engagement events (opens, clicks).

    Each email send creates a row with event_type="sent" and a unique tracking_id.
    The tracking_id is embedded in tracking pixels and rewritten links.
    Subsequent opens/clicks create additional rows sharing the same tracking_id.
    """

    id = fields.UUIDField(primary_key=True, default=uuid4)
    tracking_id = fields.UUIDField(db_index=True)
    provider = fields.CharField(max_length=20)  # "resend" | "gmail"
    email_type = fields.CharField(max_length=50)  # "invitation" | "approval" | "workflow" | "hitl"
    event_type = fields.CharField(max_length=30, default="sent")  # "sent" | "opened" | "clicked"
    recipient = fields.CharField(max_length=320)
    subject = fields.CharField(max_length=500, null=True)
    provider_email_id = fields.CharField(max_length=255, null=True)
    organization_id = fields.IntField(null=True, db_index=True)
    workflow_run_id = fields.CharField(max_length=255, null=True, db_index=True)
    user_id = fields.CharField(max_length=255, null=True)
    link_url = fields.TextField(null=True)
    metadata_json = fields.JSONField(null=True)  # pylint: disable=invalid-name  # Reason: avoid collision with Tortoise internal .metadata
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "email_events"
