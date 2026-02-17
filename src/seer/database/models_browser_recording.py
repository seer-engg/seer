"""Session recording models for browser replay and observability."""
from __future__ import annotations

from tortoise import fields, models


class SessionRecording(models.Model):
    """
    Stores rrweb session recordings for browser sessions.

    Events are gzip-compressed JSON stored as binary. Supports both
    interactive login sessions and workflow execution recordings.
    """

    id = fields.UUIDField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="session_recordings")
    browser_profile = fields.ForeignKeyField(
        "models.BrowserProfile", related_name="recordings", null=True
    )
    workflow_run_id = fields.CharField(max_length=64, null=True, index=True)
    session_type = fields.CharField(max_length=20)  # "interactive" | "workflow"

    # Compressed rrweb event data
    events_compressed = fields.BinaryField()  # gzip(json(rrweb_events))
    event_count = fields.IntField(default=0)
    duration_ms = fields.IntField(default=0)
    compressed_size_bytes = fields.IntField(default=0)

    start_url = fields.CharField(max_length=2048, null=True)
    status = fields.CharField(max_length=20, default="recording")  # recording, completed, failed

    created_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)

    class Meta:
        table = "session_recordings"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Recording {self.id} ({self.session_type}, {self.status})"
