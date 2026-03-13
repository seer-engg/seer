"""Session recording models for browser replay and observability."""
from __future__ import annotations

from tortoise import fields, models


class SessionRecording(models.Model):
    """
    Stores rrweb session recordings for browser sessions.

    Events are gzip-compressed JSON stored as binary. Supports both
    interactive login sessions and workflow execution recordings.

    For long-running sessions, events may be stored in chunks (SessionRecordingChunk)
    with periodic flushing to prevent data loss on CDP target detachment.
    When is_chunked=True, events_compressed is null and events are in chunks.
    """

    id = fields.UUIDField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="session_recordings")
    browser_profile = fields.ForeignKeyField(
        "models.BrowserProfile", related_name="recordings", null=True
    )
    workflow_run_id = fields.CharField(max_length=64, null=True, index=True)
    session_type = fields.CharField(max_length=20)  # "interactive" | "workflow"

    # Compressed rrweb event data (null when is_chunked=True)
    events_compressed = fields.BinaryField(null=True)  # gzip(json(rrweb_events))
    event_count = fields.IntField(default=0)
    duration_ms = fields.IntField(default=0)
    compressed_size_bytes = fields.IntField(default=0)

    # Chunked recording support
    is_chunked = fields.BooleanField(default=False)
    chunk_count = fields.IntField(default=0)
    last_flush_at = fields.DatetimeField(null=True)

    start_url = fields.CharField(max_length=2048, null=True)
    status = fields.CharField(max_length=20, default="recording")  # recording, completed, failed

    created_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)

    class Meta:
        table = "session_recordings"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Recording {self.id} ({self.session_type}, {self.status})"


class SessionRecordingChunk(models.Model):
    """
    Stores incremental chunks of rrweb events flushed periodically.

    Chunks are flushed every ~45 seconds during recording to prevent
    data loss when CDP target detaches mid-session. On playback, chunks
    are reassembled in sequence_number order.
    """

    id = fields.UUIDField(primary_key=True)
    recording = fields.ForeignKeyField(
        "models.SessionRecording",
        related_name="chunks",
        on_delete=fields.CASCADE,
    )
    sequence_number = fields.IntField()  # Order for reassembly (0, 1, 2, ...)
    events_compressed = fields.BinaryField()  # gzip(json(events_slice))
    event_count = fields.IntField(default=0)
    compressed_size_bytes = fields.IntField(default=0)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "session_recording_chunks"
        ordering = ["sequence_number"]
        unique_together = [("recording", "sequence_number")]

    def __str__(self) -> str:
        return f"Chunk {self.sequence_number} of {self.recording_id}"  # pylint: disable=no-member  # Reason: Tortoise ORM generates _id accessor for FK fields dynamically
