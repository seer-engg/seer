from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from tortoise import fields, models

from shared.database.models import User


WORKFLOW_ID_PREFIX = "wf_"
RUN_ID_PREFIX = "run_"


class WorkflowRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowRunSource(str, Enum):
    MANUAL = "manual"
    TRIGGER = "trigger"


class TriggerEventStatus(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    PROCESSED = "processed"
    FAILED = "failed"


def make_workflow_public_id(pk: int) -> str:
    return f"{WORKFLOW_ID_PREFIX}{pk}"


def parse_workflow_public_id(value: str) -> int:
    if not value.startswith(WORKFLOW_ID_PREFIX):
        raise ValueError("Invalid workflow_id format")
    return int(value.removeprefix(WORKFLOW_ID_PREFIX))


def make_run_public_id(pk: int) -> str:
    return f"{RUN_ID_PREFIX}{pk}"


def parse_run_public_id(value: str) -> int:
    if not value.startswith(RUN_ID_PREFIX):
        raise ValueError("Invalid run_id format")
    return int(value.removeprefix(RUN_ID_PREFIX))


class WorkflowRecord(models.Model):
    """Normalized workflow entity backed by WorkflowSpec JSON."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="workflow_records")
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    spec = fields.JSONField()
    version = fields.IntField(default=1)
    tags = fields.JSONField(null=True)
    meta = fields.JSONField(null=True)
    last_compile_ok = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflow_records"
        ordering = ("-updated_at", "id")

    def __str__(self) -> str:
        return f"WorkflowRecord<{self.name} v{self.version}>"

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)


class WorkflowRun(models.Model):
    """Persisted workflow run metadata (no telemetry)."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="workflow_runs")
    workflow = fields.ForeignKeyField(
        "models.WorkflowRecord", related_name="runs", null=True
    )
    workflow_version = fields.IntField(null=True)
    spec = fields.JSONField()
    inputs = fields.JSONField(null=True)
    config = fields.JSONField(null=True)
    source = fields.CharEnumField(
        WorkflowRunSource, max_length=20, default=WorkflowRunSource.MANUAL
    )
    subscription = fields.ForeignKeyField(
        "models.TriggerSubscription", related_name="runs", null=True
    )
    trigger_event = fields.ForeignKeyField(
        "models.TriggerEvent", related_name="runs", null=True
    )
    status = fields.CharEnumField(
        WorkflowRunStatus, max_length=20, default=WorkflowRunStatus.QUEUED
    )
    output = fields.JSONField(null=True)
    error = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    started_at = fields.DatetimeField(null=True)
    finished_at = fields.DatetimeField(null=True)
    metrics = fields.JSONField(null=True)

    class Meta:
        table = "workflow_runs"
        ordering = ("-created_at", "id")

    def __str__(self) -> str:
        return f"WorkflowRun<{self.run_id}:{self.status}>"

    @property
    def run_id(self) -> str:
        return make_run_public_id(self.id)



class WorkflowChatSession(models.Model):
    """Chat session for workflow assistant."""
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.WorkflowRecord', related_name='chat_sessions')
    user = fields.ForeignKeyField('models.User', related_name='chat_sessions')
    thread_id = fields.CharField(max_length=255, unique=True, db_index=True)  # LangGraph thread ID
    title = fields.CharField(max_length=255, null=True)  # Optional title for the session
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "workflow_chat_sessions"
        ordering = ("-updated_at",)
    
    def __str__(self) -> str:
        return f"WorkflowChatSession<{self.workflow_id}:{self.thread_id}>"
    
    @property
    def workflow_public_id(self) -> str:
        """Expose wf_* identifier used by public APIs."""
        return make_workflow_public_id(self.workflow_id)


class TriggerSubscription(models.Model):
    """Trigger configuration attached to a workflow."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="trigger_subscriptions")
    workflow = fields.ForeignKeyField(
        "models.WorkflowRecord", related_name="trigger_subscriptions"
    )
    trigger_key = fields.CharField(max_length=255)
    provider_connection_id = fields.IntField(null=True)
    enabled = fields.BooleanField(default=True)
    filters = fields.JSONField(null=True)
    bindings = fields.JSONField(null=True)
    provider_config = fields.JSONField(null=True)
    secret_token = fields.CharField(max_length=255, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "trigger_subscriptions"
        indexes = (
            ("user_id", "workflow_id"),
            ("trigger_key", "provider_connection_id", "enabled"),
        )

    def __str__(self) -> str:
        return f"TriggerSubscription<{self.id}:{self.trigger_key}>"


class TriggerEvent(models.Model):
    """Normalized incoming trigger event."""

    id = fields.IntField(primary_key=True)
    trigger_key = fields.CharField(max_length=255)
    provider_connection_id = fields.IntField(null=True)
    provider_event_id = fields.CharField(max_length=255, null=True)
    occurred_at = fields.DatetimeField(null=True)
    received_at = fields.DatetimeField(auto_now_add=True)
    event = fields.JSONField()
    raw_payload = fields.JSONField(null=True)
    status = fields.CharEnumField(
        TriggerEventStatus, max_length=20, default=TriggerEventStatus.RECEIVED
    )
    error = fields.JSONField(null=True)

    class Meta:
        table = "trigger_events"
        unique_together = (("trigger_key", "provider_connection_id", "provider_event_id"),)
        indexes = (
            ("status", "received_at"),
            ("trigger_key", "provider_connection_id"),
        )

    def __str__(self) -> str:
        return f"TriggerEvent<{self.id}:{self.trigger_key}>"


class WorkflowChatMessage(models.Model):
    """Individual message in a chat session."""
    
    id = fields.IntField(primary_key=True)
    session = fields.ForeignKeyField('models.WorkflowChatSession', related_name='messages')
    proposal = fields.OneToOneField('models.WorkflowProposal', related_name='message', null=True)
    role = fields.CharField(max_length=20)  # 'user' or 'assistant'
    content = fields.TextField()
    thinking = fields.TextField(null=True)  # Optional thinking/reasoning steps
    suggested_edits = fields.JSONField(null=True)  # Suggested workflow edits
    metadata = fields.JSONField(null=True)  # Additional metadata (model used, etc.)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "workflow_chat_messages"
        ordering = ("created_at",)
    
    def __str__(self) -> str:
        return f"WorkflowChatMessage<{self.role}:{self.content[:50]}>"


class WorkflowProposal(models.Model):
    """Reviewable workflow edit proposal."""
    
    STATUS_PENDING = "pending"
    STATUS_ACCEPTED = "accepted"
    STATUS_REJECTED = "rejected"
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.WorkflowRecord', related_name='proposals')
    session = fields.ForeignKeyField('models.WorkflowChatSession', related_name='proposals', null=True)
    created_by = fields.ForeignKeyField('models.User', related_name='workflow_proposals')
    summary = fields.CharField(max_length=512)
    spec = fields.JSONField()
    status = fields.CharField(max_length=20, default=STATUS_PENDING)
    preview_graph = fields.JSONField(null=True)
    applied_graph = fields.JSONField(null=True)
    metadata = fields.JSONField(null=True)
    decided_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "workflow_proposals"
        ordering = ("-created_at",)
    
    def __str__(self) -> str:
        return f"WorkflowProposal<{self.id}:{self.status}>"
    
    @property
    def workflow_public_id(self) -> str:
        """Expose wf_* identifier used by public APIs."""
        return make_workflow_public_id(self.workflow_id)

