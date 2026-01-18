from datetime import datetime, timezone
from enum import Enum

from tortoise import fields, models

WORKFLOW_ID_PREFIX = "wf_"
RUN_ID_PREFIX = "run_"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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


class WorkflowVersionStatus(str, Enum):
    DRAFT = "DRAFT"
    RELEASED = "RELEASED"
    ARCHIVED = "ARCHIVED"


class Workflow(models.Model):
    """New workflow entity that owns drafts, versions, and published state."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="workflows")
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    tags = fields.JSONField(null=True)
    meta = fields.JSONField(null=True)
    published_version = fields.ForeignKeyField(
        "models.WorkflowVersion",
        related_name="published_workflows",
        null=True,
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflows"
        ordering = ("-updated_at", "id")

    def __str__(self) -> str:
        return f"Workflow<{self.workflow_id}>"

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)


class WorkflowDraft(models.Model):
    """Mutable draft state for a workflow."""

    id = fields.IntField(primary_key=True)
    workflow = fields.OneToOneField(
        "models.Workflow", related_name="draft", on_delete=fields.CASCADE
    )
    spec = fields.JSONField()
    revision = fields.IntField(default=1)
    updated_at = fields.DatetimeField(auto_now=True)
    updated_by = fields.ForeignKeyField(
        "models.User",
        related_name="updated_workflow_drafts",
        null=True,
    )
    validation_errors = fields.JSONField(null=True)
    validation_warnings = fields.JSONField(null=True)

    class Meta:
        table = "workflow_drafts"

    def __str__(self) -> str:
        return f"WorkflowDraft<wf={self.id} rev={self.revision}>"

    @property
    def workflow_public_id(self) -> str:
        return self.workflow.workflow_id


class WorkflowVersion(models.Model):
    """Immutable runnable workflow version."""

    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField(
        "models.Workflow", related_name="versions", on_delete=fields.CASCADE
    )
    status = fields.CharEnumField(
        WorkflowVersionStatus,
        max_length=20,
        default=WorkflowVersionStatus.DRAFT,
    )
    spec = fields.JSONField()
    created_from_draft_revision = fields.IntField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    created_by = fields.ForeignKeyField(
        "models.User",
        related_name="created_workflow_versions",
        null=True,
    )
    manifest = fields.JSONField(null=True)
    spec_hash = fields.CharField(max_length=64)
    version_number = fields.IntField(default=0)

    class Meta:
        table = "workflow_versions"
        ordering = ("-created_at", "id")
        unique_together = (
            ("workflow_id", "version_number"),
        )

    def __str__(self) -> str:
        return f"WorkflowVersion<wf={self.id} status={self.status}>"

    @property
    def workflow_public_id(self) -> str:
        return self.workflow.workflow_id


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
        "models.Workflow", related_name="runs", null=True
    )
    workflow_version = fields.ForeignKeyField(
        "models.WorkflowVersion", related_name="runs", null=True
    )
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
    thread_id = fields.CharField(max_length=255, null=True)
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
    workflow = fields.ForeignKeyField("models.Workflow", related_name="chat_sessions")
    user = fields.ForeignKeyField("models.User", related_name="chat_sessions")
    thread_id = fields.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        description="LangGraph thread ID",
    )
    title = fields.CharField(
        max_length=255,
        null=True,
        description="Optional title for the session",
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflow_chat_sessions"
        ordering = ("-updated_at",)

    def __str__(self) -> str:
        return f"WorkflowChatSession<{make_workflow_public_id(self.id)}:{self.thread_id}>"

    @property
    def workflow_public_id(self) -> str:
        """Expose wf_* identifier used by public APIs."""
        return make_workflow_public_id(self.workflow.id)


class TriggerSubscription(models.Model):
    """Trigger configuration attached to a workflow."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="trigger_subscriptions", on_delete=fields.CASCADE)
    workflow = fields.ForeignKeyField(
        "models.Workflow", related_name="trigger_subscriptions", on_delete=fields.CASCADE
    )
    # Unique instance identifier (allows multiple triggers of same type per workflow)
    trigger_id = fields.CharField(max_length=255)
    # Trigger type identifier (e.g., "gmail_new_email", "webhook.github")
    trigger_key = fields.CharField(max_length=255)
    # Human-readable title for reference resolution (e.g., "Gmail_Inbox", "Webhook")
    title = fields.CharField(max_length=255, default="")
    provider_connection_id = fields.IntField(null=True)
    enabled = fields.BooleanField(default=True)
    is_polling = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    ## feild for webhook type triggers
    filters = fields.JSONField(null=True)
    provider_config = fields.JSONField(null=True)
    secret_token = fields.CharField(max_length=255, null=True)
    event_data_schema = fields.JSONField(null=True)


    ## feild for polling type triggers
    # NOTE: Adding/changing these poll_* fields requires a manual DB migration.
    poll_interval_seconds = fields.IntField(default=60)
    next_poll_at = fields.DatetimeField(
        default=_now_utc,
        description="Next scheduled poll time (UTC).",
    )
    poll_cursor_json = fields.JSONField(null=True)
    poll_status = fields.CharField(max_length=32, default="ok")
    poll_error_json = fields.JSONField(null=True)
    poll_backoff_seconds = fields.IntField(default=0)
    poll_lock_owner = fields.CharField(max_length=255, null=True)
    poll_lock_expires_at = fields.DatetimeField(null=True)



    class Meta:
        table = "trigger_subscriptions"
        indexes = (
            ("user_id", "workflow_id"),
            ("workflow_id", "trigger_id"),  # For trigger instance lookups
            ("trigger_key", "enabled"),     # For querying by trigger type
        )
        unique_together = (("workflow_id", "trigger_id"),)  # Ensure unique trigger IDs per workflow

    def __str__(self) -> str:
        return f"TriggerSubscription<{self.id}:{self.trigger_id}:{self.trigger_key}>"


class TriggerEvent(models.Model):
    """Normalized incoming trigger event."""

    id = fields.IntField(primary_key=True)
    trigger_key = fields.CharField(max_length=255)
    provider_connection_id = fields.IntField(null=True)
    provider_event_id = fields.CharField(max_length=255, null=True)
    # NOTE: Requires DB migration to add event_hash + supporting indexes.
    event_hash = fields.CharField(
        max_length=255,
        null=True,
        description="Deterministic hash used when provider_event_id is unavailable.",
    )
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
        unique_together = (
            ("trigger_key", "provider_connection_id", "provider_event_id"),
            ("trigger_key", "provider_connection_id", "event_hash"),
        )
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
    workflow = fields.ForeignKeyField("models.Workflow", related_name="proposals")
    session = fields.ForeignKeyField(
        "models.WorkflowChatSession",
        related_name="proposals",
        null=True,
    )
    created_by = fields.ForeignKeyField("models.User", related_name="workflow_proposals")
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
        return make_workflow_public_id(self.workflow.id)
