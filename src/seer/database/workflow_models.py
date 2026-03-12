# pylint: disable=too-many-lines  # Reason: workflow ORM models, version history, run tracking, and status enums all belong together
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
    INTERRUPTED = "interrupted"  # HITL: workflow paused waiting for user input


class WorkflowRunSource(str, Enum):
    MANUAL = "manual"
    TRIGGER = "trigger"


class TriggerEventStatus(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    PROCESSED = "processed"
    FAILED = "failed"


class ChatExecutionStatus(str, Enum):
    """Status of chat agent execution."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class WorkflowCreationMode(str, Enum):
    AUTO_CREATE = "AUTO_CREATE"
    ASK_FIRST = "ASK_FIRST"
    ON_ACCEPTANCE = "ON_ACCEPTANCE"


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


class WorkflowVisibility(str, Enum):
    """Visibility of workflows within an organization."""
    PRIVATE = "private"    # Only creator can see
    TEAM = "team"          # All org members can see
    ASSIGNED = "assigned"  # Only assigned users can see (for consultants)


class WorkflowApprovalStatus(str, Enum):
    """Approval status for consultant-created workflows."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Workflow(models.Model):
    """New workflow entity that owns drafts, versions, and published state."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="workflows")
    name = fields.CharField(max_length=255)

    # Organization ownership (replaces user-only ownership)
    # Nullable initially for migration compatibility
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="workflows",
        null=True,
        on_delete=fields.CASCADE,
    )

    # Approval status for consultant-created workflows
    approval_status = fields.CharEnumField(
        WorkflowApprovalStatus,
        max_length=20,
        null=True,
        description="Approval status for consultant-created workflows",
    )

    # Workflow visibility within org
    visibility = fields.CharEnumField(
        WorkflowVisibility,
        max_length=20,
        default=WorkflowVisibility.TEAM,
        description="Who can see this workflow within the organization",
    )

    is_published = fields.BooleanField(default=False, description="Whether this workflow is published as a template")

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflows"
        ordering = ("-updated_at", "id")
        indexes = (
            ("organization_id",),
            ("user_id", "organization_id"),
            ("visibility",),
        )

    def __str__(self) -> str:
        return f"Workflow<{self.workflow_id}>"

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)




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
    # New fields for draft functionality
    updated_at = fields.DatetimeField(auto_now=True)
    updated_by = fields.ForeignKeyField(
        "models.User",
        related_name="updated_workflow_versions",
        null=True,
    )
    validation_errors = fields.JSONField(null=True)
    validation_warnings = fields.JSONField(null=True)

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
    node_traces = fields.JSONField(null=True)  # Per-node execution traces (including error traces for failed nodes)
    thread_id = fields.CharField(max_length=255, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    started_at = fields.DatetimeField(null=True)
    finished_at = fields.DatetimeField(null=True)
    metrics = fields.JSONField(null=True)

    # HITL interrupt fields
    pending_interrupt_node_id = fields.CharField(
        max_length=255,
        null=True,
        description="Node ID where workflow is waiting for user input",
    )
    pending_interrupt_data = fields.JSONField(
        null=True,
        description="HITL interrupt payload waiting for user response",
    )
    interrupt_expires_at = fields.DatetimeField(
        null=True,
        description="When the HITL interrupt times out (null = indefinite)",
    )

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
    current_workflow_state = fields.JSONField(
        null=True,
        description="Current workflow graph state (nodes and edges) for this session",
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    # Execution tracking fields
    current_execution_status = fields.CharEnumField(
        ChatExecutionStatus,
        max_length=20,
        null=True,
        description="Status of current chat execution"
    )
    current_execution_task_id = fields.CharField(
        max_length=255,
        null=True,
        description="Taskiq task ID for current execution"
    )
    current_execution_started_at = fields.DatetimeField(null=True)
    current_execution_finished_at = fields.DatetimeField(null=True)
    current_execution_error = fields.JSONField(null=True)

    # Interrupt handling fields
    pending_interrupt_type = fields.CharField(
        max_length=50,
        null=True,
        description="Type of pending interrupt (e.g., 'clarification_question')"
    )
    pending_interrupt_data = fields.JSONField(
        null=True,
        description="Interrupt data waiting for user response"
    )

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
    secret_token = fields.CharField(max_length=255, null=True)  # Deprecated: use webhook_slug
    webhook_slug = fields.CharField(max_length=64, null=True, unique=True, db_index=True)
    event_data_schema = fields.JSONField(null=True)

    ## fields for form type triggers
    form_suffix = fields.CharField(max_length=255, null=True)
    form_fields = fields.JSONField(null=True)
    form_config = fields.JSONField(null=True)


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
    subscription_id = fields.IntField(null=True)

    class Meta:
        table = "trigger_events"
        unique_together = (
            ("subscription_id", "trigger_key", "provider_connection_id", "provider_event_id"),
            ("subscription_id", "trigger_key", "provider_connection_id", "event_hash"),
        )
        indexes = (
            ("status", "received_at"),
            ("trigger_key", "provider_connection_id"),
            ("subscription_id",),
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
    thread_id = fields.CharField(
        max_length=255,
        null=True,
        db_index=True,
        description="Thread ID for lookup during agent execution",
    )
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


class WorkflowDiscoveryChatSession(models.Model):
    """Discovery chat session before workflow creation."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="discovery_chat_sessions")
    thread_id = fields.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        description="LangGraph thread ID for discovery session",
    )
    title = fields.CharField(
        max_length=255,
        null=True,
        description="Optional title for discovery session",
    )
    workflow_creation_mode = fields.CharEnumField(
        WorkflowCreationMode,
        max_length=20,
        default=WorkflowCreationMode.ASK_FIRST,
        description="How workflow should be created",
    )
    created_workflow = fields.ForeignKeyField(
        "models.Workflow",
        related_name="discovery_sessions",
        null=True,
        description="Workflow created from this discovery session",
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflow_discovery_chat_sessions"
        ordering = ("-updated_at",)
        indexes = (
            ("user_id",),
            ("thread_id",),
        )

    def __str__(self) -> str:
        return f"WorkflowDiscoveryChatSession<{self.id}:{self.thread_id}>"


class GlobalVariable(models.Model):
    """Organization-scoped key-value variable reusable across workflows."""

    id = fields.IntField(primary_key=True)
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="global_variables",
        on_delete=fields.CASCADE,
    )
    key = fields.CharField(max_length=255)
    value = fields.TextField()
    is_secret = fields.BooleanField(default=False)
    description = fields.TextField(null=True)
    created_by = fields.ForeignKeyField(
        "models.User",
        related_name="created_global_variables",
        on_delete=fields.CASCADE,
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "global_variables"
        ordering = ("key",)
        unique_together = (("organization_id", "key"),)
        indexes = (
            ("organization_id",),
        )

    def __str__(self) -> str:
        return f"GlobalVariable<{self.key}>"


class WorkflowFile(models.Model):
    """
    Tracks files created during workflow execution or uploaded by users.

    Files are stored in S3-compatible storage (S3/R2) and this table tracks
    metadata for management, debugging, and API access.
    """

    id = fields.IntField(primary_key=True)
    file_id = fields.CharField(
        max_length=36,
        unique=True,
        db_index=True,
        description="UUID for the file (used in file references)",
    )
    user = fields.ForeignKeyField(
        "models.User",
        related_name="files",
        on_delete=fields.CASCADE,
        description="User who owns this file",
    )
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="workflow_files",
        on_delete=fields.CASCADE,
        null=True,
        description="Organization this file belongs to (for team access)",
    )
    workflow_run = fields.ForeignKeyField(
        "models.WorkflowRun",
        related_name="files",
        on_delete=fields.CASCADE,
        null=True,
        description="Workflow run that created this file (null for user uploads)",
    )
    storage_path = fields.CharField(
        max_length=512,
        description="Full storage path (e.g., s3://bucket/path/to/file)",
    )
    filename = fields.CharField(
        max_length=255,
        description="Original filename",
    )
    mime_type = fields.CharField(
        max_length=128,
        description="MIME type of the file",
    )
    size_bytes = fields.BigIntField(
        description="File size in bytes",
    )
    md5_hash = fields.CharField(
        max_length=32,
        null=True,
        description="MD5 hash for integrity verification",
    )
    source_node_id = fields.CharField(
        max_length=255,
        null=True,
        description="Node ID that created this file",
    )
    source_tool = fields.CharField(
        max_length=255,
        null=True,
        description="Tool name that created this file (e.g., google_drive_download_file)",
    )
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "workflow_files"
        ordering = ("-created_at",)
        indexes = (
            ("workflow_run_id",),
            ("file_id",),
            ("user_id",),
            ("organization_id",),
            ("mime_type",),
            ("source_tool",),
            ("created_at",),
            ("size_bytes",),
        )

    def __str__(self) -> str:
        return f"WorkflowFile<{self.file_id}:{self.filename}>"

    @property
    def size_human(self) -> str:
        """Get human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
