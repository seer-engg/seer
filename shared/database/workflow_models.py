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

class Workflow(models.Model):
    """Main workflow entity."""
    
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    user = fields.ForeignKeyField('models.User', related_name='workflows')
    graph_data = fields.JSONField()  # ReactFlow nodes/edges JSON
    schema_version = fields.CharField(max_length=50, default="1.0")
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "workflows"
        ordering = ("-created_at",)
    
    def __str__(self) -> str:
        return f"Workflow<{self.name}>"


class WorkflowBlock(models.Model):
    """Individual blocks (nodes) in workflow."""
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.Workflow', related_name='blocks')
    block_id = fields.CharField(max_length=255)  # ReactFlow node ID
    block_type = fields.CharField(max_length=100)  # 'tool', 'code', 'llm', 'if_else', 'for_loop', 'input'
    block_config = fields.JSONField()  # Block-specific config
    python_code = fields.TextField(null=True)  # For code blocks
    position_x = fields.FloatField()
    position_y = fields.FloatField()
    oauth_scope = fields.CharField(max_length=255, null=True)  # From frontend
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "workflow_blocks"
        unique_together = (("workflow", "block_id"),)
    
    def __str__(self) -> str:
        return f"WorkflowBlock<{self.block_id}:{self.block_type}>"


class WorkflowEdge(models.Model):
    """Connections between blocks."""
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.Workflow', related_name='edges')
    source_block = fields.ForeignKeyField('models.WorkflowBlock', related_name='outgoing_edges')
    target_block = fields.ForeignKeyField('models.WorkflowBlock', related_name='incoming_edges')
    source_handle = fields.CharField(max_length=100, null=True)  # Output port
    target_handle = fields.CharField(max_length=100, null=True)  # Input port
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "workflow_edges"
    
    def __str__(self) -> str:
        return f"WorkflowEdge<{self.source_block.block_id}->{self.target_block.block_id}>"


class WorkflowExecution(models.Model):
    """Workflow execution history."""
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.Workflow', related_name='executions')
    user = fields.ForeignKeyField('models.User', related_name='workflow_executions')
    status = fields.CharField(max_length=50)  # 'running', 'completed', 'failed'
    input_data = fields.JSONField(null=True)
    output_data = fields.JSONField(null=True)
    error_message = fields.TextField(null=True)
    started_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)
    
    class Meta:
        table = "workflow_executions"
        ordering = ("-started_at",)
    
    def __str__(self) -> str:
        return f"WorkflowExecution<{self.workflow.name}:{self.status}>"


class BlockExecution(models.Model):
    """Per-block execution logs."""
    
    id = fields.IntField(primary_key=True)
    execution = fields.ForeignKeyField('models.WorkflowExecution', related_name='block_executions')
    block = fields.ForeignKeyField('models.WorkflowBlock', related_name='executions')
    status = fields.CharField(max_length=50)  # 'pending', 'running', 'completed', 'failed'
    input_data = fields.JSONField(null=True)
    output_data = fields.JSONField(null=True)
    error_message = fields.TextField(null=True)
    execution_time_ms = fields.IntField(null=True)  # Execution time in milliseconds
    started_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)
    
    class Meta:
        table = "block_executions"
        ordering = ("started_at",)
    
    def __str__(self) -> str:
        return f"BlockExecution<{self.block.block_id}:{self.status}>"


class WorkflowChatSession(models.Model):
    """Chat session for workflow assistant."""
    
    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField('models.Workflow', related_name='chat_sessions')
    user = fields.ForeignKeyField('models.User', related_name='chat_sessions')
    thread_id = fields.CharField(max_length=255, unique=True, db_index=True)  # LangGraph thread ID
    title = fields.CharField(max_length=255, null=True)  # Optional title for the session
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "workflow_chat_sessions"
        ordering = ("-updated_at",)
    
    def __str__(self) -> str:
        return f"WorkflowChatSession<{self.workflow.name}:{self.thread_id}>"


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
    workflow = fields.ForeignKeyField('models.Workflow', related_name='proposals')
    session = fields.ForeignKeyField('models.WorkflowChatSession', related_name='proposals', null=True)
    created_by = fields.ForeignKeyField('models.User', related_name='workflow_proposals')
    summary = fields.CharField(max_length=512)
    patch_ops = fields.JSONField()
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

