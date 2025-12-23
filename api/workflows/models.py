"""
Database models for workflow system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from tortoise import fields, models


class Workflow(models.Model):
    """Main workflow entity."""
    
    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    user_id = fields.CharField(max_length=255, null=True, index=True)  # Optional: NULL for self-hosted, set for cloud
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
    user_id = fields.CharField(max_length=255, null=True, index=True)  # Optional: NULL for self-hosted, set for cloud
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


# ============================================================================
# Pydantic Models for API
# ============================================================================

class WorkflowBase(BaseModel):
    """Shared attributes for create/update."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    graph_data: Dict[str, Any] = Field(..., description="ReactFlow nodes/edges JSON")
    schema_version: str = Field(default="1.0", max_length=50)
    is_active: bool = True


class WorkflowCreate(WorkflowBase):
    """Payload for creating a workflow."""
    pass


class WorkflowUpdate(BaseModel):
    """Payload for updating a workflow."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    graph_data: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class WorkflowPublic(WorkflowBase):
    """Response model returned to API clients."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: Optional[str]  # Optional: NULL for self-hosted, set for cloud
    created_at: datetime
    updated_at: datetime


class WorkflowListResponse(BaseModel):
    """Wrapper response for list endpoints."""
    
    workflows: list[WorkflowPublic]


class WorkflowExecutionCreate(BaseModel):
    """Payload for creating a workflow execution."""
    
    input_data: Optional[Dict[str, Any]] = None
    stream: bool = Field(default=False, description="Stream execution events")


class WorkflowExecutionPublic(BaseModel):
    """Response model for workflow execution."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    workflow_id: int
    user_id: Optional[str]  # Optional: NULL for self-hosted, set for cloud
    status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


__all__ = [
    "Workflow",
    "WorkflowBlock",
    "WorkflowEdge",
    "WorkflowExecution",
    "BlockExecution",
    "WorkflowBase",
    "WorkflowCreate",
    "WorkflowUpdate",
    "WorkflowPublic",
    "WorkflowListResponse",
    "WorkflowExecutionCreate",
    "WorkflowExecutionPublic",
]

