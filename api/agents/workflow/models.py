"""
Database models for workflow system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.database.models import UserPublic


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
    status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None



class WorkflowProposalPatchOp(BaseModel):
    """Single patch operation inside a proposal."""
    
    op: str
    description: Optional[str] = None
    node_id: Optional[str] = None
    node: Optional[Dict[str, Any]] = None
    edge_id: Optional[str] = None
    edge: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    target: Optional[str] = None


class WorkflowProposalPublic(BaseModel):
    """Response model for workflow proposals."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    workflow_id: int
    session_id: Optional[int] = None
    created_by: UserPublic
    summary: str
    status: str
    patch_ops: List[WorkflowProposalPatchOp]
    preview_graph: Optional[Dict[str, Any]] = None
    applied_graph: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    decided_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime




__all__ = [
    "WorkflowBase",
    "WorkflowCreate",
    "WorkflowUpdate",
    "WorkflowPublic",
    "WorkflowListResponse",
    "WorkflowExecutionCreate",
    "WorkflowExecutionPublic",
    "WorkflowProposalPublic",
    "WorkflowProposalPatchOp",
]

