"""
Pydantic models for trace API responses.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class TracePublic(BaseModel):
    """Public trace model for API responses."""
    thread_id: str = Field(..., description="LangGraph thread ID")
    checkpoint_id: Optional[str] = Field(None, description="Checkpoint ID")
    timestamp: datetime = Field(..., description="Checkpoint timestamp")
    trace_type: str = Field(..., description="Type of trace: 'workflow', 'chat', 'orchestrator'")
    status: str = Field(..., description="Trace status: 'running', 'completed', 'interrupted', 'failed'")
    node: Optional[str] = Field(None, description="Current or last executed node")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional trace metadata")
    message_count: Optional[int] = Field(None, description="Number of messages in trace")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")


class TraceDetail(BaseModel):
    """Detailed trace model with checkpoint history."""
    thread_id: str = Field(..., description="LangGraph thread ID")
    checkpoints: List[Dict[str, Any]] = Field(..., description="List of checkpoints in chronological order")
    current_state: Optional[Dict[str, Any]] = Field(None, description="Current state snapshot")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional trace metadata")


class TraceListResponse(BaseModel):
    """Response model for trace list endpoint."""
    traces: List[TracePublic] = Field(..., description="List of traces")
    total: int = Field(..., description="Total number of traces")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")
