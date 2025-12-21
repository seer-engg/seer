"""Pydantic models for the FastAPI server."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ThreadCreate(BaseModel):
    """Request model for creating a new thread."""
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to attach to the thread"
    )


class ThreadResponse(BaseModel):
    """Response model for thread operations."""
    thread_id: str = Field(..., description="Unique identifier for the thread")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class ThreadState(BaseModel):
    """Response model for thread state."""
    thread_id: str
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current state values"
    )
    next: List[str] = Field(
        default_factory=list,
        description="Next nodes to execute"
    )
    checkpoint_id: Optional[str] = None
    created_at: Optional[datetime] = None
    parent_checkpoint_id: Optional[str] = None


class RunInput(BaseModel):
    """Input model for creating a run."""
    graph_name: str = Field(
        default="eval_agent",
        description="Name of the graph to run (eval_agent, supervisor, codex)"
    )
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input values for the graph"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional configuration for the run"
    )
    stream_mode: Literal["values", "updates", "messages", "events"] = Field(
        default="updates",
        description="Stream mode: values (full state), updates (incremental), messages (LLM tokens), events (all events)"
    )


class RunStreamEvent(BaseModel):
    """Model for a single stream event."""
    event: str = Field(..., description="Event type (e.g., 'on_chain_start', 'on_chain_end')")
    name: Optional[str] = None
    data: Any = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_type: Optional[str] = None

