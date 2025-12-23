"""
Pydantic schemas for workflow chat assistant.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class WorkflowEdit(BaseModel):
    """Represents a single edit operation on a workflow."""
    operation: str = Field(..., description="Operation type: 'add_block', 'modify_block', 'remove_block', 'add_edge', 'remove_edge'")
    block_id: Optional[str] = Field(None, description="Block ID for the operation")
    block_type: Optional[str] = Field(None, description="Block type (for add_block operation)")
    config: Optional[Dict[str, Any]] = Field(None, description="Block configuration (for add/modify operations)")
    position: Optional[Dict[str, float]] = Field(None, description="Block position {x, y} (for add_block)")
    source_id: Optional[str] = Field(None, description="Source block ID (for edge operations)")
    target_id: Optional[str] = Field(None, description="Target block ID (for edge operations)")
    source_handle: Optional[str] = Field(None, description="Source handle (for edge operations)")
    target_handle: Optional[str] = Field(None, description="Target handle (for edge operations)")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's chat message")
    workflow_state: Dict[str, Any] = Field(..., description="Current workflow state (nodes and edges)")
    model: Optional[str] = Field(default=None, description="Model to use for chat (e.g., 'gpt-5.2', 'claude-opus-4-5')")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's text response")
    suggested_edits: List[WorkflowEdit] = Field(default_factory=list, description="Suggested workflow edits")

