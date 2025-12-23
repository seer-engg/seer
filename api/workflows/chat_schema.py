"""
Pydantic schemas for workflow chat assistant.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from shared.database.models import UserPublic


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
    session_id: Optional[int] = Field(default=None, description="Chat session ID to resume conversation")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID to resume conversation")
    resume_thread: bool = Field(default=True, description="Whether to resume existing thread if thread_id provided")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's text response")
    suggested_edits: List[WorkflowEdit] = Field(default_factory=list, description="Suggested workflow edits")
    session_id: Optional[int] = Field(default=None, description="Chat session ID")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID")
    thinking: Optional[List[str]] = Field(default=None, description="Agent thinking/reasoning steps (collapsible)")
    interrupt_required: bool = Field(default=False, description="Whether human input is required (human-in-the-loop)")
    interrupt_data: Optional[Dict[str, Any]] = Field(default=None, description="Data for human-in-the-loop interrupt")


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""
    title: Optional[str] = Field(default=None, description="Optional title for the session")


class ChatSession(BaseModel):
    """Response model for chat session."""
    id: int
    workflow_id: int
    user: UserPublic
    thread_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    """Response model for chat message."""
    id: int
    session_id: int
    role: str
    content: str
    thinking: Optional[str] = None
    suggested_edits: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class ChatSessionWithMessages(ChatSession):
    """Chat session with messages."""
    messages: List[ChatMessage] = Field(default_factory=list)


class InterruptResponse(BaseModel):
    """Response model for human-in-the-loop interrupt."""
    decision: str = Field(..., description="Decision: 'approve', 'edit', or 'reject'")
    edited_args: Optional[Dict[str, Any]] = Field(default=None, description="Edited arguments if decision is 'edit'")

