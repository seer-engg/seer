"""
Pydantic schemas for workflow chat assistant.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from shared.database.models import UserPublic
from .models import WorkflowProposalPublic


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
    proposal: Optional[WorkflowProposalPublic] = Field(default=None, description="Workflow proposal with patch operations")
    proposal_error: Optional[str] = Field(default=None, description="Validation error message if proposal creation failed")
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
    proposal: Optional[WorkflowProposalPublic] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class ChatSessionWithMessages(ChatSession):
    """Chat session with messages."""
    messages: List[ChatMessage] = Field(default_factory=list)


class WorkflowProposalActionResponse(BaseModel):
    """Response for proposal accept/reject actions."""
    proposal: WorkflowProposalPublic
    workflow_graph: Optional[Dict[str, Any]] = Field(default=None, description="Updated workflow graph when accepted")


class InterruptResponse(BaseModel):
    """Response model for human-in-the-loop interrupt."""
    decision: str = Field(..., description="Decision: 'approve', 'edit', or 'reject'")
    edited_args: Optional[Dict[str, Any]] = Field(default=None, description="Edited arguments if decision is 'edit'")

