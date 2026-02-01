"""
Pydantic schemas for workflow chat assistant.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from seer.database import UserPublic

from .models import WorkflowProposalPublic


class WorkflowCreationMode(str, Enum):
    """Mode for workflow creation during discovery."""
    AUTO_CREATE = "AUTO_CREATE"
    ASK_FIRST = "ASK_FIRST"
    ON_ACCEPTANCE = "ON_ACCEPTANCE"


class QuestionType(str, Enum):
    """Type of clarification question."""
    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's chat message")
    workflow_state: Dict[str, Any] = Field(..., description="Legacy workflow state snapshot (ReactFlow schema)")
    model: Optional[str] = Field(default=None, description="Model to use for chat (e.g., 'gpt-5.2', 'claude-opus-4-5')")
    session_id: Optional[int] = Field(default=None, description="Chat session ID to resume conversation")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID to resume conversation")
    resume_thread: bool = Field(default=True, description="Whether to resume existing thread if thread_id provided")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's text response")
    proposal: Optional[WorkflowProposalPublic] = Field(default=None, description="Workflow proposal containing a full compiler-ready spec")
    proposal_error: Optional[str] = Field(default=None, description="Validation error message if proposal creation failed")
    session_id: Optional[int] = Field(default=None, description="Chat session ID")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID")
    thinking: Optional[List[str]] = Field(default=None, description="Agent thinking/reasoning steps (collapsible)")
    interrupt_required: bool = Field(default=False, description="Whether human input is required (human-in-the-loop)")
    interrupt_data: Optional[Dict[str, Any]] = Field(default=None, description="Data for human-in-the-loop interrupt")
    workflow_created_id: Optional[str] = Field(default=None, description="Workflow ID if created during discovery chat")


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""
    title: Optional[str] = Field(default=None, description="Optional title for the session")


class ChatSession(BaseModel):
    """Response model for chat session."""
    id: int
    workflow_id: str
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
    workflow_graph: Optional[Dict[str, Any]] = Field(default=None, description="Updated WorkflowSpec when accepted")


class ClarificationQuestionOption(BaseModel):
    """Option for a clarification question."""
    value: str = Field(..., description="Machine-readable value")
    label: str = Field(..., description="Human-readable label")
    is_wildcard: bool = Field(default=False, description="If true, allows custom user input")


class ClarificationQuestion(BaseModel):
    """Clarification question during chat."""
    question_id: str = Field(..., description="Unique identifier for this question")
    question: str = Field(..., description="The question text")
    question_type: QuestionType = Field(..., description="Single or multi-choice")
    options: List[ClarificationQuestionOption] = Field(..., description="Available options")
    min_selections: int = Field(default=1, description="Minimum selections for multi-choice")
    max_selections: Optional[int] = Field(default=None, description="Maximum selections for multi-choice")


class DiscoveryChatRequest(BaseModel):
    """Request model for discovery chat endpoint (no workflow)."""
    message: str = Field(..., description="User's chat message")
    workflow_creation_mode: Optional[WorkflowCreationMode] = Field(default=None, description="Mode for workflow creation")
    model: Optional[str] = Field(default=None, description="Model to use for chat")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID to resume conversation")
    session_id: Optional[int] = Field(default=None, description="Discovery session ID")


class InterruptResponse(BaseModel):
    """Response model for human-in-the-loop interrupt."""
    decision: str = Field(..., description="Decision: 'approve', 'edit', or 'reject'")
    edited_args: Optional[Dict[str, Any]] = Field(default=None, description="Edited arguments if decision is 'edit'")


class ClarificationAnswer(BaseModel):
    """User's answer to a clarification question."""
    question_id: str = Field(..., description="ID of the question being answered")
    selected_values: List[str] = Field(..., description="Selected option values")
    custom_input: Optional[str] = Field(default=None, description="Custom input if wildcard selected")


class ChatResumeRequest(BaseModel):
    """Request model for resuming chat after interrupt."""
    thread_id: str = Field(..., description="Thread ID to resume")
    answer: Optional[ClarificationAnswer] = Field(default=None, description="Answer to clarification question")
    command: Optional[Dict[str, Any]] = Field(default=None, description="Raw Command data for other interrupt types")

    @model_validator(mode="after")
    def validate_one_of_answer_or_command(self) -> "ChatResumeRequest":
        """Ensure either answer or command is provided."""
        if not self.answer and not self.command:
            raise ValueError("Either answer or command must be provided")
        return self
