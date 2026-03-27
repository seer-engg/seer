"""
Pydantic schemas for workflow chat assistant.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from seer.database import UserPublic

from .models import WorkflowProposalPublic


class StreamEventType(str, Enum):
    """Type of SSE stream event published during agent execution."""

    SESSION_INFO = (
        "session_info"  # First event: {session_id, thread_id, execution_task_id}
    )
    AGENT_START = "agent_start"  # Agent invocation begins
    TOOL_START = "tool_start"  # {tool_name, input_preview}
    TOOL_END = "tool_end"  # {tool_name, output_preview}
    AI_MESSAGE = "ai_message"  # Intermediate reasoning {content}
    AGENT_END = "agent_end"  # Final answer {content, proposal_id?}
    INTERRUPT = "interrupt"  # {questions: [...]}
    ERROR = "error"  # {message, status_code}
    DONE = "done"  # Stream sentinel {}


class StreamEvent(BaseModel):
    """A single event in the nexus agent SSE stream."""

    type: StreamEventType
    data: Dict[str, Any] = {}
    session_id: int


class WorkflowCreationMode(str, Enum):
    """Mode for workflow creation during discovery."""

    AUTO_CREATE = "AUTO_CREATE"
    ASK_FIRST = "ASK_FIRST"
    ON_ACCEPTANCE = "ON_ACCEPTANCE"


class QuestionType(str, Enum):
    """Type of clarification question."""

    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"
    RESOURCE_PICKER = "resource_picker"
    ACCOUNT_PICKER = "account_picker"


class AccountPickerAccountInfo(BaseModel):
    """Account information for account picker questions."""

    id: int = Field(..., description="Database ID of the OAuth connection")
    display_name: str = Field(
        ..., description="Human-readable account name (email, username)"
    )
    has_required_scopes: bool = Field(
        ..., description="Whether account has all required scopes for the tool"
    )
    missing_scopes: List[str] = Field(
        default_factory=list, description="List of missing scopes if any"
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User's chat message")
    model: Optional[str] = Field(
        default=None,
        description="Model to use for chat (e.g., 'qwen/qwen3-235b-a22b-2507', 'moonshotai/kimi-k2.5')",
    )
    session_id: Optional[int] = Field(
        default=None, description="Chat session ID to resume conversation"
    )
    thread_id: Optional[str] = Field(
        default=None, description="LangGraph thread ID to resume conversation"
    )
    resume_thread: bool = Field(
        default=True,
        description="Whether to resume existing thread if thread_id provided",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Client-generated request identifier for tracing and idempotency",
    )
    timezone: Optional[str] = Field(
        default=None, description="IANA timezone e.g. 'America/New_York'"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Assistant's text response")
    proposal: Optional[WorkflowProposalPublic] = Field(
        default=None,
        description="Workflow proposal containing a full compiler-ready spec",
    )
    proposal_error: Optional[str] = Field(
        default=None, description="Validation error message if proposal creation failed"
    )
    session_id: Optional[int] = Field(default=None, description="Chat session ID")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread ID")
    thinking: Optional[List[str]] = Field(
        default=None, description="Agent thinking/reasoning steps (collapsible)"
    )
    interrupt_required: bool = Field(
        default=False, description="Whether human input is required (human-in-the-loop)"
    )
    interrupt_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Data for human-in-the-loop interrupt"
    )
    workflow_created_id: Optional[str] = Field(
        default=None, description="Workflow ID if created during discovery chat"
    )

    # Optional fields for async mode
    execution_status: Optional[str] = Field(
        default=None,
        description="Execution status (queued/running/completed/failed/interrupted)",
    )
    execution_task_id: Optional[str] = Field(
        default=None, description="Taskiq task ID for background execution"
    )


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""

    title: Optional[str] = Field(
        default=None, description="Optional title for the session"
    )


class ChatSession(BaseModel):
    """Response model for chat session."""

    id: int
    workflow_id: str
    user: UserPublic
    thread_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    current_execution_status: Optional[str] = None
    current_execution_task_id: Optional[str] = None
    pending_interrupt_type: Optional[str] = None
    pending_interrupt_data: Optional[Dict[str, Any]] = None


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
    # Interrupt fields — populated on the last assistant message when the session is interrupted.
    # Allows the frontend to restore the clarification card from message history (reconnect or history navigation).
    interrupt_required: bool = False
    interrupt_data: Optional[Dict[str, Any]] = None


class ChatSessionWithMessages(ChatSession):
    """Chat session with messages."""

    messages: List[ChatMessage] = Field(default_factory=list)


class WorkflowProposalActionResponse(BaseModel):
    """Response for proposal accept/reject actions."""

    proposal: WorkflowProposalPublic
    workflow_graph: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated WorkflowSpec when accepted"
    )


class ClarificationQuestionOption(BaseModel):
    """Option for a clarification question."""

    value: str = Field(..., description="Machine-readable value")
    label: str = Field(..., description="Human-readable label")
    is_wildcard: bool = Field(
        default=False, description="If true, allows custom user input"
    )


class ClarificationQuestion(BaseModel):
    """Clarification question during chat."""

    question_id: str = Field(..., description="Unique identifier for this question")
    question: str = Field(..., description="The question text")
    question_type: QuestionType = Field(
        ..., description="Single, multi-choice, or resource_picker"
    )
    options: List[ClarificationQuestionOption] = Field(
        default_factory=list, description="Available options (for choice types)"
    )
    min_selections: int = Field(
        default=1, description="Minimum selections for multi-choice"
    )
    max_selections: Optional[int] = Field(
        default=None, description="Maximum selections for multi-choice"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Why the agent is asking this question"
    )

    # Resource picker specific fields (only used when question_type is resource_picker)
    provider: Optional[str] = Field(
        default=None,
        description="OAuth provider (google, github, discord, supabase_mgmt)",
    )
    resource_type: Optional[str] = Field(
        default=None,
        description="Type of resource (google_spreadsheet, guild, channel, etc.)",
    )
    display_field: Optional[str] = Field(
        default="name", description="Field to display in picker"
    )
    value_field: Optional[str] = Field(
        default="id", description="Field to use as value"
    )
    search_enabled: Optional[bool] = Field(
        default=True, description="Whether search is supported"
    )
    hierarchy: Optional[bool] = Field(
        default=False, description="Whether folder navigation is supported"
    )
    depends_on: Optional[str] = Field(
        default=None, description="Question ID this depends on (for cascading pickers)"
    )
    depends_on_field: Optional[str] = Field(
        default=None, description="Field name from the dependent resource"
    )

    # Account picker specific fields (only used when question_type is account_picker)
    tool_name: Optional[str] = Field(
        default=None, description="Tool name requiring OAuth (e.g., 'gmail_send_email')"
    )
    accounts: Optional[List[AccountPickerAccountInfo]] = Field(
        default=None, description="Available OAuth accounts for the tool"
    )
    required_scopes: Optional[List[str]] = Field(
        default=None, description="Required scopes for display purposes"
    )


class ClarificationQuestions(BaseModel):
    """Multiple clarification questions asked at once (batch mode)."""

    questions: List[ClarificationQuestion] = Field(
        ..., description="List of questions to answer"
    )


class DiscoveryChatRequest(BaseModel):
    """Request model for discovery chat endpoint (no workflow)."""

    message: str = Field(..., description="User's chat message")
    workflow_creation_mode: Optional[WorkflowCreationMode] = Field(
        default=None, description="Mode for workflow creation"
    )
    model: Optional[str] = Field(default=None, description="Model to use for chat")
    thread_id: Optional[str] = Field(
        default=None, description="LangGraph thread ID to resume conversation"
    )
    session_id: Optional[int] = Field(default=None, description="Discovery session ID")


class InterruptResponse(BaseModel):
    """Response model for human-in-the-loop interrupt."""

    decision: str = Field(..., description="Decision: 'approve', 'edit', or 'reject'")
    edited_args: Optional[Dict[str, Any]] = Field(
        default=None, description="Edited arguments if decision is 'edit'"
    )


class ClarificationAnswer(BaseModel):
    """User's answer to a clarification question."""

    question_id: str = Field(..., description="ID of the question being answered")
    selected_values: List[str] = Field(..., description="Selected option values")
    custom_input: Optional[str] = Field(
        default=None, description="Custom input if wildcard selected"
    )


class ClarificationAnswers(BaseModel):
    """User's answers to multiple clarification questions (batch mode)."""

    answers: List[ClarificationAnswer] = Field(
        ..., description="List of answers, one per question"
    )


class ChatResumeRequest(BaseModel):
    """Request model for resuming chat after interrupt."""

    thread_id: str = Field(..., description="Thread ID to resume")
    answers: Optional[ClarificationAnswers] = Field(
        default=None, description="Answers to clarification questions"
    )
    message: Optional[str] = Field(
        default=None, description="Free-text clarification reply from the user"
    )
    command: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw Command data for other interrupt types"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Client-generated request identifier for tracing and idempotency",
    )

    @model_validator(mode="after")
    def validate_answers_or_command(self) -> "ChatResumeRequest":
        """Ensure exactly one resume payload is provided."""
        provided_payloads = [
            self.answers is not None,
            bool(self.message and self.message.strip()),
            self.command is not None,
        ]

        if sum(provided_payloads) != 1:
            raise ValueError(
                "Exactly one of answers, message, or command must be provided"
            )

        if self.message is not None:
            self.message = self.message.strip()

        return self


class ChatStatusResponse(BaseModel):
    """Response for polling chat execution status."""

    status: str = Field(
        ...,
        description="Execution status (queued/running/completed/failed/interrupted)",
    )
    session_id: int = Field(..., description="Chat session ID")
    thread_id: str = Field(..., description="LangGraph thread ID")

    # Available when completed
    response: Optional[str] = Field(
        default=None, description="Assistant's response text"
    )
    thinking: Optional[List[str]] = Field(
        default=None, description="Agent thinking/reasoning steps"
    )
    proposal: Optional[WorkflowProposalPublic] = Field(
        default=None, description="Workflow proposal if any"
    )

    # Available when interrupted
    interrupt_required: bool = Field(
        default=False, description="Whether human input is required"
    )
    interrupt_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Interrupt data (e.g., clarification question)"
    )

    # Available when failed
    error: Optional[Dict[str, Any]] = Field(
        default=None, description="Error details if execution failed"
    )

    # Timing information
    started_at: Optional[datetime] = Field(
        default=None, description="Execution start time"
    )
    finished_at: Optional[datetime] = Field(
        default=None, description="Execution finish time"
    )
