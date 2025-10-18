"""Event message schemas for inter-agent communication"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from datetime import datetime
import uuid


class EventType:
    """Event type constants"""
    # User interaction events
    MESSAGE_FROM_USER = "MessageFromUser"
    MESSAGE_TO_USER = "MessageToUser"
    
    # Workflow events
    INITIAL_AGENT_QUERY = "InitialAgentQuery"
    USER_CONFIRMATION_QUERY = "UserConfirmationQuery"
    USER_CONFIRMATION = "UserConfirmation"
    
    # Eval workflow milestones
    EVALS_CONFIRMED = "EvalsConfirmed"  # Reserved for future use - when user explicitly confirms eval suite
    TEST_RESULTS_READY = "TestResultsReady"
    
    # System events
    AGENT_STARTED = "AgentStarted"
    AGENT_ERROR = "AgentError"


class EventMessage(BaseModel):
    """Base event message format"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(description="Type of event (see EventType)")
    sender: str = Field(description="Name of the agent/user sending the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: dict[str, Any] = Field(default={}, description="Event-specific data")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    reply_to: Optional[str] = Field(default=None, description="Message ID this replies to")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Specific event payload schemas for type safety

class InitialAgentQueryPayload(BaseModel):
    """Payload for InitialAgentQuery event"""
    target_agent_url: str = Field(description="URL of agent to evaluate (e.g., http://localhost:2024)")
    target_agent_id: str = Field(description="LangGraph assistant ID")
    expectations: str = Field(description="Natural language expectations for the agent")
    github_url: Optional[str] = Field(default=None, description="Optional GitHub URL (not used in v2)")


class MessageFromUserPayload(BaseModel):
    """Payload for MessageFromUser event"""
    content: str = Field(description="User's message content")
    user_id: str = Field(default="lokesh", description="User identifier")


class MessageToUserPayload(BaseModel):
    """Payload for MessageToUser event"""
    content: str = Field(description="Message content to send to user")
    message_type: Literal["info", "question", "error", "success"] = Field(default="info")


class UserConfirmationQueryPayload(BaseModel):
    """Payload for UserConfirmationQuery event"""
    question: str = Field(description="Question to ask the user")
    context: dict[str, Any] = Field(default={}, description="Context data (e.g., generated tests)")
    options: Optional[list[str]] = Field(default=None, description="Optional choices for user")


class UserConfirmationPayload(BaseModel):
    """Payload for UserConfirmation event"""
    answer: str = Field(description="User's answer/confirmation")
    confirmed: bool = Field(description="Whether user confirmed (yes/no)")


class EvalsConfirmedPayload(BaseModel):
    """
    Payload for EvalsConfirmed event.
    
    Note: Currently reserved for future use. This event would be published after user
    explicitly confirms the generated test suite, before execution begins.
    Currently, confirmation flows directly from UserConfirmation to test execution.
    """
    eval_suite_id: str = Field(description="ID of the confirmed eval suite")
    spec_summary: dict[str, Any] = Field(description="Summary of spec and test cases")


class TestResultsReadyPayload(BaseModel):
    """Payload for TestResultsReady event"""
    eval_suite_id: str
    total_tests: int
    passed: int
    failed: int
    overall_score: float
    summary: str = Field(description="Human-readable summary")
    details: dict[str, Any] = Field(default={}, description="Detailed results")


class AgentStartedPayload(BaseModel):
    """Payload for AgentStarted event"""
    agent_name: str
    agent_type: str  # "customer_success", "eval_agent"
    capabilities: list[str] = Field(default=[])


class AgentErrorPayload(BaseModel):
    """Payload for AgentError event"""
    error_message: str
    error_type: str
    context: dict[str, Any] = Field(default={})

