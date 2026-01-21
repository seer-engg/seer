"""State schema for supervisor-style multi-agent architecture."""
from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SupervisorState(TypedDict):
    """State shared across supervisor and specialist agents."""

    # Message history
    messages: Annotated[list[BaseMessage], add_messages]

    # Specialist outputs
    discovered_tools: Optional[List[Dict[str, Any]]]
    discovered_triggers: Optional[List[Dict[str, Any]]]
    workflow_draft: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]

    # Routing metadata
    current_specialist: Optional[str]
    workflow_complete: bool

    # Original request context
    user_intent: Optional[str]
    workflow_state: Optional[Dict[str, Any]]  # For editing existing workflows
