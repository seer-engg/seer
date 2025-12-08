"""
Simplified state model for eval agent using Supervisor pattern.
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Import complex types for optional storage
from shared.schema import (
    AgentContext,
    DatasetExample,
    AlignmentState,
    AgentSpec,
    CodexOutput,
    DatasetContext,
    ExperimentContext,
    ExperimentResultContext,
)
from agents.eval_agent.models import Hypothesis
from shared.schema import UserIntent


class EvalAgentState(TypedDict, total=False):
    """
    Simplified state for eval agent using Supervisor pattern.
    
    Follows Supervisor pattern: simple state with messages, todos, and tool_call_counts.
    Complex data stored as optional fields (accessed via state dict).
    
    Note: Using total=False makes all fields optional except those with defaults.
    This allows gradual state building.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    todos: List[str]  # Phase-based todos: ["PLANNING", "EXECUTION", "REFLECTION", "FINALIZATION"]
    tool_call_counts: Optional[Dict[str, int]]  # Track tool calls
    
    # Minimal eval-specific fields
    current_phase: Optional[str]  # "planning", "execution", "reflection", "finalization", None
    plan_only_mode: bool  # Flag for plan-only mode
    
    # Complex state (optional, stored directly in state dict for simplicity)
    # Using Any type to allow Pydantic models
    agent_context: Optional[Dict[str, Any]]  # AgentContext as dict
    dataset_examples: Optional[List[Dict[str, Any]]]  # List of DatasetExample dicts
    alignment_state: Optional[Dict[str, Any]]  # AlignmentState as dict
    agent_spec: Optional[Dict[str, Any]]  # AgentSpec as dict
    codex_output: Optional[Dict[str, Any]]  # CodexOutput as dict
    hypothesis: Optional[Dict[str, Any]]  # Hypothesis as dict
    latest_results: Optional[List[Dict[str, Any]]]  # List of ExperimentResultContext dicts
    dataset_context: Optional[Dict[str, Any]]  # DatasetContext as dict
    active_experiment: Optional[Dict[str, Any]]  # ExperimentContext as dict
    target_agent_version: Optional[int]  # Version tracking
    attempts: Optional[int]  # Number of completed eval attempts
    user_intent: Optional[Dict[str, Any]]  # UserIntent as dict

