"""models for the evaluation agent"""
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from shared.schema import (
    DatasetExample,
    DatasetContext,
    ExperimentContext,
    ExperimentResultContext,
    FailureAnalysis,
    CodexOutput,
    AgentSpec,
    AlignmentState,
    UserIntent,
)

# Import AgentContext after schema to avoid circular imports
from shared.schema import AgentContext

class Hypothesis(BaseModel):
    """
    The creative output of the Eval Agent.
    Contains only the fields the LLM is responsible for generating.
    """
    summary: str = Field(description="Concise summary of new insights, including any flakiness.")
    test_generation_critique: Optional[str] = Field(
        default=None,
        description="A critique of the test cases that were just run. Were they too easy? Did they find the *right* bugs? What could be improved for next time?"
    )

class EvalAgentState(BaseModel):
    """State for the evaluation agent."""

    # Core agent context (shared with Codex)
    context: AgentContext = Field(default_factory=AgentContext, description="Shared agent context")
    # Working resources (aligned with execution subgraph expectations)
    mcp_resources: Dict[str, Any] = Field(default_factory=dict, description="Working MCP resources for this run")
    
    # Agent-specific state
    messages: Annotated[list[BaseMessage], add_messages]
    attempts: int = Field(default=0, description="Number of completed eval attempts")
    
    # Evaluation-specific context
    dataset_context: DatasetContext = Field(default_factory=DatasetContext, description="Dataset metadata used across experiments")
    active_experiment: Optional[ExperimentContext] = Field(default=None, description="Currently running experiment context")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the latest experiment execution")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="List of generated test cases")
    
    # Handoff from Codex
    codex_output: Optional[CodexOutput] = Field(default=None, description="Output from the codex agent, used for handoff.")
    
    # Agent specification and alignment (for plan-only mode)
    agent_spec: Optional[AgentSpec] = Field(default=None, description="Agent specification derived from user request")
    alignment_state: Optional[AlignmentState] = Field(default=None, description="State for user alignment workflow")
    
    # Intent classification
    user_intent: Optional[UserIntent] = Field(default=None, description="Classification of user's intent")

    input_context:Optional[Dict[Any, Any]] = Field(default=None, description="Input context for the evaluation agent")

    step: Optional[str] = Field(default=None, description="The step to execute next")

    # Preflight config checks (used to early-exit subgraphs cleanly)
    should_exit: bool = Field(default=False, description="Whether the current graph/subgraph should early-exit")
    missing_config: List[str] = Field(default_factory=list, description="Missing required config keys detected by preflight")
    


class EvalAgentPlannerState(EvalAgentState):
    """State for the evaluation agent planner."""
    reflections_text: Optional[str] = Field(default=None, description="Text of the reflections to use for test generation")



# -----------------------------------------------------------------------------
# Subgraph state for per-example test execution
# -----------------------------------------------------------------------------
class TestExecutionState(BaseModel):
    """State for executing a single DatasetExample through provision → invoke → assert."""
    context: AgentContext = Field(default_factory=AgentContext, description="Shared agent context")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="Batch of dataset examples to execute")
    dataset_example: Optional[DatasetExample] = Field(default=None, description="Dataset example to execute")
    mcp_resources: Dict[str, Any] = Field(default_factory=dict, description="Working MCP resources for this test")
    pending_examples: List[DatasetExample] = Field(default_factory=list, description="Internal queue of pending examples (initialized from dataset_examples)")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from running the current batch, aligned with EvalAgentState")
    thread_id: Optional[str] = Field(default=None, description="Thread ID from target agent invocation")
    agent_output: str = Field(default="", description="Final text output from the target agent invocation")
    analysis: Optional[FailureAnalysis] = Field(default=None, description="Evaluation analysis from assertion phase")
    result: Optional[ExperimentResultContext] = Field(default=None, description="Final experiment result context for this example")
    started_at: Optional[datetime] = Field(default=None, description="Start time of this example execution")
    completed_at: Optional[datetime] = Field(default=None, description="End time of this example execution")
    assertion_output:Optional[str] = Field(default=None, description="The output from the assertion agent")
    provisioning_output:Optional[str] = Field(default=None, description="The output from the provisioning agent")
    provisioning_verification: Optional[Dict[str, Any]] = Field(default=None, description="Verification result of provisioning success (before target agent invocation)")
    current_seed: Optional[str] = Field(default=None, description="The current seed for the test execution")

    messages: Annotated[list[BaseMessage], add_messages]



# Rebuild models to resolve forward references
EvalAgentState.model_rebuild()
EvalAgentPlannerState.model_rebuild()
TestExecutionState.model_rebuild()
