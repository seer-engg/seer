"""models for the evaluation agent"""
import uuid
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from shared.schema import (
    DatasetExample,
    DatasetContext,
    ExperimentContext,
    ExperimentResultContext,
    FailureAnalysis,
    CodexOutput,
    ActionStep,
)
from shared.tools import ToolEntry

# Import AgentContext after schema to avoid circular imports
from shared.agent_context import AgentContext

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
    
    # Dynamic cleanup stack (LIFO: last created = first deleted)
    cleanup_stack: List['ActionStep'] = Field(
        default_factory=list,
        description="Stack of inverse cleanup actions generated during provisioning. Executed in reverse order (LIFO)."
    )
    
    # Tools
    tool_entries: Dict[str, ToolEntry] = Field(
        default_factory=dict, 
        description="Subset of tools selected for this evaluation run"
    )


class EvalAgentPlannerState(EvalAgentState):
    """State for the evaluation agent planner."""
    reflections_text: Optional[str] = Field(default=None, description="Text of the reflections to use for test generation")
    structured_response: Optional[dict] = Field(default=None, description="The structured response from the test generation agent")



class TestGenerationOutput(BaseModel):
    """Helper for structured output"""
    model_config = ConfigDict(extra="forbid")
    dataset_example: DatasetExample


# -----------------------------------------------------------------------------
# Subgraph state for per-example test execution
# -----------------------------------------------------------------------------
class TestExecutionState(BaseModel):
    """State for executing a single DatasetExample through provision → invoke → assert."""
    context: AgentContext = Field(default_factory=AgentContext, description="Shared agent context")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="Batch of dataset examples to execute")
    dataset_example: Optional[DatasetExample] = Field(default=None, description="Dataset example to execute")
    mcp_resources: Dict[str, Any] = Field(default_factory=dict, description="Working MCP resources for this test")
    cleanup_stack: List[ActionStep] = Field(default_factory=list, description="Cleanup actions collected during execution (LIFO)")
    pending_examples: List[DatasetExample] = Field(default_factory=list, description="Internal queue of pending examples (initialized from dataset_examples)")
    accumulated_results: List[ExperimentResultContext] = Field(default_factory=list, description="Internal accumulator of per-example results")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from running the current batch, aligned with EvalAgentState")
    thread_id: Optional[str] = Field(default=None, description="Thread ID from target agent invocation")
    agent_output: str = Field(default="", description="Final text output from the target agent invocation")
    analysis: Optional[FailureAnalysis] = Field(default=None, description="Evaluation analysis from assertion phase")
    result: Optional[ExperimentResultContext] = Field(default=None, description="Final experiment result context for this example")
    started_at: Optional[datetime] = Field(default=None, description="Start time of this example execution")
    completed_at: Optional[datetime] = Field(default=None, description="End time of this example execution")
    assertion_output:Optional[str] = Field(default=None, description="The output from the assertion agent")
    provisioning_output:Optional[str] = Field(default=None, description="The output from the provisioning agent")
    
    tool_entries: Dict[str, ToolEntry] = Field(
        default_factory=dict, 
        description="Subset of tools selected for this evaluation run"
    )


# Rebuild models to resolve forward references
EvalAgentState.model_rebuild()
EvalAgentPlannerState.model_rebuild()
TestExecutionState.model_rebuild()
