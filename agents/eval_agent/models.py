"""models for the evaluation agent"""
import uuid
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from shared.schema import (
    SandboxContext,
    GithubContext,
    UserContext,
    DatasetExample,
    DatasetContext,
    ExperimentContext,
    ExperimentResultContext,
    CodexOutput,
)
from shared.tool_catalog import ToolEntry
import os

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

class EvalReflection(BaseModel):
    """A meta-evaluation insight to improve future eval generation only."""
    user_id: str = Field(description="The user this reflection belongs to.")
    reflection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique ID for this reflection."
    )
    agent_name: str = Field(description="Target agent/graph this reflection applies to")
    hypothesis: Hypothesis = Field(description="Creative hypothesis about a failure mode in the latest attempt")

    # Metadata
    latest_score: Optional[float] = Field(
        default=None,
        description="Most recent aggregate score when this reflection was generated",
    )
    attempt: Optional[int] = Field(
        default=None,
        description="Eval attempt count (1-indexed) when this reflection was produced",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Dataset involved in the run that produced this reflection",
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="Experiment name involved in the run that produced this reflection",
    )
    created_at: datetime = Field(default_factory=datetime.now)


class ToolSelectionLog(BaseModel):
    """
    A record of which tools were selected for test generation
    and the context used to make that selection.
    This provides transparency for debugging in LangGraph Studio.
    """
    selection_context: str = Field(description="The 'why' - the context string used to score and prioritize tools.")
    selected_tools: List[str] = Field(description="The 'what' - the list of tool names that were prioritized and selected.")

class EvalAgentState(BaseModel):
    """State for the evaluation agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    attempts: int = Field(default=0, description="Number of completed eval attempts")
    user_context: Optional[UserContext] = Field(default=None, description="Context for the user")
    sandbox_context: Optional[SandboxContext] = Field(default=None, description="Context for the active sandbox")
    github_context: Optional[GithubContext] = Field(default=None, description="Context for the active GitHub repository")
    dataset_context: DatasetContext = Field(default_factory=DatasetContext, description="Dataset metadata used across experiments")
    active_experiment: Optional[ExperimentContext] = Field(default=None, description="Currently running experiment context")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the latest experiment execution")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="List of generated test cases")
    target_agent_version: int = Field(default=0, description="Version of the target agent being evaluated")
    codex_output: Optional[CodexOutput] = Field(default=None, description="Output from the codex agent, used for handoff.")
    tool_selection_log: Optional[ToolSelectionLog] = Field(
        default=None, 
        description="The log of how MCP tools were selected for the current round."
    )
    mcp_services: List[str] = Field(
        default_factory=list, 
        description="List of MCP service names required for this eval, e.g., ['asana', 'github']"
    )
    mcp_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="MCP resource IDs created for this experiment, e.g., {'asana_project_id': '123', 'github_repo_id': '456'}"
    )


class EvalAgentPlannerState(EvalAgentState):
    """State for the evaluation agent planner."""
    reflections_text: Optional[str] = Field(default=None, description="Text of the reflections to use for test generation")
    available_tools: List[str] = Field(default_factory=list, description="List of available tools to use for test generation")
    tool_entries: Dict[str, ToolEntry] = Field(default_factory=dict, description="Tool entries to use for test generation")
    use_genetic_test_generation: bool = Field(default=os.getenv("USE_GENETIC_TEST_GENERATION", "false").lower() == "true", description="Whether to use genetic test generation")
    use_agentic_test_generation: bool = Field(default=os.getenv("USE_AGENTIC_TEST_GENERATION", "false").lower() == "true", description="Whether to use agentic test generation")



class TestGenerationOutput(BaseModel):
    """Helper for structured output"""
    model_config = ConfigDict(extra="forbid")
    dataset_example: DatasetExample
