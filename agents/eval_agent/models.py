"""models for the evaluation agent"""
from datetime import datetime
from typing import Annotated, Optional, List

from pydantic import BaseModel, Field
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
)


class EvalReflection(BaseModel):
    """A meta-evaluation insight to improve future eval generation only."""
    reflection_id: str = Field(..., description="Unique ID for this reflection")
    agent_name: str = Field(description="Target agent/graph this reflection applies to")
    summary: str = Field(description="Concise reflection focused on improving test quality/coverage")
    failure_modes: List[str] = Field(
        default_factory=list,
        description="Key failure themes observed across the latest attempt(s)",
    )
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="Specific future test ideas derived from the observed failures",
    )
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
