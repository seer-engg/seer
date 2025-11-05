"""models for the evaluation agent"""
import uuid
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

class Hypothesis(BaseModel):
    """
    The creative output of the Eval Agent.
    Contains only the fields the LLM is responsible for generating.
    """
    summary: str = Field(description="Concise summary of new insights, including any flakiness.")
    failure_modes: List[str] = Field(
        default_factory=list,
        description="Key failure themes observed (e.g., 'Flakiness in divide_by_zero', 'New failure in dict_key_handling').",
    )
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="Specific, new test ideas to create next (e.g., 're-run divide_by_zero 3 times', 'test dict access with missing key').",
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


class FinalReflection(BaseModel):
    """
    The final, synthesized reflection produced by the Eval Agent.
    This is the *input* to the 'save_reflection' tool.
    """
    summary: str = Field(description="Concise summary of new insights, including any flakiness.")
    failure_modes: List[str] = Field(
        default_factory=list,
        description="Key failure themes observed (e.g., 'Flakiness in divide_by_zero', 'New failure in dict_key_handling').",
    )
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="Specific, new test ideas to create next (e.g., 're-run divide(10,0) 3 times', 'test dict access with missing key').",
    )


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
