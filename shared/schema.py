"""
This file contains the schemas for the shared data between the agents.
Please review each agents code before making any changes to this file.
"""
from uuid import uuid4
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field


class DatasetExample(BaseModel):
    """Single example in a dataset."""

    example_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID of the example. A hexadecimal string (e.g., 'a1b2c3d4...')",
    )  
    reasoning: str = Field(
        description="Why is this example important? What aspect of target agent will it be testing?"
    )
    input_message: str = Field(..., description="The input message that should be send to target agent")
    expected_output: str = Field(..., description="The expected output that should be produced by the target agent")
    model_config = ConfigDict(extra="forbid")


class ExperimentResultContext(BaseModel):
    """Full record of a single test execution."""

    dataset_example: DatasetExample = Field(..., description="The example that was evaluated")
    actual_output: str = Field(..., description="The output produced by the target agent")
    score: float = Field(ge=0.0, le=1.0, description="Judge's score 0-1")
    passed: bool = Field(..., description="Whether the evaluator deemed this test successful. False by default.")
    judge_reasoning: str = Field(..., description="Explanation from the judge about the score")
    started_at: datetime = Field(..., description="Timestamp when the execution started")
    completed_at: datetime = Field(..., description="Timestamp when the execution completed")
    model_config = ConfigDict(extra="forbid")


class ExperimentContext(BaseModel):
    """Aggregate metadata for a single experiment run."""

    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    experiment_name: str = Field(..., description="Human-readable experiment name")
    attempt_index: int = Field(1, description="1-indexed attempt number within the dataset context")
    results: List[ExperimentResultContext] = Field(default_factory=list, description="Ordered test results")
    mean_score: float = Field(0.0, description="Mean score across results in this experiment")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the experiment started")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when the experiment completed")

    @computed_field
    @property
    def failed_results(self) -> List[ExperimentResultContext]:
        """Results that did not pass in this experiment."""
        return [result for result in self.results if not result.passed]

    model_config = ConfigDict(extra="allow")


class DatasetSplitContext(BaseModel):
    """Context for a dataset split."""

    split_name: str
    example_ids: List[str] = Field(default_factory=list, description="The example IDs in the split")
    model_config = ConfigDict(extra="forbid")


class DatasetContext(BaseModel):
    """Dataset metadata, including examples, splits, and experiments."""

    dataset_id: str = Field("", description="Unique identifier for the dataset")
    dataset_name: str = Field("", description="Human-readable dataset name")
    splits: List[DatasetSplitContext] = Field(default_factory=list, description="The splits of the dataset")
    experiments: List[ExperimentContext] = Field(default_factory=list, description="The experiments in the dataset")
    model_config = ConfigDict(extra="forbid")


class GithubContext(BaseModel):
    """Context for the active GitHub repository"""

    agent_name: str = Field(description="The name of the agent")
    repo_url: str = Field(description="The URL of the repository")
    branch_name: str = Field(description="The name of the branch. Default is main if not specified")
    model_config = ConfigDict(extra="forbid")


class SandboxContext(BaseModel):
    """Context for the active sandbox session."""

    sandbox_id: str = Field(..., description="The ID of the sandbox session")
    working_directory: str = Field("", description="The working directory of the sandbox")
    working_branch: str = Field(description="The working branch of the sandbox. Default is main if not specified")

    @computed_field
    @property
    def deployment_url(self) -> str:
        """helper method to get the deployment url of the sandbox"""
        return f"https://2024-{self.sandbox_id}.e2b.app"


class UserContext(BaseModel):
    """Context for the user."""

    user_id: str = Field(description="The ID of the user. Default is user_123 if not specified")
    user_expectation: str = Field(..., description="The user's expectation")
    model_config = ConfigDict(extra="forbid")


class CodexInput(BaseModel):
    """Input for the Codex agent"""

    github_context: GithubContext = Field(..., description="The GitHub context")
    sandbox_context: Optional[SandboxContext] = Field(None, description="The sandbox context")
    user_context: UserContext = Field(..., description="The user context")
    dataset_context: DatasetContext = Field(..., description="The dataset context associated with the evaluation")
    experiment_context: ExperimentContext = Field(..., description="The experiment context associated with the evaluation")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="Dataset examples used in the evaluation")


class CodexOutput(BaseModel):
    """Output for the Codex agent"""

    agent_updated: bool = Field(False, description="Whether the agent was updated")
    new_branch_name: Optional[str] = Field(None, description="The name of the new branch")
    updated_sandbox_context: Optional[SandboxContext] = Field(None, description="The updated sandbox context")
