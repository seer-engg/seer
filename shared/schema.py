"""
This file contains the schemas for the shared data between the agents.
Please review each agents code before making any changes to this file.
"""
import os
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, computed_field, model_validator
from shared.tool_catalog import canonicalize_tool_name


class FailureAnalysis(BaseModel):
    """
    Structured analysis of a test case failure, provided by the judge.
    """
    model_config = ConfigDict(extra="forbid")
    
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="The final numeric score from 0.0 (total failure) to 1.0 (perfect)."
    )
    failure_type: Optional[Literal[
        "syntax_error", 
        "logical_error", 
        "instruction_following", 
        "structure_preservation",
        "completeness",
        "assertion_error",
        "runtime_error",
        "other"
    ]] = Field(
        default=None, 
        description="The primary category of the failure. Null if score is 1.0."
    )
    judge_reasoning: str = Field(
        ..., 
        description="Detailed explanation from the judge about the score and failure."
    )


class ActionStep(BaseModel):
    """Single action executed by the evaluator."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(
        ...,
        description="Tool identifier (e.g., 'ASANA_CREATE_TASK', 'GITHUB_CREATE_PULL_REQUEST', 'system.wait').",
    )
    params: str = Field(
        ...,
        description="A JSON string containing the parameters for the tool (e.g., '{\"name\": \"Test\"}' or '{}').",
    )
    assign_to_var: str = Field(
        ...,
        description="Variable name used to store tool output (\"\" if unused).",
    )
    assert_field: str = Field(
        ...,
        description="JSON path to assert against the tool output (\"\" if unused).",
    )
    assert_expected: str = Field(
        ...,
        description="The expected value for the assertion, stored as a string (\"\" if unused).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_tool(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(values, dict):
            return values

        service_hint = values.pop("service", None)
        tool_name = values.get("tool")
        if tool_name:
            values["tool"] = canonicalize_tool_name(tool_name, service_hint=service_hint)
        elif service_hint:
            raise ValueError("'tool' must be provided when 'service' is specified.")
        return values


class ExpectedOutput(BaseModel):
    """
    The expected output of the target agent.
    """
    model_config = ConfigDict(extra="forbid")
    actions: List[ActionStep] = Field(..., description="A list of action steps to execute for the test.")


class DatasetExample(BaseModel):
    """Single example in a dataset."""

    example_id: str = Field(..., description="UUID of the example. A hexadecimal string (e.g., 'a1b2c3d4...')")
    reasoning: str = Field(...,
        description="Why is this example important? What aspect of target agent will it be testing?"
    )
    input_message: str = Field(..., description="The input message that should be send to target agent. MUST NOT CONTAIN ANY HINTS. MUST NOT CONTAIN EXPECTED OUTPUT!")
    expected_output: ExpectedOutput = Field(...)
    status: Literal["active", "retired"] = Field(
        "active",
        description="Fitness status: 'active' tests are in the pool, 'retired' tests passed too often and are culled."
    )
    model_config = ConfigDict(extra="forbid")


class ExperimentResultContext(BaseModel):
    """Full record of a single test execution."""

    thread_id: str = Field(..., description="The thread ID of the evaluation")
    dataset_example: DatasetExample = Field(..., description="The example that was evaluated")
    actual_output: str = Field(..., description="The output produced by the target agent")
    analysis: FailureAnalysis = Field(..., description="Structured analysis from the judge.")
    
    @computed_field
    @property
    def score(self) -> float:
        """The score from the analysis."""
        return self.analysis.score

    @computed_field
    @property
    def judge_reasoning(self) -> str:
        """The judge's reasoning from the analysis."""
        return self.analysis.judge_reasoning
    
    passed: bool = Field(..., description="Whether the evaluator deemed this test successful.")
    started_at: datetime = Field(..., description="Timestamp when the execution started")
    completed_at: datetime = Field(..., description="Timestamp when the execution completed")
    model_config = ConfigDict(extra="allow")


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

class DatasetContext(BaseModel):
    """Dataset metadata, including examples, splits, and experiments."""

    dataset_id: str = Field("", description="Unique identifier for the dataset")
    dataset_name: str = Field("", description="Human-readable dataset name")
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

    user_id: str = Field(
        default_factory=lambda: os.getenv("USER_ID", ""),
        description=f"The ID of the user. Default is {os.getenv('USER_ID', '')} if not specified"
    )
    raw_request: str = Field(..., description="The raw request from the user")
    model_config = ConfigDict(extra="forbid")


class CodexInput(BaseModel):
    """Input for the Codex agent"""

    github_context: GithubContext = Field(..., description="The GitHub context")
    sandbox_context: Optional[SandboxContext] = Field(None, description="The sandbox context")
    user_context: UserContext = Field(..., description="The user context")
    dataset_context: DatasetContext = Field(..., description="The dataset context associated with the evaluation")
    experiment_context: ExperimentContext = Field(..., description="The experiment context associated with the evaluation")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="Dataset examples used in the evaluation")
    target_agent_version: int = Field(..., description="Version of the target agent")
    mcp_services: List[str] = Field(default_factory=list, description="List of MCP service names required for this eval, e.g., ['asana', 'github']")


class CodexOutput(BaseModel):
    """Output for the Codex agent"""

    agent_updated: bool = Field(False, description="Whether the agent was updated")
    new_branch_name: Optional[str] = Field(None, description="The name of the new branch")
    updated_sandbox_context: Optional[SandboxContext] = Field(None, description="The updated sandbox context")
    target_agent_version: int = Field(..., description="Version of the target agent")
