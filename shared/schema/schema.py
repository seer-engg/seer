"""
This file contains the schemas for the shared data between the agents.
Please review each agents code before making any changes to this file.
"""
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, computed_field
from .agent_context import AgentContext
from shared.config import config

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
    judge_reasoning: str = Field(
        ..., 
        description="Detailed explanation from the judge about the score and failure."
    )

class ServiceInstructions(BaseModel):
    """
    The instructions for the service.
    """
    model_config = ConfigDict(extra="forbid")
    
    service_name: str = Field(
        ...,
        description="The name of the service. e.g. 'asana', 'github', 'jira'."
    )
    instructions: List[str] = Field(
        ...,
        description="The instructions for the service."
    )


class TestCaseIntent(BaseModel):
    """High-level intent of a test case (not exact instructions)."""
    model_config = ConfigDict(extra="forbid")
    
    intent_id: str = Field(..., description="Unique identifier for this test intent")
    description: str = Field(..., description="Human-readable description of what this test validates")
    expected_behavior: str = Field(..., description="What the agent should do")
    validation_criteria: List[str] = Field(..., description="Key things to check")
    complexity: Literal["simple", "moderate", "complex"] = Field(..., description="Complexity level of the test")
    estimated_duration: Optional[str] = Field(None, description="Estimated duration for this test")


class AgentSpec(BaseModel):
    """Agent specification derived from user request."""
    model_config = ConfigDict(extra="forbid")
    
    agent_name: str = Field(..., description="Name of the agent being evaluated")
    primary_goal: str = Field(..., description="Primary goal of the agent")
    key_capabilities: List[str] = Field(..., description="Key capabilities the agent should have")
    required_integrations: List[str] = Field(..., description="MCP services/integrations needed")
    test_scenarios: List[TestCaseIntent] = Field(..., description="Test scenarios/intents to validate")
    assumptions: List[str] = Field(..., description="What the LLM assumes about the agent")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="How confident the LLM is about the spec (0-1)")


class AlignmentQuestion(BaseModel):
    """A single alignment question with metadata."""
    model_config = ConfigDict(extra="forbid")
    
    question_id: str = Field(..., description="Unique identifier for tracking this question")
    question: str = Field(..., description="The actual question text")
    context: str = Field(..., description="Why this question is being asked")
    answer: Optional[str] = Field(None, description="User's answer (if provided)")


class AlignmentState(BaseModel):
    """State for user alignment workflow."""
    model_config = ConfigDict(extra="forbid")
    
    questions: List[AlignmentQuestion] = Field(default_factory=list, description="Exactly 3 alignment questions")
    answers: dict[str, str] = Field(default_factory=dict, description="question_id -> answer mapping")
    is_complete: bool = Field(default=False, description="True when all questions answered (or skipped)")
    # TODO: Add persistence backend for alignment history (future feature)


class UserIntent(BaseModel):
    """Classification of user's intent."""
    model_config = ConfigDict(extra="forbid")
    
    intent_type: Literal["informational", "evaluation_request"] = Field(
        description="Type of user intent"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0-1)"
    )
    reasoning: str = Field(
        description="Why this intent was chosen"
    )

class ExpectedOutput(BaseModel):
    """
    The expected output of the target agent.
    """
    model_config = ConfigDict(extra="forbid")
    
    create_test_data: List[ServiceInstructions] = Field(
        ...,
        description="Prerequisite data/objects in external apps .Prior to target agent being invoked, the environment should be in this state. e.g. there should be a PR with a specific label."
    )
    assert_final_state: List[ServiceInstructions] = Field(
        ...,
        description="Final state of the environements. After target has been invoked, the environment should be in this state. e.g. the asna ticket with name 'test' should be ccompleted."
    )
    expected_action:str = Field(
        ...,
        description="Short description of the expected action that should be taken by the target agent. e.g. 'sync the asana tasks with the github PRs'."
    )

class DatasetExample(BaseModel):
    """Single example in a dataset."""

    example_id: str = Field(..., description="unique id for the test case")
    input_message: str = Field(..., description="The input message that should be send to target agent. MUST NOT CONTAIN ANY HINTS. MUST NOT CONTAIN EXPECTED OUTPUT! MUST NOT CONTAIN ANY PLACEHOLDERS")
    expected_output: ExpectedOutput = Field(...)
    status: Literal["active", "retired"] = Field(
        ...,
        description="Fitness status: 'active' tests are in the pool, 'retired' tests passed too often and are culled."
    )
    model_config = ConfigDict(extra="forbid")

    def to_markdown(self) -> str:
        """
        Render this dataset example as a human-friendly Markdown string.

        Intended for UIs/logs/debugging; it does not affect serialization.
        """

        def _as_code_block(text: str) -> str:
            # Avoid breaking markdown when the content itself contains fenced blocks.
            if "```" in text:
                indented = "\n".join(("    " + line) if line else "" for line in text.splitlines())
                return f"\n{indented}\n"
            return f"\n```\n{text}\n```\n"

        def _render_service_instructions(title: str, items: List[ServiceInstructions]) -> str:
            if not items:
                return f"- **{title}**: (none)\n"
            lines: List[str] = [f"- **{title}**:"]
            for si in items:
                lines.append(f"  - **{si.service_name}**")
                if si.instructions:
                    for inst in si.instructions:
                        lines.append(f"    - {inst}")
                else:
                    lines.append("    - (none)")
            return "\n".join(lines) + "\n"

        parts: List[str] = []
        parts.append(f"### Dataset Example `{self.example_id}`\n")
        parts.append(f"- **Status**: `{self.status}`\n")

        parts.append("#### Input Message")
        parts.append(_as_code_block(self.input_message))

        parts.append("#### Expected Output\n")
        parts.append(f"- **Expected action**: {self.expected_output.expected_action}\n")
        parts.append(_render_service_instructions("Create test data", self.expected_output.create_test_data))
        parts.append(_render_service_instructions("Assert final state", self.expected_output.assert_final_state))

        return "\n".join(parts).rstrip() + "\n"

    


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

    experiment_name: str = Field(..., description="Human-readable experiment name")
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
        default_factory=lambda: config.user_id,
        description=f"The ID of the user. Default is {config.user_id} if not specified"
    )
    raw_request: str = Field(..., description="The raw request from the user")
    model_config = ConfigDict(extra="forbid")


class CodexInput(BaseModel):
    """Input for the Codex agent"""
    
    context: "AgentContext" = Field(..., description="Shared agent context")
    dataset_context: DatasetContext = Field(..., description="The dataset context associated with the evaluation")
    experiment_context: ExperimentContext = Field(..., description="The experiment context associated with the evaluation")
    dataset_examples: List[DatasetExample] = Field(default_factory=list, description="Dataset examples used in the evaluation")


class CodexOutput(BaseModel):
    """Output for the Codex agent"""

    agent_updated: bool = Field(False, description="Whether the agent was updated")
    new_branch_name: Optional[str] = Field(None, description="The name of the new branch")
    updated_context: Optional["AgentContext"] = Field(None, description="The updated agent context")


# Resolve forward references now that AgentContext and context models are defined
AgentContext.model_rebuild(_types_namespace={
    "UserContext": UserContext,
    "GithubContext": GithubContext,
    "SandboxContext": SandboxContext,
})

CodexInput.model_rebuild(_types_namespace={"AgentContext": AgentContext})
CodexOutput.model_rebuild(_types_namespace={"AgentContext": AgentContext})
