from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from shared.schema import SandboxContext, GithubContext, UserContext, GeneratedTestCase


class RunContext(BaseModel):
    """Context for the latest evaluation attempt."""
    dataset_name: str = Field(default="", description="Dataset name for the latest evaluation attempt")
    experiment_name: str = Field(default="", description="Experiment identifier for the latest evaluation attempt")
    score: float = Field(default=0.0, description="Latest mean correctness score")
    score_history: List[float] = Field(default_factory=list, description="History of aggregate scores across attempts")
    attempts: int = Field(default=0, description="Number of completed eval attempts")
    current_thread_id: Optional[str] = Field(default=None, description="Temporary thread used during the current run invocation")
    last_results: List[Dict[str, Any]] = Field(default_factory=list, description="Raw result rows produced by the most recent run before upload")
    last_failed_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Failed test case details from the most recent run")


class GeneratedTests(BaseModel):
    """A list of generated test cases."""
    test_cases: list[GeneratedTestCase]


class EvalReflection(BaseModel):
    """A meta-evaluation insight to improve future eval generation only."""
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
    user_context: UserContext = Field(default=None, description="Context for the user")
    sandbox_context: SandboxContext = Field(default=None, description="Context for the active sandbox")
    github_context: GithubContext = Field(default=None, description="Context for the active GitHub repository")
    run: RunContext = Field(default_factory=RunContext, description="Execution context for evaluation attempts")
    test_cases: list[GeneratedTestCase] = Field(default_factory=list)
    previous_inputs: list[str] = Field(default_factory=list, description="History of prior test input messages to avoid repetition")
    codex_thread_id: Optional[str] = Field(default=None, description="Stable thread identifier used when contacting the Codex agent")
    codex_request: Optional[Dict[str, Any]] = Field(default=None, description="Last Codex handoff payload sent for remediation")
    codex_response: Optional[Dict[str, Any]] = Field(default=None, description="Codex response payload, if available")
    codex_followup_branch: Optional[str] = Field(default=None, description="Git branch produced by Codex for follow-up evaluation")
    pending_followup: bool = Field(default=False, description="Whether the eval agent must execute a follow-up evaluation on Codex's branch")
