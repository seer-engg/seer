from datetime import datetime
from typing import Annotated, Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class DeploymentContext(BaseModel):
    url: Optional[str] = Field(default=None, description="HTTP base URL for the deployed target agent")
    repo_url: Optional[str] = Field(default=None, description="Git repository associated with the current deployment")
    branch_name: Optional[str] = Field(default=None, description="Git branch targeted for evaluation")
    sandbox_id: Optional[str] = Field(default=None, description="Identifier for the active E2B sandbox session")
    sandbox_repo_dir: Optional[str] = Field(default=None, description="Absolute path of the checked-out repository inside the sandbox")
    sandbox_branch: Optional[str] = Field(default=None, description="Branch currently active inside the sandbox")
    setup_script: Optional[str] = Field(default=None, description="Shell script used to set up the project inside the sandbox")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional deployment metadata (timestamps, codex info, etc.)")


class RunContext(BaseModel):
    dataset_name: str = Field(default="", description="Dataset name for the latest evaluation attempt")
    experiment_name: str = Field(default="", description="Experiment identifier for the latest evaluation attempt")
    score: float = Field(default=0.0, description="Latest mean correctness score")
    score_history: List[float] = Field(default_factory=list, description="History of aggregate scores across attempts")
    attempts: int = Field(default=0, description="Number of completed eval attempts")
    current_thread_id: Optional[str] = Field(default=None, description="Temporary thread used during the current run invocation")
    last_results: List[Dict[str, Any]] = Field(default_factory=list, description="Raw result rows produced by the most recent run before upload")
    last_metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregated metrics/metadata from the most recent run used for handoff")
    last_failed_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Failed test case details from the most recent run")
    accumulated_failed_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Cumulative failing cases across attempts pending Codex escalation")
    accumulated_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tied to accumulated failures for Codex handoff")


class Expectation(BaseModel):
    """A single behavioral expectation for the agent"""
    description: str = Field(description="What the agent should do")
    context: Optional[str] = Field(default=None, description="When/why this applies")
    priority: Literal["must", "should", "nice-to-have"] = Field(default="should")


class AgentSpec(BaseModel):
    """Complete specification for an agent's expected behavior"""
    name: str = Field(description="Name of the agent being specified")
    version: str = Field(default="1.0.0", description="Spec version")
    description: str = Field(description="What the agent does")
    expectations: List[Expectation] = Field(description="List of behavioral expectations")
    created_at: datetime = Field(default_factory=datetime.now)


class GeneratedTestCase(BaseModel):
    input_message: str
    expected_behavior: str
    success_criteria: str
    expected_output: Optional[str] = None


class GeneratedTests(BaseModel):
    test_cases: list[GeneratedTestCase]


class TargetAgentConfig(BaseModel):
    graph_name: str = Field(description="Name of the graph (NOT ASSISTANT ID WHICH IS A HEX STRING) to evaluate")
    repo_url: str = Field(description="Git repository containing the target agent")
    expectations: str = Field(description="User's natural language expectations")
    url: Optional[str] = Field(default=None, description="Base URL where the agent is already hosted, if any")
    branch_name: str = Field(default="main", description="Git branch to check out inside the sandbox")
    setup_script: str = Field(
        default="pip install -e .",
        description="Shell script (single line) to set up the project inside the sandbox before launch",
    )

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
    messages: Annotated[list[BaseMessage], add_messages]
    target_agent_config: Optional[TargetAgentConfig] = Field(default=None, description="Configuration for the target agent to evaluate")
    deployment: DeploymentContext = Field(default_factory=DeploymentContext, description="Context for the active deployment/sandbox")
    run: RunContext = Field(default_factory=RunContext, description="Execution context for evaluation attempts")
    # Planning
    test_cases: list[GeneratedTestCase] = Field(default_factory=list)
    previous_inputs: list[str] = Field(default_factory=list, description="History of prior test input messages to avoid repetition")
    planning_trace: Dict[str, Any] = Field(default_factory=dict, description="LLM prompt/response traces captured during planning")
    codex_thread_id: Optional[str] = Field(default=None, description="Stable thread identifier used when contacting the Codex agent")
    codex_request: Optional[Dict[str, Any]] = Field(default=None, description="Last Codex handoff payload sent for remediation")
    codex_response: Optional[Dict[str, Any]] = Field(default=None, description="Codex response payload, if available")
    codex_followup_branch: Optional[str] = Field(default=None, description="Git branch produced by Codex for follow-up evaluation")
    codex_followup_metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metadata returned by Codex during handoff")
    pending_followup: bool = Field(default=False, description="Whether the eval agent must execute a follow-up evaluation on Codex's branch")
    finalize_context: Dict[str, Any] = Field(default_factory=dict, description="Temporary data assembled while finalizing an eval attempt")
