from datetime import datetime
from typing import Annotated, Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



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


class TestResult(BaseModel):
    """Result of running a single test case"""
    test_case_id: str
    input_sent: str
    actual_output: str
    expected_behavior: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Judge's score 0-1")
    judge_reasoning: str = Field(description="Why the judge scored this way")

class GeneratedTestCase(BaseModel):
    expectation_ref: str
    input_message: str
    expected_behavior: str
    success_criteria: str
    expected_output: Optional[str] = None


class GeneratedTests(BaseModel):
    test_cases: list[GeneratedTestCase]


class JudgeVerdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


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
    expectation_ref: Optional[str] = Field(default=None, description="Expectation/test reference this reflection relates to")
    summary: str = Field(description="Concise reflection focused on improving test quality/coverage")
    created_at: datetime = Field(default_factory=datetime.now)


class EvalAgentState(BaseModel):
    """State for eval-agent v2 reflexion loop (Plan → Run → Reflect)."""
    messages: Annotated[list[BaseMessage], add_messages]
    target_agent_config: Optional[TargetAgentConfig] = Field(default=None, description="Configuration for the target agent to evaluate")
    # Planning
    test_cases: list[GeneratedTestCase] = Field(default_factory=list)
    previous_inputs: list[str] = Field(default_factory=list, description="History of prior test input messages to avoid repetition")
    sandbox_id: Optional[str] = Field(default=None, description="Identifier for the active E2B sandbox session")
    sandbox_repo_dir: Optional[str] = Field(default=None, description="Absolute path of the checked-out repository inside the sandbox")
    sandbox_branch: Optional[str] = Field(default=None, description="Branch currently active inside the sandbox")
    deployment_url: Optional[str] = Field(default=None, description="HTTP base URL for the deployed target agent")
    deployment_metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata captured during deployment (e.g., handles, timestamps)")
    codex_thread_id: Optional[str] = Field(default=None, description="Stable thread identifier used when contacting the Codex agent")
    codex_request: Optional[Dict[str, Any]] = Field(default=None, description="Last Codex handoff payload sent for remediation")
    codex_response: Optional[Dict[str, Any]] = Field(default=None, description="Codex response payload, if available")
    last_failed_cases: list[Dict[str, Any]] = Field(default_factory=list, description="Failed test case details from the most recent run")
    last_run_metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregated metrics/metadata from the most recent run used for handoff")
    current_run_thread_id: Optional[str] = Field(default=None, description="Temporary thread used during the current run invocation")
    last_run_results: list[Dict[str, Any]] = Field(default_factory=list, description="Raw result rows produced by the most recent run before upload")
    accumulated_failed_cases: list[Dict[str, Any]] = Field(default_factory=list, description="Cumulative failing cases across attempts pending Codex escalation")
    accumulated_run_context: Dict[str, Any] = Field(default_factory=dict, description="Metadata tied to accumulated failures for Codex handoff")
    # Execution
    dataset_name: str = ""
    experiment_name: str = ""
    score: float = 0.0
    score_history: list[float] = Field(default_factory=list)
    # Loop control
    attempts: int = 0
    # Reflections for future test generation
    reflections: list[EvalReflection] = Field(default_factory=list)
