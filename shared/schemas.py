"""Shared data schemas (reused from Seer v1, simplified)"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
import uuid


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


class TestCase(BaseModel):
    """A single test case generated from a spec"""
    id: str = Field(description="Unique test case identifier")
    expectation_ref: str = Field(description="Which expectation this tests")
    input_message: str = Field(description="Message to send to agent")
    expected_behavior: str = Field(description="How agent should behave")
    success_criteria: str = Field(description="How to judge success")


class EvalSuite(BaseModel):
    """Collection of test cases for an agent"""
    id: str = Field(description="Unique ID for this eval suite")
    spec_name: str = Field(description="Name of the spec this is derived from")
    spec_version: str = Field(description="Version of the spec")
    test_cases: List[TestCase] = Field(description="List of test cases")
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


# Shared, state-only additions for orchestrator <> eval agent
class TargetConfig(BaseModel):
    """How to reach the target agent."""
    url: str = Field(description="e.g. http://127.0.0.1:2024")
    port: Optional[int] = Field(default=None, description="Port if not embedded in URL")
    assistant_id: Optional[str] = Field(default=None, description="LangGraph assistant/graph ID")
    github_url: Optional[str] = Field(default=None, description="Source repository URL")


class AgentExpectation(BaseModel):
    """User-facing expectations as if-then scenarios."""
    if_condition: str = Field(description="Trigger context or user input")
    then_behavior: str = Field(description="Expected behavior or output")
    priority: Literal["must", "should", "nice-to-have"] = "should"


class Eval(BaseModel):
    """A planned evaluation for a target agent (draft → approved → executed)."""
    id: str = Field(default_factory=lambda: f"eval_{uuid.uuid4().hex[:8]}")
    target: TargetConfig
    expectations: List[AgentExpectation]
    status: Literal["draft", "approved", "executed"] = "draft"
    spec_summary: Optional[str] = None
    suite_preview: List[str] = Field(default_factory=list, description="Preview of test inputs")
    suite_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class EvalRun(BaseModel):
    """A single execution of an Eval (running tests and capturing results)."""
    id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    eval_id: str
    eval_thread_id: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: List[TestResult] = Field(default_factory=list)
    passed: int = 0
    failed: int = 0
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

