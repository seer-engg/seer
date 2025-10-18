"""Shared data schemas (reused from Seer v1, simplified)"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


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

