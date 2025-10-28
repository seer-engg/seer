from datetime import datetime
from typing import Annotated, Optional, List, Literal
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
    url: str = Field(description="Base URL where the agent is hosted")
    expectations: str = Field(description="User's natural language expectations")

class EvalAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    target_agent_config: TargetAgentConfig
    test_cases: list[GeneratedTestCase]
    dataset_name: str
