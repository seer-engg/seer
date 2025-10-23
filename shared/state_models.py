"""
Pydantic models for LangGraph state storage.
These replace the old Peewee database models.
All data is now stored in LangGraph's state persistence layer.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EvalSuite(BaseModel):
    """Evaluation test suite"""
    suite_id: str
    spec_name: str
    spec_version: str = "1.0.0"
    target_agent_url: Optional[str] = None
    target_agent_id: Optional[str] = None
    langgraph_thread_id: Optional[str] = None
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class TestResult(BaseModel):
    """Test execution result"""
    result_id: str
    suite_id: str
    test_case_id: str
    input_sent: str
    actual_output: str
    expected_behavior: str
    passed: bool
    score: float
    judge_reasoning: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class TargetAgentExpectation(BaseModel):
    """Target agent expectations collected from user"""
    expectations: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class TargetAgentConfig(BaseModel):
    """Target agent configuration"""
    target_agent_port: Optional[int] = None
    target_agent_url: Optional[str] = None
    target_agent_github_url: Optional[str] = None
    target_agent_assistant_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class RemoteThreadLink(BaseModel):
    """Mapping from user thread to remote agent thread"""
    src_agent: str
    dst_agent: str
    remote_base_url: str
    remote_thread_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentActivity(BaseModel):
    """Agent activity log for debugging and tracing"""
    agent_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    activity_type: str
    description: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

