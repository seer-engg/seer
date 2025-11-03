from pydantic import BaseModel, Field, computed_field, ConfigDict
from typing import List, Optional


class TestResult(BaseModel):
    """Result of running a single test case"""
    test_case_id: str
    input_sent: str
    actual_output: str
    expected_behavior: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Judge's score 0-1")
    judge_reasoning: str = Field(description="Why the judge scored this way")
    model_config = ConfigDict(extra="forbid")

class GithubContext(BaseModel):
    """Context for the active GitHub repository"""
    agent_name: str = Field(..., description="The name of the agent")
    repo_url: str
    branch_name: Field(default="main", description="The name of the branch")
    model_config = ConfigDict(extra="forbid")


class TestingContext(BaseModel):
    """Context for the testing session."""
    test_results: List[TestResult]
    model_config = ConfigDict(extra="forbid")


class SandboxContext(BaseModel):
    """Context for the active sandbox session."""
    sandbox_id: str = Field(..., description="The ID of the sandbox session")
    working_directory: str = Field("", description="The working directory of the sandbox")
    working_branch: str = Field(default="main", description="The working branch of the sandbox")

    @computed_field
    @property
    def deployment_url(self) -> str:
        """helper method to get the deployment url of the sandbox"""
        return f"https://2024-{self.sandbox_id}.e2b.app"


class UserContext(BaseModel):
    """Context for the user."""
    user_id: str = Field(default='user_123', description="The ID of the user")
    user_expectation: str = Field(..., description="The user's expectation")
    model_config = ConfigDict(extra="forbid")


class CodexInput(BaseModel):
    github_context: GithubContext = Field(..., description="The GitHub context")
    sandbox_context: Optional[SandboxContext] = Field(None, description="The sandbox context")
    user_context: UserContext = Field(..., description="The user context")
    testing_context: TestingContext = Field(..., description="The testing context")

class CodexOutput(BaseModel):
    agent_updated: bool = Field(False, description="Whether the agent was updated")
    new_branch_name: Optional[str] = Field(None, description="The name of the new branch")
    updated_sandbox_context: Optional[SandboxContext] = Field(None, description="The updated sandbox context")
