from pydantic import BaseModel, Field
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




class GithubContext(BaseModel):
    repo_url: str
    branch_name: str

class SandboxContext(BaseModel):
    sandbox_id: str
    working_directory: str
    working_branch: str

class UserContext(BaseModel):
    user_expectation: str

class TestingContext(BaseModel):
    test_results: List[TestResult]



class CodexInput(BaseModel):
    github_context: GithubContext
    sandbox_context: SandboxContext
    user_context: UserContext
    testing_context: TestingContext

class CodexOutput(BaseModel):
    agent_updated: bool = Field(False, description="Whether the agent was updated")
    new_branch_name: Optional[str] = Field(None, description="The name of the new branch")


