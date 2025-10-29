from pydantic import BaseModel, Field
from typing import List, Optional
from agents.codex.common.state import BaseState


class TestResult(BaseModel):
    """Individual test execution result"""
    test_name: str = Field(description="Name of the test case")
    passed: bool = Field(description="Whether the test passed")
    error: Optional[str] = Field(default=None, description="Error message if test failed")
    output: str = Field(default="", description="Test output")


class TestVerdict(BaseModel):
    """Overall test execution verdict"""
    all_passed: bool = Field(description="Whether all tests passed")
    total_tests: int = Field(description="Total number of tests run")
    passed_tests: int = Field(description="Number of tests that passed")
    failed_tests: int = Field(description="Number of tests that failed")
    test_results: List[TestResult] = Field(default_factory=list, description="Individual test results")
    execution_summary: str = Field(default="", description="Summary of test execution")
    issues: List[str] = Field(default_factory=list, description="Specific issues found")


class TestSuite(BaseModel):
    """Collection of test cases"""
    test_code: str = Field(description="Complete test code to execute")
    test_count: int = Field(description="Number of test cases")
    description: str = Field(description="Description of what tests cover")


class ReviewerState(BaseState):
    """State for reviewer graph - extends BaseState with test-specific fields"""
    
    # Test generation
    test_suite: Optional[TestSuite] = Field(default=None, description="Generated test suite")
    
    # Test execution
    test_verdict: TestVerdict = Field(
        default=TestVerdict(
            all_passed=False, 
            total_tests=0, 
            passed_tests=0, 
            failed_tests=0
        ), 
        description="Test execution verdict"
    )
    
    # Tracking
    current_attempt: int = Field(default=0, description="Current test attempt number")
    max_test_attempts: int = Field(default=2, description="Maximum test generation attempts")

