from langchain_core.runnables import RunnableConfig
from agents.codex.graphs.reviewer.models import ReviewerState, TestVerdict, TestResult
from shared.logger import get_logger
from e2b_code_interpreter import Sandbox
import os
import re

logger = get_logger('codex.reviewer')


def parse_pytest_output(output: str, error: str) -> list[TestResult]:
    """
    Parse pytest output to extract individual test results.
    This is a simplified parser - in production you might want more robust parsing.
    """
    test_results = []
    
    # Try to extract test results from output
    # Pytest format: "test_file.py::test_name PASSED" or "FAILED"
    test_pattern = r'(test_\w+)\s+(PASSED|FAILED)'
    matches = re.findall(test_pattern, output + error)
    
    for test_name, status in matches:
        test_results.append(TestResult(
            test_name=test_name,
            passed=(status == "PASSED"),
            error=None if status == "PASSED" else "Test failed",
            output=""
        ))
    
    # If no matches found, create a generic result
    if not test_results:
        # Check if there's any assertion error or exception
        has_error = "Error" in error or "Failed" in error or "assert" in error.lower()
        test_results.append(TestResult(
            test_name="test_execution",
            passed=not has_error and not error,
            error=error if error else None,
            output=output
        ))
    
    return test_results


def execute_tests_in_sandbox(test_code: str, repo_path: str = None) -> TestVerdict:
    """
    Execute test code in E2B sandbox and return verdict.
    
    Args:
        test_code: Python test code to execute
        repo_path: Optional path to repository (for context)
    
    Returns:
        TestVerdict with execution results
    """
    logger.info("Executing tests in E2B sandbox...")
    
    # Check E2B API key
    if not os.getenv('E2B_API_KEY'):
        logger.error("E2B_API_KEY not found in environment")
        return TestVerdict(
            all_passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            test_results=[],
            execution_summary="E2B_API_KEY not configured",
            issues=["E2B_API_KEY environment variable not set"]
        )
    
    sandbox = None
    try:
        # Create E2B sandbox
        logger.info("Creating E2B sandbox for test execution...")
        sandbox = Sandbox.create()
        logger.info(f"Sandbox created: {sandbox.sandbox_id}")
        
        # If repo_path is provided, we could potentially copy files to sandbox
        # For now, we'll just execute the test code directly
        
        # Execute the test code
        logger.info("Executing test code in sandbox...")
        execution = sandbox.run_code(test_code)
        
        output = execution.text or ""
        error = str(execution.error) if execution.error else ""
        
        logger.info(f"Test execution completed")
        logger.info(f"Output: {output[:200]}...")
        if error:
            logger.warning(f"Error: {error[:200]}...")
        
        # Parse test results
        test_results = parse_pytest_output(output, error)
        
        # Calculate verdict
        total_tests = len(test_results)
        passed_tests = sum(1 for t in test_results if t.passed)
        failed_tests = total_tests - passed_tests
        all_passed = (failed_tests == 0 and total_tests > 0)
        
        # Build execution summary
        execution_summary = f"Executed {total_tests} test(s): {passed_tests} passed, {failed_tests} failed"
        if output:
            execution_summary += f"\n\nOutput:\n{output}"
        if error:
            execution_summary += f"\n\nErrors:\n{error}"
        
        # Collect issues
        issues = []
        for test in test_results:
            if not test.passed and test.error:
                issues.append(f"{test.test_name}: {test.error}")
        
        verdict = TestVerdict(
            all_passed=all_passed,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=test_results,
            execution_summary=execution_summary,
            issues=issues
        )
        
        logger.info(f"Test verdict - All passed: {all_passed}, Total: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
        
        return verdict
    
    except Exception as e:
        logger.error(f"Failed to execute tests in sandbox: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return TestVerdict(
            all_passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            test_results=[],
            execution_summary=f"Sandbox execution failed: {str(e)}",
            issues=[f"Sandbox error: {str(e)}"]
        )
    
    finally:
        # Always kill the sandbox
        if sandbox:
            try:
                logger.info(f"Killing E2B sandbox: {sandbox.sandbox_id}")
                sandbox.kill()
                logger.info("E2B sandbox killed successfully")
            except Exception as e:
                logger.error(f"Failed to kill E2B sandbox: {str(e)}")


def test_executor_node(state: ReviewerState, config: RunnableConfig) -> dict:
    """
    Test Executor node - executes generated tests in E2B sandbox.
    Returns test verdict with pass/fail results.
    """
    logger.info("Test Executor node - Running tests")
    
    # Get test suite from state
    test_suite = state.test_suite
    
    if not test_suite or not test_suite.test_code:
        logger.warning("No test suite found in state")
        return {
            "test_verdict": TestVerdict(
                all_passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                test_results=[],
                execution_summary="No test suite to execute",
                issues=["No test code generated"]
            ).model_dump()
        }
    
    # Execute tests
    test_code = test_suite.test_code
    repo_path = state.repo_path
    
    logger.info(f"Executing {test_suite.test_count} test(s)...")
    verdict = execute_tests_in_sandbox(test_code, repo_path)
    
    # Update state with verdict
    return {
        "test_verdict": verdict.model_dump(),
        "success": verdict.all_passed
    }

