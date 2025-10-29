"""
Reviewer Agent for Codex

A test-driven code validation agent that generates and executes unit tests:
- Test Creator: Generates comprehensive unit tests based on request and task plan
- Test Executor: Runs tests in E2B sandbox and validates implementation
- Finalize: Reports test results with pass/fail status

The agent validates code implementation by:
1. Analyzing user request and task plan
2. Generating comprehensive test cases (happy path, edge cases, errors)
3. Executing tests in isolated E2B sandbox
4. Reporting detailed test results

Key Features:
- Automated test generation from requirements
- Sandbox-based test execution for safety
- Comprehensive test coverage (happy path, edge cases, errors)
- Detailed test reports with pass/fail status

Similar to reflexion agent but focused on validation rather than code generation.
"""

from agents.codex.graphs.reviewer.graph import graph

__all__ = ['graph']

