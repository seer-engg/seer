from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.codex.graphs.reviewer.models import ReviewerState, TestSuite
from shared.logger import get_logger
from shared.llm import get_llm
from langchain.agents import create_agent
import os

logger = get_logger('codex.reviewer')


TEST_CREATOR_PROMPT = """You are a Test Generation Agent - an expert QA engineer who designs comprehensive unit tests to validate code implementations.

YOUR ROLE:
- Analyze the user's request and task plan
- Design comprehensive unit tests to validate the implementation
- Cover happy paths, edge cases, error scenarios, and requirements
- Generate executable Python test code using pytest/unittest

TEST GENERATION GUIDELINES:
1. **Understand Requirements**: Carefully read the user request and task plan
2. **Identify Test Scenarios**:
   - Happy path: Normal use cases
   - Edge cases: Boundary values, empty inputs, null/None
   - Error handling: Invalid inputs, exceptions
   - Requirements validation: Specific behaviors requested
   - Integration: Multiple components working together

3. **Write Executable Tests**:
   - Use pytest or unittest framework
   - Include assert statements for validation
   - Add descriptive test names
   - Include setup/teardown if needed
   - Handle test dependencies

4. **Coverage Goals**:
   - Test each task item in the plan
   - Cover critical functionality
   - Test error conditions
   - Validate output formats
   - Check performance (if applicable)

TEST CODE STRUCTURE:
```python
import pytest
# Add necessary imports

def test_happy_path():
    # Normal case test
    result = function_to_test(valid_input)
    assert result == expected_output
    
def test_edge_case_empty():
    # Edge case test
    result = function_to_test([])
    assert result == []
    
def test_error_handling():
    # Error scenario test
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

IMPORTANT:
- Generate COMPLETE, EXECUTABLE test code
- Include all necessary imports
- Use appropriate test framework (pytest recommended)
- Make tests independent and isolated
- Add comments explaining what each test validates
- Aim for 5-15 test cases depending on complexity

OUTPUT:
Provide a TestSuite with:
- test_code: Complete executable test code as a string
- test_count: Number of test cases
- description: Summary of what tests cover
"""


def test_creator_node(state: ReviewerState, config: RunnableConfig) -> dict:
    """
    Test Creator node - generates comprehensive unit tests based on request and task plan.
    Uses LLM with structured output to create a TestSuite.
    """
    logger.info(f"Test Creator node - Attempt {state.current_attempt + 1}/{state.max_test_attempts}")
    
    # Build context for test generation
    request = state.request
    task_plan = state.taskPlan
    repo_path = state.repo_path
    
    # Format task plan for prompt
    task_plan_text = ""
    if task_plan:
        task_plan_text = f"**Task Plan: {task_plan.title}**\n"
        for i, item in enumerate(task_plan.items, 1):
            status_marker = "✓" if item.status == "done" else "○"
            task_plan_text += f"{i}. [{status_marker}] {item.description}\n"
    
    # Build test generation prompt
    test_generation_task = f"""
**TEST GENERATION TASK**

User Request:
{request}

{task_plan_text}

Repository Path: {repo_path or "Not specified"}

**YOUR MISSION:**
Generate comprehensive unit tests to validate the implementation of this request.

1. Analyze the user request and understand what needs to be tested
2. Review the task plan to identify key functionality to validate
3. Design test cases covering:
   - Happy path scenarios
   - Edge cases (empty, null, boundary values)
   - Error handling (invalid inputs, exceptions)
   - Requirements from user request
   - Each task item in the plan

4. Generate complete, executable Python test code using pytest
5. Include necessary imports, setup, and assert statements
6. Make tests independent and self-contained

Provide a TestSuite with complete test code, count, and description.
"""
    
    # Create prompt messages
    prompt_messages = [
        SystemMessage(content=TEST_CREATOR_PROMPT),
        HumanMessage(content=test_generation_task)
    ]
    
    # Get structured test suite from LLM
    try:
        llm = get_llm(temperature=0.3).with_structured_output(TestSuite)
        test_suite: TestSuite = llm.invoke(prompt_messages)
        
        logger.info(f"Generated test suite with {test_suite.test_count} tests")
        logger.info(f"Test description: {test_suite.description}")
        
        return {
            "test_suite": test_suite.model_dump(),
            "current_attempt": state.current_attempt + 1
        }
    
    except Exception as e:
        logger.error(f"Failed to generate test suite: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return empty test suite on failure
        return {
            "test_suite": TestSuite(
                test_code="# Failed to generate tests",
                test_count=0,
                description=f"Test generation failed: {str(e)}"
            ).model_dump(),
            "current_attempt": state.current_attempt + 1
        }

