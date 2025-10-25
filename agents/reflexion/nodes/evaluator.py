from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState, Verdict
from shared.logger import get_logger
from shared.llm import get_llm
from langchain.tools import tool
import json
import os
import re
from e2b_code_interpreter import Sandbox
from langchain.agents import create_agent

logger = get_logger('reflexion_agent')


EVALUATOR_PROMPT = """You are a Code Evaluator Agent in a reflexion system - an expert test engineer who validates code quality through comprehensive unit testing executed in an E2B sandbox.

YOUR ROLE:
- You have access to tools for code execution in E2B sandbox
- Design and run executable unit tests against the Actor's code
- Provide objective pass/fail verdict based on actual execution results
- Identify bugs, errors, and quality issues from real test runs
- Return a structured verdict with passed status, score, reasoning, issues, and execution results

AVAILABLE TOOLS:
1. **extract_code_from_response(response)**: Extract code from actor's markdown response
2. **execute_code_in_sandbox(code)**: Execute Python code in E2B sandbox
3. **run_test_in_sandbox(test_code, test_name)**: Run a specific unit test in sandbox

EVALUATION PROCESS:
1. **Extract Code**: Use extract_code_from_response to get the actor's code
2. **Load Code**: Use execute_code_in_sandbox to verify code loads without errors
3. **Design Tests**: Think about what tests are needed:
   - Happy path scenarios (normal inputs)
   - Edge cases (empty, null, boundary values)
   - Error scenarios (invalid inputs, exceptions)
   - Performance considerations (if relevant)
   - Security issues (if applicable)
4. **Run Tests**: Use run_test_in_sandbox for each test you design
5. **Analyze Results**: Review actual execution output and failures
6. **Provide Structured Verdict**: Return verdict with all required fields filled

CODE EVALUATION CRITERIA:
1. **Correctness**: Does the code produce correct outputs for all test cases?
2. **Completeness**: Does it handle all requirements from user query?
3. **Edge Cases**: Does it handle boundary conditions and special cases?
4. **Error Handling**: Does it gracefully handle invalid inputs/errors?
5. **Code Quality**: Is it clean, readable, and well-structured?
6. **Best Practices**: Does it follow coding standards and patterns?
7. **Security**: Are there any security vulnerabilities?
8. **Performance**: Is it reasonably efficient?

# NOTE: The sandbox is PERSISTED throughout your execution. Once you load the actor's code,
# it remains available in the sandbox for all subsequent test runs.

HOW TO USE TOOLS:
1. First, call extract_code_from_response(actor_response) to get the code
2. Then, call execute_code_in_sandbox(code) to load and verify it works
   - This loads the code into a PERSISTENT sandbox
   - All functions/variables defined here will be available for subsequent tests
3. For each test you want to run:
   - Write test code as a string with assert statements
   - Call run_test_in_sandbox(test_code, "test_name") - just pass the test code, NOT the actor's code
   - The actor's code is already loaded in the sandbox from step 2
   - Check the result (passed/failed, error messages)
4. Track all test results and analyze failures
5. Return a structured verdict with all required fields

EXAMPLE TEST WORKFLOW:
```
# Step 1: Extract code
code = call extract_code_from_response(actor_response)

# Step 2: Load code into persistent sandbox
result = call execute_code_in_sandbox(code)

# Step 3: Run tests (code is already loaded, just pass test assertions)
test1 = call run_test_in_sandbox("result = merge_intervals([]); assert result == []", "test_empty")
test2 = call run_test_in_sandbox("result = merge_intervals([(1,3),(2,6)]); assert result == [(1,6)]", "test_overlap")
...

# Step 4: Analyze and return structured verdict
```

TEST CATEGORIES TO COVER:
- **Happy path**: Normal inputs that match the expected use case
- **Edge cases**: Empty/null/None inputs, boundary values (min/max, zero, negative)
- **Type validation**: Type mismatches and invalid input types
- **Error handling**: Invalid inputs that should raise exceptions
- **Requirements**: Specific behaviors mentioned in user query
- **Performance**: Large datasets (if applicable)
- **Security**: Input validation, sanitization (if applicable)

EVALUATION GUIDELINES:
- Be thorough but realistic - tests should validate requirements
- Write executable tests using the tools provided
- Document specific failures when tests don't pass
- Provide actionable feedback based on actual execution results
- Pass only if code handles all critical test cases correctly
- Make autonomous decisions about which tests to run

IMPORTANT: You are a ReAct agent - think, use tools, observe results, and iterate as needed.
Use the tools multiple times if necessary to thoroughly test the code.
"""


def create_execute_code_tool(sandbox: Sandbox):
    """Factory function to create execute_code tool with sandbox closure"""
    @tool
    async def execute_code_in_sandbox(code: str) -> str:
        """
        Execute Python code in E2B sandbox and return the execution result.
        Use this to load and test the actor's code before running unit tests.
        The code will be persisted in the sandbox for subsequent test runs.
        
        Args:
            code: Python code to execute (string)
        
        Returns:
            JSON string with execution result including output, errors, and status
        """
        try:
            logger.info("Executing code in persisted E2B sandbox...")
            execution = sandbox.run_code(code)
            
            if execution.error:
                logger.error(f"Code execution error: {execution.error}")
                return json.dumps({
                    "success": False,
                    "error": str(execution.error),
                    "output": execution.text or ""
                })
            
            logger.info("Code executed successfully and persisted in sandbox")
            return json.dumps({
                "success": True,
                "error": None,
                "output": execution.text or ""
            })
        
        except Exception as e:
            logger.error(f"Sandbox execution failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"Sandbox error: {str(e)}",
                "output": ""
            })
    
    return execute_code_in_sandbox


def create_run_test_tool(sandbox: Sandbox):
    """Factory function to create run_test tool with sandbox closure"""
    @tool
    async def run_test_in_sandbox(test_code: str, test_name: str) -> str:
        """
        Run a single unit test in the persisted E2B sandbox.
        The actor's code is already loaded in the sandbox, so you only need to pass test code.
        
        Args:
            test_code: Python test code with assert statements
            test_name: Name of the test (for logging)
        
        Returns:
            JSON string with test result including passed status, error, and output
        """
        try:
            logger.info(f"Running test '{test_name}' in persisted sandbox...")
            execution = sandbox.run_code(test_code)
            
            if execution.error:
                # Test failed
                logger.warning(f"Test '{test_name}' failed: {execution.error}")
                return json.dumps({
                    "test_name": test_name,
                    "passed": False,
                    "error": str(execution.error),
                    "output": execution.text or ""
                })
            
            # Test passed
            logger.info(f"Test '{test_name}' passed ✓")
            return json.dumps({
                "test_name": test_name,
                "passed": True,
                "error": None,
                "output": execution.text or ""
            })
        
        except Exception as e:
            logger.error(f"Exception running test '{test_name}': {str(e)}")
            return json.dumps({
                "test_name": test_name,
                "passed": False,
                "error": str(e),
                "output": ""
            })
    
    return run_test_in_sandbox


@tool
def extract_code_from_response(response: str) -> str:
    """
    Extract code blocks from the actor's response.
    Looks for markdown code blocks (```python...```) or returns the whole response.
    
    Args:
        response: The actor's full response text
    
    Returns:
        Extracted Python code as a string
    """
    # Try to find code blocks with triple backticks
    code_block_pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        # Return the first (or concatenated) code block
        extracted = '\n\n'.join(matches)
        logger.info(f"Extracted {len(matches)} code block(s), total {len(extracted)} chars")
        return extracted
    
    # If no code blocks found, return the whole response
    logger.info(f"No markdown code blocks found, using full response ({len(response)} chars)")
    return response


async def evaluator_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Evaluator ReAct agent creates and executes tests in E2B sandbox.
    Uses async tools to extract code, run tests, and generate verdict.
    Returns a structured verdict based on actual test execution results.
    
    This node creates a PERSISTENT E2B sandbox for the entire evaluation:
    1. Creates sandbox at start
    2. Loads actor's code into sandbox (persists for all tests)
    3. Runs multiple tests using the same sandbox
    4. Kills sandbox at the end
    """
    logger.info("Evaluator node - Agent starting evaluation")
    
    # Extract user query and actor response from message history
    user_message = ""
    actor_response = ""
    
    # Get the last human message (user query)
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
            actor_response = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    if not user_message or not actor_response:
        logger.warning("Missing user message or actor response in history")
        # Return failed verdict if we can't evaluate
        return {
            "evaluator_verdict": Verdict(
                passed=False,
                score=0.0,
                reasoning="Cannot evaluate: missing user query or actor response",
                issues=["Missing required information for evaluation"]
            ).model_dump()
        }
    
    # Check if E2B_API_KEY is set
    if not os.getenv('E2B_API_KEY'):
        logger.error("E2B_API_KEY not found in environment")
        return {
            "evaluator_verdict": Verdict(
                passed=False,
                score=0.0,
                reasoning="E2B_API_KEY not configured in environment",
                issues=["E2B_API_KEY environment variable not set"]
            ).model_dump()
        }
    
    # Create persistent E2B sandbox for this evaluation
    sandbox = None
    try:
        logger.info("Creating persistent E2B sandbox for evaluation...")
        sandbox = Sandbox.create()
        logger.info(f"Sandbox created successfully: {sandbox.sandbox_id}")
        
        # Create tools that use this specific sandbox instance
        execute_code_tool = create_execute_code_tool(sandbox)
        run_test_tool = create_run_test_tool(sandbox)
        
        evaluator_tools = [
            extract_code_from_response,
            execute_code_tool,
            run_test_tool
        ]
        
        # Create evaluator agent with sandbox-bound tools
        evaluator_agent = create_agent(
            model=get_llm(temperature=0.0),
            tools=evaluator_tools,
            system_prompt=EVALUATOR_PROMPT,
            response_format=Verdict
        )
        
        # Build evaluation task for the agent
        evaluation_task = f"""
**EVALUATION TASK**

User's Coding Request:
{user_message}

Actor's Code Response:
{actor_response}

**YOUR MISSION:**
1. Extract the code from the actor's response using extract_code_from_response tool
2. Execute the code in E2B sandbox using execute_code_in_sandbox tool to verify it loads
   - This loads the code into the PERSISTENT sandbox
3. Design and run comprehensive unit tests using run_test_in_sandbox tool:
   - Test happy path (normal inputs)
   - Test edge cases (empty, null, boundary values)
   - Test error scenarios (invalid inputs, exceptions)
   - Test requirements from user's query
   - The actor's code is ALREADY LOADED in the sandbox, so only pass test code
4. Track all test results (pass/fail, errors)
5. Provide your final verdict as a structured response with:
   - passed: true/false
   - score: 0.0-1.0
   - reasoning: detailed explanation with test results
   - issues: list of specific problems if any
   - execution_results: summary of all test executions

Use the available tools to execute real tests in E2B sandbox. Be thorough and test all critical scenarios.
"""
        
        # Invoke the evaluator agent asynchronously
        try:
            agent_result = await evaluator_agent.ainvoke(
                {"messages": [HumanMessage(content=evaluation_task)]},
                config=config
            )
            
            # Extract the structured Verdict from agent result
            # When using response_format, create_agent returns the structured output in "structured_response" key
            if isinstance(agent_result, dict) and "structured_response" in agent_result:
                # LangChain v1 pattern: structured output in "structured_response" key
                structured_data = agent_result["structured_response"]
                verdict = structured_data if isinstance(structured_data, Verdict) else Verdict(**structured_data)
                logger.info(f"Extracted structured response - Passed: {verdict.passed}, Score: {verdict.score}")
            elif isinstance(agent_result, Verdict):
                # Direct Verdict instance
                verdict = agent_result
            elif isinstance(agent_result, dict):
                # Try to construct Verdict from dict
                if "passed" in agent_result:
                    verdict = Verdict(**agent_result)
                else:
                    # Fallback: verdict might be nested somewhere, or we need to parse messages
                    raise ValueError(f"Could not find structured_response in agent result. Keys: {list(agent_result.keys())}")
            else:
                raise ValueError(f"Unexpected agent result type: {type(agent_result)}")
            
            logger.info(f"Evaluator agent completed - Passed: {verdict.passed}, Score: {verdict.score}")
            
        except Exception as e:
            logger.error(f"Evaluator agent failed: {str(e)}")
            import traceback
            traceback.print_exc()
            verdict = Verdict(
                passed=False,
                score=0.0,
                reasoning=f"Evaluation agent error: {str(e)}",
                issues=[f"Agent execution failed: {str(e)}"]
            )
        
    except Exception as e:
        logger.error(f"Failed to create E2B sandbox: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "evaluator_verdict": Verdict(
                passed=False,
                score=0.0,
                reasoning=f"Failed to create E2B sandbox: {str(e)}",
                issues=[f"Sandbox creation failed: {str(e)}"]
            ).model_dump()
        }
    
    finally:
        # Always kill the sandbox after evaluation
        if sandbox:
            try:
                logger.info(f"Killing E2B sandbox: {sandbox.sandbox_id}")
                sandbox.kill()
                logger.info("E2B sandbox killed successfully")
            except Exception as e:
                logger.error(f"Failed to kill E2B sandbox: {str(e)}")
    
    # Store verdict in state but don't add to message history
    # User only sees actor's responses, not internal evaluations
    return {
        "evaluator_verdict": verdict.model_dump()
    }

