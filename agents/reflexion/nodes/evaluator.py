from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState, Verdict
from shared.logger import get_logger
from shared.llm import get_llm
from langchain.tools import tool
from typing import List
from pydantic import BaseModel, Field
import json
import os
import re
from e2b_code_interpreter import Sandbox
from langchain.agents import create_agent
from agents.reflexion.mem0_client import mem0_search_memories, mem0_add_memory
from agents.reflexion.memory_artifacts import (
    parse_artifact_from_content,
    Artifact,
    build_mem0_messages_for_artifact,
)

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


# Structured test planning models
class TestCaseSpec(BaseModel):
    name: str = Field(description="Short, unique test name")
    category: str = Field(description="happy|edge|error|type|performance|security|requirement")
    importance: float = Field(ge=0.0, le=1.0, description="Relative importance 0-1")
    code: str = Field(description="Standalone Python assertions to run in sandbox. The actor's code is already loaded.")
    description: str = Field(default="", description="What this test validates")


class TestPlan(BaseModel):
    cases: List[TestCaseSpec] = Field(default_factory=list)


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


def extract_code_from_text(response: str) -> str:
    """
    Pure helper: extract code from a markdown-like response without tool semantics.
    Preference order:
    1) First ```python block
    2) Any fenced block (largest)
    3) Fallback to full text
    """
    python_pattern = r'```(?:python|py)\n(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        extracted = matches[0]
        logger.info(f"Extracted python code block, {len(extracted)} chars")
        return extracted
    any_pattern = r'```\n(.*?)```'
    matches_any = re.findall(any_pattern, response, re.DOTALL)
    if matches_any:
        extracted = max(matches_any, key=len)
        logger.info(f"Extracted generic fenced block, {len(extracted)} chars")
        return extracted
    logger.info(f"No fenced code blocks found, using full response ({len(response)} chars)")
    return response


@tool
def extract_code_from_response(response: str) -> str:
    """Tool wrapper for code extraction; returns extracted Python code as string."""
    return extract_code_from_text(response)


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
    
    # Deterministic evaluation flow using artifacts and explicit test plan
    sandbox = None
    try:
        logger.info("Creating persistent E2B sandbox for evaluation...")
        sandbox = Sandbox.create()
        logger.info(f"Sandbox created successfully: {sandbox.sandbox_id}")

        # 1) Retrieve artifacts to guide testing
        artifacts_raw = []
        try:
            mem_results = mem0_search_memories(query=user_message, user_id=state.memory_key)
            for item in mem_results:
                text = None
                if isinstance(item, dict):
                    text = item.get('content') or item.get('memory') or item.get('text')
                    if text is None:
                        data = item.get('data')
                        if isinstance(data, dict):
                            text = data.get('content') or data.get('text')
                if not text:
                    text = str(item)
                artifacts_raw.append(parse_artifact_from_content(text))
        except Exception:
            import traceback
            logger.error(traceback.format_exc())
            artifacts_raw = []

        # 2) Extract and load actor code
        actor_code = extract_code_from_text(actor_response)
        # Deterministic setup before loading user code
        sandbox.run_code(
            """
import random
random.seed(42)
try:
    import numpy as _np
    _np.random.seed(42)
except Exception:
    pass
"""
        )
        load_result = sandbox.run_code(actor_code)
        if load_result.error:
            verdict = Verdict(
                passed=False,
                score=0.0,
                reasoning=f"Actor code failed to load in sandbox: {str(load_result.error)}",
                issues=[str(load_result.error)],
                execution_results=load_result.text or "",
            )
            # early exit with cleanup in finally
            return {"evaluator_verdict": verdict.model_dump()}

        # 3) Plan tests using artifacts as guidance
        artifacts_summary_lines = []
        for a in artifacts_raw[:8]:
            rule = a.get('rule', '')
            atype = a.get('type', '')
            tags = ",".join(a.get('tags', []) or [])
            if rule:
                artifacts_summary_lines.append(f"- [{atype}] {rule} (tags: {tags})")
        artifacts_summary = "\n".join(artifacts_summary_lines)

        plan_prompt = f"""
You are a senior test engineer. Produce a compact test plan with 5-8 executable tests.
The actor's code is already loaded in a persistent sandbox. Tests should only include assertions and necessary setup.
Cover: happy, edge, error, type, and any requirement-specific behaviors from the user's request.
Emphasize lessons from artifacts.

User Request:\n{user_message}

Relevant Artifacts:\n{artifacts_summary}
"""

        test_plan_llm = get_llm(temperature=0.0).with_structured_output(TestPlan)
        test_plan: TestPlan = test_plan_llm.invoke([
            SystemMessage(content="You generate precise, minimal Python assertion-based tests."),
            HumanMessage(content=plan_prompt),
        ])

        # 4) Execute tests deterministically with timeouts
        total = 0
        passed = 0
        failures: List[str] = []
        exec_log_lines: List[str] = []

        def _wrap_test_with_timeout(code: str, seconds: int = 3) -> str:
            lines = code.splitlines()
            indented = "\n".join(["    " + l for l in lines]) if lines else ""
            return f"""
import random
random.seed(42)
try:
    import numpy as _np
    _np.random.seed(42)
except Exception:
    pass
import signal

def _timeout_handler(signum, frame):
    raise TimeoutError('test timeout')

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({seconds})
try:
{indented}
finally:
    signal.alarm(0)
"""

        executed_cases = []
        for case in test_plan.cases[:8]:
            total += 1
            exec_log_lines.append(f"RUN {case.name} [{case.category}] (importance={case.importance:.2f})")
            wrapped = _wrap_test_with_timeout(case.code)
            result = sandbox.run_code(wrapped)
            if result.error:
                failures.append(f"{case.name}: {str(result.error)}")
                exec_log_lines.append(f"FAIL {case.name}: {str(result.error)}")
                if result.text:
                    exec_log_lines.append(result.text)
            else:
                passed += 1
                exec_log_lines.append(f"PASS {case.name}")
                if result.text:
                    exec_log_lines.append(result.text)
            executed_cases.append(case)

        # Weighted scoring and category coverage
        weight_total = sum(max(0.0, min(1.0, c.importance)) for c in executed_cases) or 1.0
        # A simple way: assume failed tests contribute 0 to weighted pass sum
        passed_names = set([n.split(':')[0] for n in exec_log_lines if n.startswith('PASS ')])
        weighted_pass = 0.0
        for c in executed_cases:
            if f"PASS {c.name}" in exec_log_lines:
                weighted_pass += max(0.0, min(1.0, c.importance))
        score = weighted_pass / weight_total

        categories_present = set([getattr(c, 'category', '').strip().lower() for c in executed_cases if getattr(c, 'category', '')])
        required_categories = {"happy", "edge", "error"}
        coverage_ok = required_categories.issubset(categories_present)
        all_passed = (passed == total and total > 0) and coverage_ok

        reasoning = f"Passed {passed}/{total} tests. Weighted score={score:.2f}. "
        if failures:
            reasoning += "Failures: " + "; ".join(failures)
        missing = sorted(list(required_categories - categories_present))
        if missing:
            reasoning += f" | Missing coverage: {', '.join(missing)}"

        verdict = Verdict(
            passed=all_passed,
            score=round(score, 2),
            reasoning=reasoning,
            issues=failures,
            execution_results="\n".join(exec_log_lines)[:8000],
        )

        # 5) Failure artifact write-back (limit to 3 entries)
        try:
            user_id = state.memory_key
            failures_limited = failures[:3]
            for fmsg in failures_limited:
                # fmsg format: "<test_name>: <error>"
                parts = fmsg.split(":", 1)
                tname = parts[0].strip()
                err = parts[1].strip() if len(parts) > 1 else fmsg
                # Find category for this test
                cat = None
                for c in executed_cases:
                    if c.name == tname:
                        cat = getattr(c, 'category', '')
                        break
                tags = [t for t in [cat or '', 'testing', 'failure'] if t]
                failure_art = Artifact(
                    type="FailureSignature",
                    rule=f"{tname} fails with error: {err[:160]}",
                    tags=tags,
                )
                mem0_add_memory(messages=build_mem0_messages_for_artifact(failure_art, user_message), user_id=user_id)

                lesson_art = Artifact(
                    type="Lesson",
                    rule=f"Add handling for {cat or 'edge'} scenario that triggers: {err.split('\n',1)[0][:120]}",
                    tags=[t for t in [cat or 'edge', 'python', 'input-validation'] if t],
                )
                mem0_add_memory(messages=build_mem0_messages_for_artifact(lesson_art, user_message), user_id=user_id)
        except Exception:
            import traceback
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Evaluator execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "evaluator_verdict": Verdict(
                passed=False,
                score=0.0,
                reasoning=f"Evaluator crashed: {str(e)}",
                issues=[f"Evaluator exception: {str(e)}"]
            ).model_dump()
        }
    finally:
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

