import json
import os
import re
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolCallId
from langchain.agents import create_agent

# E2B Code Interpreter for sandbox execution
from e2b_code_interpreter import Sandbox

from agents.reflexion.prompts import ACTOR_PROMPT, EVALUATOR_PROMPT, REFLECTION_PROMPT
from shared.llm import get_llm
from shared.logger import get_logger

# Get logger for reflexion agent
logger = get_logger('reflexion_agent')

# Memory store namespace for reflexion feedback
MEMORY_NAMESPACE = ("reflexion", "feedback")


# Pydantic models for structured outputs
class Verdict(BaseModel):
    """Evaluator's verdict on the Actor's response"""
    passed: bool = Field(description="Whether the response passes evaluation")
    score: float = Field(ge=0.0, le=1.0, description="Quality score from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed explanation of the judgment")
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    execution_results: str = Field(default="", description="Actual execution results from sandbox")


class Reflection(BaseModel):
    """Reflection agent's feedback for improvement"""
    key_issues: list[str] = Field(description="Main problems identified")
    suggestions: list[str] = Field(description="Specific, actionable improvements")
    focus_areas: list[str] = Field(description="What to prioritize in next attempt")
    examples: list[str] = Field(default_factory=list, description="Concrete examples if helpful")


class ReflexionState(TypedDict, total=False):
    """State for the reflexion agent graph"""
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Current attempt tracking
    current_attempt: int
    max_attempts: int
    
    # Evaluator's verdict
    evaluator_verdict: Verdict | None
    
    # Final result status
    success: bool
    
    # Memory store key for this conversation (e.g., user_id or domain)
    memory_key: str


# Memory store tools
@tool
def get_reflection_memory(memory_key: str, runtime: ToolRuntime) -> str:
    """
    Retrieve reflection feedback from persistent memory store.
    Returns all reflections for the given memory_key across all threads.
    """
    try:
        store = runtime.store
        if not store:
            logger.warning("No store available, returning empty memory")
            return json.dumps([])
        
        # Search for all reflections under this memory key
        items = store.search(MEMORY_NAMESPACE, filter={"memory_key": memory_key})
        
        reflections = []
        for item in items:
            if item.value:
                reflections.append(item.value)
        
        logger.info(f"Retrieved {len(reflections)} reflections from memory for key: {memory_key}")
        return json.dumps(reflections)
        
    except Exception as e:
        logger.error(f"Failed to retrieve memory: {str(e)}")
        return json.dumps([])


@tool
def store_reflection(memory_key: str, reflection_data: str, runtime: ToolRuntime) -> str:
    """
    Store reflection feedback in persistent memory store.
    This persists across threads and can be retrieved by memory_key.
    """
    try:
        store = runtime.store
        if not store:
            logger.warning("No store available, cannot persist reflection")
            return json.dumps({"success": False, "error": "No store available"})
        
        reflection_dict = json.loads(reflection_data)
        
        # Generate unique key for this reflection (timestamp-based)
        import time
        reflection_id = f"{memory_key}_{int(time.time() * 1000)}"
        
        # Store with metadata for filtering
        store.put(
            MEMORY_NAMESPACE,
            reflection_id,
            {"memory_key": memory_key, **reflection_dict}
        )
        
        logger.info(f"Stored reflection in memory: {reflection_id}")
        return json.dumps({"success": True, "reflection_id": reflection_id})
        
    except Exception as e:
        logger.error(f"Failed to store reflection: {str(e)}")
        return json.dumps({"success": False, "error": str(e)})


# Tools for Evaluator ReAct Agent

@tool
async def execute_code_in_sandbox(code: str) -> str:
    """
    Execute Python code in E2B sandbox and return the execution result.
    Use this to load and test the actor's code before running unit tests.
    
    Args:
        code: Python code to execute (string)
    
    Returns:
        JSON string with execution result including output, errors, and status
    """
    # Check if E2B_API_KEY is set (E2B SDK reads it automatically from env)
    if not os.getenv('E2B_API_KEY'):
        logger.error("E2B_API_KEY not found in environment")
        return json.dumps({
            "success": False,
            "error": "E2B_API_KEY not configured in environment",
            "output": ""
        })
    
    try:
        # E2B Sandbox.create() is the correct way to instantiate
        logger.info("Creating E2B sandbox...")
        sandbox = Sandbox.create()
        
        try:
            logger.info("Executing code in E2B sandbox...")
            execution =  sandbox.run_code(code)
            
            if execution.error:
                logger.error(f"Code execution error: {execution.error}")
                return json.dumps({
                    "success": False,
                    "error": str(execution.error),
                    "output": execution.text or ""
                })
            
            logger.info("Code executed successfully")
            return json.dumps({
                "success": True,
                "error": None,
                "output": execution.text or ""
            })
        finally:
            # Always close the sandbox
            logger.info("skipping E2B sandbox close...")
    
    except Exception as e:
        logger.error(f"Sandbox execution failed: {str(e)}")
        return json.dumps({
            "success": False,
            "error": f"Sandbox error: {str(e)}",
            "output": ""
        })


@tool
async def run_test_in_sandbox(test_code: str, test_name: str) -> str:
    """
    Run a single unit test in E2B sandbox.
    The actor's code should already be loaded in the sandbox context.
    
    Args:
        test_code: Python test code with assert statements
        test_name: Name of the test (for logging)
    
    Returns:
        JSON string with test result including passed status, error, and output
    """
    # Check if E2B_API_KEY is set (E2B SDK reads it automatically from env)
    if not os.getenv('E2B_API_KEY'):
        return json.dumps({
            "test_name": test_name,
            "passed": False,
            "error": "E2B_API_KEY not configured in environment",
            "output": ""
        })
    
    try:
        # E2B Sandbox.create() is the correct way to instantiate
        logger.info(f"Creating E2B sandbox for test: {test_name}")
        sandbox = Sandbox.create()
        
        try:
            logger.info(f"Running test: {test_name}")
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
        finally:
            # Always close the sandbox
            logger.info("skipping E2B sandbox close...")
    
    except Exception as e:
        logger.error(f"Exception running test '{test_name}': {str(e)}")
        return json.dumps({
            "test_name": test_name,
            "passed": False,
            "error": str(e),
            "output": ""
        })


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


@tool
def create_test_summary(test_results: str) -> str:
    """
    Create a summary of test results for the final verdict.
    
    Args:
        test_results: JSON string containing list of test results
    
    Returns:
        Human-readable summary with pass/fail counts and details
    """
    try:
        results = json.loads(test_results)
        if not isinstance(results, list):
            return "Invalid test results format"
        
        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)
        failed = total - passed
        
        summary = f"Test Summary: {passed}/{total} tests passed\n\n"
        
        for r in results:
            status = "✓ PASSED" if r.get("passed") else "✗ FAILED"
            summary += f"{status}: {r.get('test_name', 'unnamed')}\n"
            if not r.get("passed") and r.get("error"):
                summary += f"  Error: {r.get('error')}\n"
        
        return summary
    
    except Exception as e:
        return f"Error creating summary: {str(e)}"


# Node functions
def actor_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Actor generates a response to the user's message.
    Uses reflection memory from persistent store to improve responses.
    """
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    memory_key = state.get("memory_key", "default")
    messages = state.get("messages", [])
    
    logger.info(f"Actor node - Attempt {current_attempt}/{max_attempts}")
    
    # Extract user's latest message from message history
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    if not user_message:
        logger.warning("No user message found in history")
        user_message = "Please provide a response."
    
    # Retrieve reflection memory from persistent store
    try:
        store = config.get("configurable", {}).get("store")
        if store:
            items = list(store.search(MEMORY_NAMESPACE, filter={"memory_key": memory_key}))
            reflection_memory = [item.value for item in items if item.value]
            logger.info(f"Retrieved {len(reflection_memory)} reflections from persistent memory")
        else:
            reflection_memory = []
            logger.warning("No store available in config")
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        reflection_memory = []
    
    # Build memory context from reflection feedback
    memory_context = ""
    if reflection_memory:
        memory_context = "\n\n=== MEMORY_CONTEXT (Reflection Feedback from Previous Interactions) ===\n"
        for idx, reflection in enumerate(reflection_memory, 1):
            memory_context += f"\n--- Reflection {idx} ---\n"
            if isinstance(reflection, dict):
                memory_context += f"Key Issues: {', '.join(reflection.get('key_issues', []))}\n"
                memory_context += f"Suggestions: {', '.join(reflection.get('suggestions', []))}\n"
                memory_context += f"Focus Areas: {', '.join(reflection.get('focus_areas', []))}\n"
                examples = reflection.get('examples', [])
                if examples:
                    memory_context += f"Examples: {', '.join(examples)}\n"
        memory_context += "\n=== END MEMORY_CONTEXT ===\n"
    
    # Build prompt with memory
    prompt_messages = [
        SystemMessage(content=ACTOR_PROMPT),
    ]
    
    if memory_context:
        prompt_messages.append(SystemMessage(content=memory_context))
    
    prompt_messages.append(HumanMessage(content=f"User Query: {user_message}\n\nGenerate your response now, incorporating any feedback from memory."))
    
    # Generate response
    llm = get_llm(temperature=0.7)
    response = llm.invoke(prompt_messages)
    actor_response = response.content
    
    logger.info(f"Actor generated response (length: {len(actor_response)})")
    
    return {
        "messages": [AIMessage(content=actor_response)]
    }


# Create Evaluator ReAct Agent (initialized once)
evaluator_tools = [
    extract_code_from_response,
    execute_code_in_sandbox,
    run_test_in_sandbox,
    create_test_summary
]

evaluator_agent = create_agent(
    model=get_llm(temperature=0.0),
    tools=evaluator_tools,
    system_prompt=EVALUATOR_PROMPT
)


async def evaluator_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Evaluator ReAct agent creates and executes tests in E2B sandbox.
    Uses async tools to extract code, run tests, and generate verdict.
    Returns a structured verdict based on actual test execution results.
    """
    current_attempt = state.get("current_attempt", 1)
    messages = state.get("messages", [])
    
    logger.info("Evaluator node - ReAct agent starting evaluation")
    
    # Extract user query and actor response from message history
    user_message = ""
    actor_response = ""
    
    # Get the last human message (user query)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(messages):
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
3. Design and run comprehensive unit tests using run_test_in_sandbox tool:
   - Test happy path (normal inputs)
   - Test edge cases (empty, null, boundary values)
   - Test error scenarios (invalid inputs, exceptions)
   - Test requirements from user's query
4. Track all test results (pass/fail, errors)
5. Provide your final verdict in this EXACT format:

VERDICT:
Passed: [true/false]
Score: [0.0-1.0]
Reasoning: [Detailed explanation with test results]
Issues: [List specific problems if any]

Use the available tools to execute real tests in E2B sandbox. Be thorough and test all critical scenarios.
"""
    
    # Invoke the evaluator agent asynchronously
    try:
        agent_result = await evaluator_agent.ainvoke(
            {"messages": [HumanMessage(content=evaluation_task)]},
            config=config
        )
        
        # Extract the final response
        agent_messages = agent_result.get("messages", [])
        final_response = ""
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                final_response = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        logger.info(f"Evaluator agent completed with response length: {len(final_response)}")
        
        # Parse verdict from response
        verdict = parse_verdict_from_response(final_response)
        
        logger.info(f"Evaluator verdict - Passed: {verdict.passed}, Score: {verdict.score}")
        
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
    
    # Store verdict in state but don't add to message history
    # User only sees actor's responses, not internal evaluations
    return {
        "evaluator_verdict": verdict.model_dump()
    }


def parse_verdict_from_response(response: str) -> Verdict:
    """
    Parse the verdict from the evaluator agent's response.
    Looks for VERDICT section and extracts passed, score, reasoning, issues.
    """
    try:
        # Try to find VERDICT section
        if "VERDICT:" in response:
            verdict_section = response.split("VERDICT:")[1]
        else:
            verdict_section = response
        
        # Parse passed status
        passed = False
        if "Passed: true" in verdict_section or "Passed:true" in verdict_section:
            passed = True
        elif "all tests passed" in verdict_section.lower():
            passed = True
        
        # Parse score
        score = 0.0
        score_match = re.search(r'Score:\s*([0-9.]+)', verdict_section)
        if score_match:
            score = float(score_match.group(1))
        elif passed:
            score = 1.0
        
        # Parse reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:Issues:|$)', verdict_section, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response
        
        # Parse issues
        issues = []
        issues_match = re.search(r'Issues:\s*(.+)', verdict_section, re.DOTALL)
        if issues_match:
            issues_text = issues_match.group(1).strip()
            # Split by newlines or commas
            issues = [i.strip() for i in re.split(r'[\n,]', issues_text) if i.strip() and i.strip() != 'None']
        
        return Verdict(
            passed=passed,
            score=score,
            reasoning=reasoning,
            issues=issues,
            execution_results=response  # Store full response
        )
    
    except Exception as e:
        logger.error(f"Failed to parse verdict: {str(e)}")
        # Return a verdict based on the response
        return Verdict(
            passed=False,
            score=0.0,
            reasoning=f"Could not parse verdict properly. Agent response: {response[:500]}...",
            issues=["Verdict parsing failed"],
            execution_results=response
        )


def reflection_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Reflection agent provides feedback when evaluation fails.
    Stores actionable suggestions in persistent memory for the Actor.
    """
    current_attempt = state.get("current_attempt", 1)
    memory_key = state.get("memory_key", "default")
    messages = state.get("messages", [])
    verdict_dict = state.get("evaluator_verdict", {})
    
    logger.info("Reflection node - Generating feedback")
    
    # Extract user query and actor response from message history
    user_message = ""
    actor_response = ""
    
    # Get the last human message (user query)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
            actor_response = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Build reflection prompt
    reflection_prompt = f"""
User Query:
{user_message}

Actor's Response:
{actor_response}

Evaluator's Verdict:
- Passed: {verdict_dict.get('passed', False)}
- Score: {verdict_dict.get('score', 0.0)}
- Reasoning: {verdict_dict.get('reasoning', '')}
- Issues: {', '.join(verdict_dict.get('issues', []))}

Analyze why the response failed and provide constructive feedback to help the Actor improve.
Focus on specific, actionable suggestions for the next attempt.
"""
    
    prompt_messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=reflection_prompt)
    ]
    
    # Get structured reflection
    llm = get_llm(temperature=0.3).with_structured_output(Reflection)
    reflection: Reflection = llm.invoke(prompt_messages)
    
    logger.info(f"Reflection generated - {len(reflection.key_issues)} issues, {len(reflection.suggestions)} suggestions")
    
    # Store reflection in persistent memory
    try:
        store = config.get("configurable", {}).get("store")
        if store:
            import time
            reflection_id = f"{memory_key}_{int(time.time() * 1000)}"
            reflection_data = reflection.model_dump()
            reflection_data["memory_key"] = memory_key
            reflection_data["attempt"] = current_attempt
            
            store.put(
                MEMORY_NAMESPACE,
                reflection_id,
                reflection_data
            )
            logger.info(f"Stored reflection in persistent memory: {reflection_id}")
        else:
            logger.warning("No store available, reflection not persisted")
    except Exception as e:
        logger.error(f"Failed to store reflection: {e}")
    
    # Increment attempt counter
    new_attempt = current_attempt + 1
    
    # Don't add reflection to message history - it's stored in persistent memory
    # User only sees actor's responses, not internal reflections
    return {
        "current_attempt": new_attempt
    }


def finalize_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Finalize the result - either success or max attempts reached.
    No additional messages added - user sees natural conversation with actor only.
    """
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    verdict_dict = state.get("evaluator_verdict", {})
    passed = verdict_dict.get("passed", False)
    
    if passed:
        logger.info(f"✅ Success! Response passed evaluation on attempt {current_attempt}/{max_attempts}")
    else:
        logger.info(f"⚠️ Max attempts ({max_attempts}) reached. Final Score: {verdict_dict.get('score', 0.0)}")
    
    # Just return success flag - no additional messages
    # User only sees the natural conversation with actor
    return {
        "success": passed
    }


# Conditional edge function
def should_continue(state: ReflexionState) -> Literal["reflect", "finalize"]:
    """
    Decide whether to continue reflection loop or finalize.
    
    Logic:
    - If evaluator passed -> finalize
    - If max attempts reached -> finalize
    - Otherwise -> reflect and try again
    """
    verdict_dict = state.get("evaluator_verdict", {})
    passed = verdict_dict.get("passed", False)
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    
    if passed:
        logger.info("Routing to finalize - evaluation passed")
        return "finalize"
    
    if current_attempt >= max_attempts:
        logger.info(f"Routing to finalize - max attempts ({max_attempts}) reached")
        return "finalize"
    
    logger.info(f"Routing to reflect - attempt {current_attempt}/{max_attempts}, continuing loop")
    return "reflect"


# Build the graph
def build_graph():
    """
    Build the reflexion agent graph with actor, evaluator, and reflection nodes.
    
    Flow:
    START -> actor -> evaluator -> [conditional]
                                    |
                                    ├─> if passed or max attempts: finalize -> END
                                    └─> if failed and attempts < max: reflection -> actor (loop)
    """
    # Create state graph
    workflow = StateGraph(ReflexionState)
    
    # Add nodes
    workflow.add_node("actor", actor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "actor")
    workflow.add_edge("actor", "evaluator")
    
    # Conditional edge from evaluator
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "reflect": "reflection",
            "finalize": "finalize"
        }
    )
    
    # Reflection loops back to actor
    workflow.add_edge("reflection", "actor")
    
    # Finalize goes to END
    workflow.add_edge("finalize", END)
    
    # Compile - LangGraph API handles persistence automatically
    graph = workflow.compile()
    
    logger.info("Reflexion graph compiled successfully")
    return graph


# Create the graph instance
graph = build_graph()

