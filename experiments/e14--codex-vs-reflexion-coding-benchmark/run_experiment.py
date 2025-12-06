#!/usr/bin/env python3
"""
Experiment E14: Codex vs Predictive Reflexion - Coding Benchmark Comparison

This experiment compares three different agent approaches on HumanEval coding tasks:
1. Codex: Developer agent with sandbox, file editing, codebase exploration
2. Predictive Reflexion: Self-improving agent with prediction learning

Design:
- 5 HumanEval tasks run sequentially
- All agents run same tasks
- Compare success rates, code quality, and execution time
"""

import os
import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Check required packages
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    sys.exit(1)

# Add paths for imports
seer_root = Path(__file__).parent.parent.parent
reflexion_root = seer_root.parent / "reflexion"
if str(seer_root) not in sys.path:
    sys.path.insert(0, str(seer_root))
if str(reflexion_root) not in sys.path:
    sys.path.insert(0, str(reflexion_root))

try:
    from reflexion import create_reflexion
    from langchain.agents import create_agent
    from langchain_core.tools import tool
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    sys.exit(1)

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "gpt-5-mini"  # For Reflexion and Codex
CODEX_MODEL = "gpt-5-mini"  # For Codex (using same model since gpt-5-codex not available)
MAX_ROUNDS = 3  # Max rounds per task (Reflexion)
MAX_ATTEMPTS = 3  # Max attempts per task (Codex)
NUM_TASKS = 2  # 2 easy tasks for testing
TEMPERATURE = 0.0
RUN_CODEX = True  # Set to False if E2B_API_KEY not available
RUN_REFLEXION = True
RUN_REACT = False  # Removed - focusing on Reflexion vs Codex only

# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class ExecutionMetrics:
    """Track comprehensive execution metrics."""
    task_id: str
    agent_type: str  # "codex" or "reflexion"
    
    # Success metrics
    success: bool = False
    eval_score: float = 0.0
    
    # Execution metrics
    execution_time: float = 0.0
    rounds_or_attempts: int = 0
    
    # Tool usage
    tool_calls_count: int = 0
    tool_calls_by_name: Dict[str, int] = field(default_factory=dict)
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Code quality
    code_generated: Optional[str] = None
    test_pass_rate: float = 0.0
    
    # Prediction metrics (Reflexion only)
    calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Memory metrics (Reflexion only)
    memory_retrievals: int = 0
    memory_creations: int = 0
    memory_retrieval_queries: List[str] = field(default_factory=list)
    memory_created_contexts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "success": self.success,
            "eval_score": self.eval_score,
            "execution_time": self.execution_time,
            "rounds_or_attempts": self.rounds_or_attempts,
            "tool_calls_count": self.tool_calls_count,
            "tool_calls_by_name": self.tool_calls_by_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "code_generated": self.code_generated[:500] if self.code_generated else None,
            "test_pass_rate": self.test_pass_rate,
            "calibration_error": self.calibration_error,
            "brier_score": self.brier_score,
            "memory_retrievals": self.memory_retrievals,
            "memory_creations": self.memory_creations,
            "memory_retrieval_queries": self.memory_retrieval_queries,
            "memory_created_contexts": self.memory_created_contexts,
        }


# ============================================================================
# Memory Tracking Wrapper
# ============================================================================

class MemoryTrackingWrapper:
    """Wrapper around Neo4jMemoryStore to track memory usage."""
    
    def __init__(self, memory_store, metrics: ExecutionMetrics):
        self.memory_store = memory_store
        self.metrics = metrics
    
    def save(self, memory) -> str:
        """Track memory saves."""
        memory_id = self.memory_store.save(memory)
        self.metrics.memory_creations += 1
        # Track context of created memory
        context = getattr(memory, 'context', 'unknown')
        self.metrics.memory_created_contexts.append(context)
        return memory_id
    
    def retrieve(self, query, record_access: bool = True):
        """Track memory retrievals."""
        results = self.memory_store.retrieve(query, record_access)
        self.metrics.memory_retrievals += 1
        # Track query details
        query_str = str(query.semantic_query) if hasattr(query, 'semantic_query') and query.semantic_query else str(query)
        self.metrics.memory_retrieval_queries.append(query_str[:100])  # Limit length
        return results
    
    def retrieve_relevant_memories(self, agent_id: str, query_context: str, llm_model=None, limit: int = 5, enable_multihop: bool = False):
        """Track relevant memory retrievals."""
        results = self.memory_store.retrieve_relevant_memories(
            agent_id, query_context, llm_model, limit, enable_multihop
        )
        self.metrics.memory_retrievals += 1
        self.metrics.memory_retrieval_queries.append(query_context[:100])  # Track query context
        return results
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped memory store."""
        return getattr(self.memory_store, name)


# ============================================================================
# HumanEval Tasks
# ============================================================================

def load_humaneval_tasks():
    """Load HumanEval tasks."""
    script_dir = Path(__file__).parent
    humaneval_file = script_dir / "humaneval_tasks.json"
    
    if humaneval_file.exists():
        with open(humaneval_file, 'r') as f:
            problems = json.load(f)
        return [
            {
                "task_id": p["task_id"],
                "instruction": p["prompt"].strip(),
                "test": p.get("test"),
                "entry_point": p.get("entry_point"),
                "expected_success": True,
                "complexity": p.get("difficulty", "unknown"),
                "source": "HumanEval"
            }
            for p in problems
        ]
    
    return []

ALL_TASKS = load_humaneval_tasks()
# Filter to only easy tasks for now
EASY_TASKS = [t for t in ALL_TASKS if t.get("complexity") == "easy"]
TEST_TASKS = EASY_TASKS[:NUM_TASKS] if len(EASY_TASKS) >= NUM_TASKS else EASY_TASKS

# ============================================================================
# Code Execution Evaluator
# ============================================================================

def evaluate_code_execution(code: str, test_code: str, entry_point: str) -> tuple[float, str]:
    """Execute code and run HumanEval tests."""
    try:
        from reflexion.evaluator.code_executor import CodeExecutor
        executor = CodeExecutor()
        score, reasoning = executor.execute_and_evaluate(
            code=code,
            task_description=f"Implement {entry_point} function",
            humaneval_test=test_code,
            entry_point=entry_point
        )
        return score, reasoning
    except Exception as e:
        return 0.0, f"Evaluation error: {str(e)}"


# ============================================================================
# Agent Implementations
# ============================================================================

async def run_reflexion_agent(task: Dict[str, Any], model, tools, memory_store=None) -> Dict[str, Any]:
    """Run Predictive Reflexion agent."""
    agent = create_reflexion(
        model=model,
        tools=tools,
        prompt="""You are a Python code generation agent with prediction capabilities.

CRITICAL REQUIREMENTS:
1. You MUST generate complete, executable Python code that solves the given problem.
2. The code MUST include the function definition with the exact name specified (entry_point).
3. You MUST use the test_code_with_humaneval tool to verify your code passes all tests.
4. Only submit code that passes all tests.

PREDICTION REQUIREMENT:
Before generating code, make a prediction about task completion.
Format: "I predict: [claim] with probability [0.0-1.0]"
Example: "I predict: The code will pass all tests with probability 0.8"

WORKFLOW:
1. Make a prediction
2. Generate complete Python code (include imports, function definition, proper logic)
3. Use test_code_with_humaneval tool to verify your code
4. If tests fail, fix the code and test again
5. Only when all tests pass, submit your solution""",
        simple_mode=True,
        max_rounds=MAX_ROUNDS,
        eval_threshold=0.8,
        memory_store=memory_store
    )
    
    instruction = task["instruction"]
    if task.get("test") and task.get("entry_point"):
        full_instruction = f"""{instruction}

CRITICAL: You must generate a complete Python function named '{task['entry_point']}' that passes these tests.

Test cases:
{task['test']}

IMPORTANT:
1. Generate the complete function code (not just a description)
2. Use the test_code_with_humaneval tool with:
   - code: Your complete function code
   - test_code: The test cases above (starting with 'def check(candidate):')
   - entry_point: '{task['entry_point']}'
3. Only submit when all tests pass."""
    else:
        full_instruction = instruction
    
    from langchain_core.runnables import RunnableConfig
    
    config_dict = {
        "agent_id": f"e14_reflexion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "max_rounds": MAX_ROUNDS
    }
    if task.get("test") and task.get("entry_point"):
        config_dict["humaneval_test"] = task["test"]
        config_dict["entry_point"] = task["entry_point"]
    
    # Create RunnableConfig with increased recursion limit
    runnable_config = RunnableConfig(
        recursion_limit=150,  # Increase from default 50 to handle complex tasks
        configurable=config_dict
    )
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=full_instruction)], "current_round": 0},
            config=runnable_config
        )
    except Exception as e:
        # If recursion limit hit, return partial result
        if "recursion" in str(e).lower() or "limit" in str(e).lower():
            print(f"  ⚠️  Recursion limit reached: {str(e)}")
            # Return failure result
            return {
                "output": f"Error: Recursion limit reached - {str(e)}",
                "messages": [],
                "current_round": MAX_ROUNDS,
                "evaluation": None,
                "predictions": [],
                "calibration_metrics": None
            }
        raise
    
    # Extract code from result
    output = ""
    for msg in result.get("messages", []):
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                output += content + "\n"
    
    return {
        "output": output,
        "messages": result.get("messages", []),
        "current_round": result.get("current_round", 0),
        "evaluation": result.get("evaluation"),
        "predictions": result.get("predictions", []),
        "calibration_metrics": result.get("calibration_metrics")
    }


async def run_codex_agent(task: Dict[str, Any], model, tools) -> Dict[str, Any]:
    """
    Run Codex agent (simplified version for coding tasks).
    
    Note: Full Codex requires sandbox setup. This is a simplified version
    that uses Codex's developer node approach but in a simpler context.
    """
    # For now, create a simplified Codex-like agent that uses file editing tools
    # In a full implementation, this would use the actual Codex graph with sandbox
    
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    
    @tool
    def write_python_file(filename: str, code: str) -> str:
        """Write Python code to a file."""
        try:
            # Create a temporary directory for this task
            temp_dir = Path("/tmp/e14_codex") / task["task_id"]
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = temp_dir / filename
            file_path.write_text(code)
            return f"✅ Code written to {file_path}"
        except Exception as e:
            return f"❌ Error writing file: {str(e)}"
    
    @tool
    def run_python_test(test_code: str, entry_point: str) -> str:
        """Run Python tests."""
        try:
            # Import the function from the file
            temp_dir = Path("/tmp/e14_codex") / task["task_id"]
            sys.path.insert(0, str(temp_dir))
            
            # Execute test
            score, reasoning = evaluate_code_execution(
                code="",  # Will be loaded from file
                test_code=test_code,
                entry_point=entry_point
            )
            return f"Test result: {reasoning} (Score: {score:.2f})"
        except Exception as e:
            return f"❌ Test error: {str(e)}"
    
    # Combine Codex-specific tools with standard tools
    codex_tools = [write_python_file, run_python_test] + tools
    
    agent = create_agent(
        model=model,
        tools=codex_tools,
        system_prompt="""You are a developer agent (Codex) that writes code to files and tests it.

WORKFLOW:
1. Write the complete Python function to a file (e.g., solution.py)
2. Use run_python_test or test_code_with_humaneval to verify your code
3. If tests fail, fix the code and test again
4. Only when all tests pass, submit your solution"""
    )
    
    instruction = task["instruction"]
    if task.get("test") and task.get("entry_point"):
        full_instruction = f"""{instruction}

CRITICAL: You must generate a complete Python function named '{task['entry_point']}' that passes these tests.

Test cases:
{task['test']}

IMPORTANT:
1. Write the complete function code to a file (e.g., solution.py)
2. Use test_code_with_humaneval tool to verify your code passes all tests
3. Only submit when all tests pass."""
    else:
        full_instruction = instruction
    
    result = agent.invoke({"messages": [HumanMessage(content=full_instruction)]})
    
    # Extract code from result - look for code in tool calls and tool messages
    output = ""
    messages = result.get("messages", [])
    
    # Strategy 1: Extract code from test_code_with_humaneval tool call arguments
    from langchain_core.messages import ToolMessage
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get('name', '')
                    args = tool_call.get('args', {})
                else:
                    tool_name = getattr(tool_call, 'name', '')
                    args = getattr(tool_call, 'args', {})
                
                # If this is test_code_with_humaneval, extract the code parameter
                if tool_name == 'test_code_with_humaneval' and isinstance(args, dict):
                    code_param = args.get('code', '')
                    if code_param and 'def ' in code_param:
                        output = code_param
                        break
        
        # Strategy 2: Extract code from ToolMessage responses
        elif isinstance(msg, ToolMessage):
            content = str(msg.content) if hasattr(msg, 'content') else ''
            # If tool response contains code (look for function definitions)
            if 'def ' in content and task.get("entry_point") and task["entry_point"] in content:
                # Try to extract code block
                import re
                code_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
                if code_match:
                    output = code_match.group(1)
                elif 'def ' + task["entry_point"] in content:
                    # Extract function definition
                    func_match = re.search(r'(def\s+' + re.escape(task["entry_point"]) + r'[^:]*:.*?)(?=\n\s*def\s+|\n\s*class\s+|\Z)', content, re.DOTALL)
                    if func_match:
                        output = func_match.group(1)
    
    # Strategy 3: Fallback - extract from file if written
    if not output or 'def ' not in output:
        temp_dir = Path("/tmp/e14_codex") / task["task_id"]
        if temp_dir.exists():
            for file_path in temp_dir.glob("*.py"):
                try:
                    file_content = file_path.read_text()
                    if task.get("entry_point") and f'def {task["entry_point"]}' in file_content:
                        output = file_content
                        break
                except:
                    pass
    
    # Strategy 4: Last resort - extract from message content (but clean it)
    if not output or 'def ' not in output:
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                # Look for code blocks
                import re
                code_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
                if code_match:
                    output = code_match.group(1)
                    break
                # Look for function definition
                elif task.get("entry_point") and f'def {task["entry_point"]}' in content:
                    func_match = re.search(r'(def\s+' + re.escape(task["entry_point"]) + r'[^:]*:.*?)(?=\n\s*CRITICAL|\n\s*Test cases|\n\s*IMPORTANT|\Z)', content, re.DOTALL)
                    if func_match:
                        output = func_match.group(1)
                        break
    
    return {
        "output": output,
        "messages": messages,
        "attempts": 1
    }


# ============================================================================
# Tools
# ============================================================================

@tool
def execute_python_code(code: str) -> str:
    """Execute Python code and return the result. Use this to test your code before submitting."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def test_code_with_humaneval(code: str, test_code: str, entry_point: str) -> str:
    """Test your generated code against HumanEval test cases.
    
    Args:
        code: Your generated Python code (must include the function definition)
        test_code: The HumanEval test code (contains 'def check(candidate): ...')
        entry_point: Name of the function to test (e.g., 'has_close_elements')
    
    Returns:
        Test results: 'All tests passed' or error message
    """
    try:
        score, reasoning = evaluate_code_execution(code, test_code, entry_point)
        if score >= 0.99:
            return f"✅ All tests passed! {reasoning}"
        else:
            return f"❌ Tests failed: {reasoning}"
    except Exception as e:
        return f"Error testing code: {str(e)}"


# ============================================================================
# Main Experiment Runner
# ============================================================================

async def run_task_with_agent(
    task: Dict[str, Any],
    agent_type: str,
    model,
    tools,
    memory_store=None,
    task_num: int = 0
) -> Dict[str, Any]:
    """Run a task with specified agent type."""
    
    print(f"\n{'='*80}")
    print(f"Task {task_num + 1}/{len(TEST_TASKS)}: {task['task_id']} | Agent: {agent_type.upper()}")
    print(f"{'='*80}")
    
    metrics = ExecutionMetrics(
        task_id=task["task_id"],
        agent_type=agent_type
    )
    
    start_time = datetime.now()
    
    try:
        # Wrap memory store with tracking for Reflexion
        tracked_memory_store = None
        if agent_type == "reflexion" and memory_store:
            tracked_memory_store = MemoryTrackingWrapper(memory_store, metrics)
        
        # Run appropriate agent
        if agent_type == "reflexion":
            result = await run_reflexion_agent(task, model, tools, tracked_memory_store)
            metrics.rounds_or_attempts = result.get("current_round", 1)
            if result.get("calibration_metrics"):
                cal_metrics = result["calibration_metrics"]
                # Handle both dict and Pydantic model
                if hasattr(cal_metrics, 'calibration_error'):
                    metrics.calibration_error = cal_metrics.calibration_error
                    metrics.brier_score = cal_metrics.brier_score
                elif isinstance(cal_metrics, dict):
                    metrics.calibration_error = cal_metrics.get("calibration_error")
                    metrics.brier_score = cal_metrics.get("brier_score")
        elif agent_type == "codex":
            codex_model = ChatOpenAI(model=CODEX_MODEL, temperature=TEMPERATURE)
            result = await run_codex_agent(task, codex_model, tools)
            metrics.rounds_or_attempts = result.get("attempts", 1)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        metrics.execution_time = elapsed_time
        
        # Extract code and evaluate
        output = result.get("output", "")
        metrics.code_generated = output
        
        # Evaluate code execution
        if task.get("test") and task.get("entry_point"):
            eval_score, eval_reasoning = evaluate_code_execution(
                code=output,
                test_code=task["test"],
                entry_point=task["entry_point"]
            )
            metrics.eval_score = eval_score
            metrics.success = eval_score >= 0.99
            metrics.test_pass_rate = eval_score
        else:
            # Fallback evaluation
            metrics.eval_score = 0.5 if output else 0.0
            metrics.success = False
        
        # Count tool calls - check both messages and run_trace for Reflexion
        messages = result.get("messages", [])
        run_trace = result.get("run_trace", [])
        
        tool_call_count = 0
        tool_calls_by_name = {}
        
        # Check messages (standard location)
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                    tool_calls_by_name[tool_name] = tool_calls_by_name.get(tool_name, 0) + 1
        
        # Check run_trace (Reflexion stores execution trace here)
        for msg in run_trace:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                    tool_calls_by_name[tool_name] = tool_calls_by_name.get(tool_name, 0) + 1
        
        metrics.tool_calls_count = tool_call_count
        metrics.tool_calls_by_name = tool_calls_by_name
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        for msg in messages:
            if hasattr(msg, 'response_metadata') and msg.response_metadata:
                usage = msg.response_metadata.get('token_usage', {})
                input_tokens += usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                output_tokens += usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
        
        metrics.input_tokens = input_tokens
        metrics.output_tokens = output_tokens
        metrics.total_tokens = input_tokens + output_tokens
        
        print(f"  ✅ Completed | Success: {metrics.success} | Score: {metrics.eval_score:.2f} | Time: {elapsed_time:.1f}s")
        
        return {
            "task_id": task["task_id"],
            "agent_type": agent_type,
            "success": metrics.success,
            "eval_score": metrics.eval_score,
            "execution_time": metrics.execution_time,
            "rounds_or_attempts": metrics.rounds_or_attempts,
            "tool_calls": metrics.tool_calls_count,
            "code_generated": output[:500],
            "metrics": metrics.to_dict()
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "task_id": task["task_id"],
            "agent_type": agent_type,
            "error": str(e),
            "success": False,
            "eval_score": 0.0
        }


async def main():
    """Main experiment execution."""
    
    print("="*80)
    print("Experiment E14: Codex vs Predictive Reflexion")
    print("="*80)
    print(f"Tasks: {len(TEST_TASKS)} (2 easy + 2 medium + 2 hard)")
    print(f"Agents: Codex={RUN_CODEX}, Reflexion={RUN_REFLEXION}")
    print(f"Model: {MODEL_NAME} (both agents)")
    print("="*80)
    
    # Check environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    # Initialize models
    model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    print(f"✅ Using model: {MODEL_NAME}")
    
    # Initialize tools
    tools = [execute_python_code, test_code_with_humaneval]
    
    # Initialize memory store (optional, for Reflexion)
    memory_store = None
    try:
        from reflexion.memory import Neo4jMemoryStore
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if neo4j_password:
            memory_store = Neo4jMemoryStore(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                username=os.getenv("NEO4J_USER", "neo4j"),
                password=neo4j_password
            )
            print(f"✅ Memory store initialized (cross-task learning enabled)")
    except Exception as e:
        print(f"⚠️  Memory store disabled: {e}")
    
    # Run experiments
    all_results = []
    
    for task_num, task in enumerate(TEST_TASKS):
        # Run each agent type (Reflexion and Codex only)
        if RUN_REFLEXION:
            result = await run_task_with_agent(
                task, "reflexion", model, tools, memory_store, task_num
            )
            all_results.append(result)
        
        if RUN_CODEX:
            result = await run_task_with_agent(
                task, "codex", model, tools, None, task_num
            )
            all_results.append(result)
    
    # Calculate summary statistics
    codex_results = [r for r in all_results if r.get("agent_type") == "codex"]
    reflexion_results = [r for r in all_results if r.get("agent_type") == "reflexion"]
    
    def calc_stats(results):
        if not results:
            return {}
        stats = {
            "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
            "avg_score": sum(r.get("eval_score", 0) for r in results) / len(results),
            "avg_time": sum(r.get("execution_time", 0) for r in results) / len(results),
            "avg_rounds": sum(r.get("rounds_or_attempts", 0) for r in results) / len(results),
            "total_tasks": len(results)
        }
        # Add memory metrics if available
        memory_retrievals = [r.get("metrics", {}).get("memory_retrievals", 0) for r in results]
        memory_creations = [r.get("metrics", {}).get("memory_creations", 0) for r in results]
        if memory_retrievals:
            stats["avg_memory_retrievals"] = sum(memory_retrievals) / len(memory_retrievals)
            stats["total_memory_retrievals"] = sum(memory_retrievals)
        if memory_creations:
            stats["avg_memory_creations"] = sum(memory_creations) / len(memory_creations)
            stats["total_memory_creations"] = sum(memory_creations)
        return stats
    
    codex_stats = calc_stats(codex_results)
    reflexion_stats = calc_stats(reflexion_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / f"results_{timestamp}.json"
    
    summary = {
        "experiment": "E14 Codex vs Predictive Reflexion",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "codex_model": CODEX_MODEL,
        "num_tasks": len(TEST_TASKS),
        "codex_results": codex_results,
        "reflexion_results": reflexion_results,
        "summary": {
            "codex": codex_stats,
            "reflexion": reflexion_stats
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary Statistics:")
    
    if codex_stats:
        print(f"\n  Codex:")
        print(f"    Success Rate: {codex_stats['success_rate']:.1%}")
        print(f"    Avg Score: {codex_stats['avg_score']:.3f}")
        print(f"    Avg Time: {codex_stats['avg_time']:.1f}s")
    
    if reflexion_stats:
        print(f"\n  Predictive Reflexion:")
        print(f"    Success Rate: {reflexion_stats['success_rate']:.1%}")
        print(f"    Avg Score: {reflexion_stats['avg_score']:.3f}")
        print(f"    Avg Time: {reflexion_stats['avg_time']:.1f}s")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Update EXPERIMENT.md
    update_experiment_md(results_file, summary)
    
    return summary


def update_experiment_md(results_file: Path, summary: Dict[str, Any]):
    """Update EXPERIMENT.md with results table."""
    experiment_md = Path(__file__).parent / "EXPERIMENT.md"
    
    if not experiment_md.exists():
        print(f"⚠️  EXPERIMENT.md not found, skipping update")
        return
    
    with open(experiment_md, 'r') as f:
        content = f.read()
    
    # Extract results
    codex_results = summary.get("codex_results", [])
    reflexion_results = summary.get("reflexion_results", [])
    
    # Build table rows
    table_rows = []
    task_ids = set()
    for r in codex_results + reflexion_results:
        task_ids.add(r.get("task_id", "unknown"))
    
    for task_id in sorted(task_ids):
        task_short = task_id.split('/')[-1]
        
        # Find results for this task
        codex_r = next((r for r in codex_results if r.get("task_id") == task_id), None)
        reflexion_r = next((r for r in reflexion_results if r.get("task_id") == task_id), None)
        
        # Codex row
        if codex_r:
            r = codex_r
            success = "✓" if r.get("success", False) else "✗"
            attempts = r.get("rounds_or_attempts", 0)
            tools = r.get("tool_calls", 0)
            eval_score = r.get("eval_score", 0)
            time_s = r.get("execution_time", 0)
            table_rows.append(f"| {task_short} | Codex | {success} | {attempts} | {tools} | {eval_score:.2f} | {time_s:.1f} |")
        
        # Reflexion row
        if reflexion_r:
            r = reflexion_r
            success = "✓" if r.get("success", False) else "✗"
            rounds = r.get("rounds_or_attempts", 0)
            tools = r.get("tool_calls", 0)
            eval_score = r.get("eval_score", 0)
            time_s = r.get("execution_time", 0)
            table_rows.append(f"| {task_short} | Reflexion | {success} | {rounds} | {tools} | {eval_score:.2f} | {time_s:.1f} |")
        
    
    # Build table
    table = "| Task | Agent | Success | Rounds/Attempts | Tools | Eval Score | Time (s) |\n"
    table += "|------|-------|---------|----------------|-------|------------|----------|\n"
    table += "\n".join(table_rows)
    
    # Calculate summary stats
    codex_stats = summary.get("summary", {}).get("codex", {})
    reflexion_stats = summary.get("summary", {}).get("reflexion", {})
    
    summary_stats = f"""### Summary Statistics

- **Codex Success Rate:** {codex_stats.get('success_rate', 0):.1%} ({codex_stats.get('total_tasks', 0)} tasks)
- **Predictive Reflexion Success Rate:** {reflexion_stats.get('success_rate', 0):.1%} ({reflexion_stats.get('total_tasks', 0)} tasks)
- **Average Execution Time (Codex):** {codex_stats.get('avg_time', 0):.1f}s
- **Average Execution Time (Reflexion):** {reflexion_stats.get('avg_time', 0):.1f}s"""
    
    # Replace results section
    import re
    pattern = r"## Results\s+### Per-Task Performance Matrix\s+\|.*?\|.*?\|\n\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|\n\n### Summary Statistics.*?(?=\n## |$)"
    replacement = f"## Results\n\n### Per-Task Performance Matrix\n\n{table}\n\n{summary_stats}"
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(experiment_md, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated EXPERIMENT.md with results table")


if __name__ == "__main__":
    asyncio.run(main())

