#!/usr/bin/env python3
"""
E16: Active Observability - Progressive Trace Summarization Experiment

Compares ReAct agent with/without progressive trace summarization (every 3 tool calls).
Refactored to use LangChain's create_agent and ToolRuntime to access state.
"""
import os
import sys
import json
import asyncio
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Annotated, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))
load_dotenv(project_root / ".env", override=True)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

# Import using relative path since folder name has dashes
import importlib.util
smart_trace_path = Path(__file__).parent / "smart_trace_logic.py"
spec = importlib.util.spec_from_file_location("smart_trace_logic", smart_trace_path)
smart_trace_logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smart_trace_logic)
get_smart_summary_from_messages = smart_trace_logic.get_smart_summary_from_messages

# Load HumanEval tasks (hard problems only)
TASK_FILE = Path(__file__).parent / "humaneval_task.json"
with open(TASK_FILE) as f:
    all_tasks = json.load(f)
    HARD_TASKS = [t for t in all_tasks if t.get("difficulty") == "hard"]  # Filter hard problems

@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    condition: str  # "baseline" or "treatment"
    seed: int
    task_id: str = ""  # HumanEval task ID
    success: bool = False
    tool_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    execution_time: float = 0.0
    trace_id: str = ""
    error: str = ""

@dataclass
class ExperimentContext:
    """Context for the experiment runtime."""
    task_id: str
    condition: str
    seed: int

def create_python_executor_tool(task):
    """Create a tool that executes Python code for a specific task."""
    @tool
    def execute_python(code: str) -> str:
        """
        Execute Python code and run tests. 
        Input should be complete Python code including the function definition.
        """
        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Create a safe execution environment
        exec_globals = {}
        try:
            exec(code, exec_globals)
        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        
        # Get the function
        func_name = task["entry_point"]
        if func_name not in exec_globals:
            return f"⚠️ Function '{func_name}' not found in code."
        
        func = exec_globals[func_name]
        # Run tests
        test_code = task["test"]
        test_globals = {"candidate": func, **exec_globals}
        try:
            exec(test_code, test_globals)
            return "✅ All tests passed!"
        except Exception as e:
            return f"❌ Test Failed: {str(e)}"
    
    return execute_python

@tool
def get_progress_summary(
    runtime: ToolRuntime[ExperimentContext]
) -> str:
    """
    Get a smart summary of your progress so far. 
    This analyzes your current execution history to summarize successful steps and highlight any failures. 
    Call this periodically (e.g., after every 3 tool calls) to stay aware of your progress and avoid repeating mistakes.
    """
    # Access context via runtime
    task_id = runtime.context.task_id
    print(f"🔍 [DEBUG] get_progress_summary called for task: {task_id}")
    
    # Access state messages via runtime.state (LangGraph feature)
    # Note: create_agent uses a state dictionary with "messages" key
    try:
        # Depending on the runtime, state might be a dict or object
        if isinstance(runtime.state, dict):
            messages = runtime.state.get("messages", [])
        else:
            # Assume it's an object with messages attribute
            messages = getattr(runtime.state, "messages", [])
            
        print(f"   Messages found: {len(messages)}")
    except Exception as e:
        print(f"   ❌ Error accessing state: {e}")
        messages = []

    if not messages:
        return "⚠️ No messages available yet. Continue working and try again after some tool calls."
    
    # Use local message-based summarization
    summary = get_smart_summary_from_messages(messages)
    
    print(f"\n📝 [GET_PROGRESS_SUMMARY OUTPUT]:\n{'-'*40}\n{summary}\n{'-'*40}\n")
    
    return summary

async def run_single_experiment(
    condition: str,
    seed: int,
    task: dict,
    summarization_enabled: bool = False
) -> ExperimentResult:
    """Run a single experiment (baseline or treatment)."""
    print(f"\n{'='*60}")
    print(f"🧪 Condition: {condition.upper()} | Seed: {seed}")
    print(f"{'='*60}")
    
    # Setup
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.0, reasoning_effort="minimal")
    python_tool = create_python_executor_tool(task)
    
    # Create tools list
    tools = [python_tool]
    
    # Add progress summary tool if summarization is enabled
    if summarization_enabled:
        tools.append(get_progress_summary)
        print("✅ Progress summary tool ADDED (uses ToolRuntime state access)")
    
    # System prompt
    base_system_prompt = """You are a Python programming assistant. 
Solve the given problem step by step. Write complete, correct Python code.
Use the execute_python tool to test your solution."""
    
    if summarization_enabled:
        system_prompt = f"""{base_system_prompt}

IMPORTANT: After every 3 tool calls (excluding get_progress_summary), call the get_progress_summary tool to review your progress.

CRITICAL: If ANY tool call fails (returns an error), you MUST call get_progress_summary IMMEDIATELY to analyze the failure before retrying.

This helps you:
- Understand what you've accomplished so far
- Identify any failures or errors
- Avoid repeating mistakes
- Stay focused on the task

Example: If you've called execute_python 3 times, call get_progress_summary before continuing.
Example: If execute_python returns an error, call get_progress_summary immediately."""
    else:
        system_prompt = base_system_prompt
    
    # Setup LangFuse tracing
    thread_id = str(uuid.uuid4())
    trace_context = TraceContext(
        session_id=thread_id,
        user_id=f"e16-{condition}-{seed}",
        tags=["e16", condition, f"seed-{seed}"]
    )
    langfuse_handler = CallbackHandler(trace_context=trace_context)
    
    # Create Agent using create_agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        context_schema=ExperimentContext if summarization_enabled else None
    )
    
    print(f"\n🔍 Agent created with {len(tools)} tool(s)")
    if summarization_enabled:
        print(f"   Tools: {[t.name for t in tools]}")
    
    # Prepare task prompt
    task_prompt = f"""Solve this Python problem:

{task['prompt']}

Write a complete solution and test it using the execute_python tool."""
    
    start_time = time.time()
    trace_id = None
    
    # Run agent
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "recursion_limit": 20
    }
    
    print(f"\n🚀 Invoking agent...")
    
    initial_messages = [HumanMessage(content=task_prompt)]
    
    # Context for runtime
    run_context = None
    if summarization_enabled:
        run_context = ExperimentContext(
            task_id=task['task_id'],
            condition=condition,
            seed=seed
        )
    
    # Execute
    if run_context:
        result = await agent.ainvoke(
            {"messages": initial_messages},
            config=config,
            context=run_context
        )
    else:
        result = await agent.ainvoke(
            {"messages": initial_messages},
            config=config
        )
    
    # Process results
    messages = result.get("messages", [])
    
    print(f"\n✅ Agent execution completed.")
    print(f"📝 Final message history: {len(messages)} messages")
    
    # Debug: Print message summary
    print(f"\n📝 Message count: {len(messages)}")
    for i, msg in enumerate(messages[-3:]):
        msg_type = type(msg).__name__
        if hasattr(msg, 'content'):
            content_preview = str(msg.content)[:100] if msg.content else "None"
            print(f"  [{i}] {msg_type}: {content_preview}...")
    
    # Count tool calls
    tool_calls = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls += len(msg.tool_calls)

    print(f"🔧 Tool calls: {tool_calls} total")
    
    # Check success
    success = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            if "✅ All tests passed!" in msg.content:
                success = True
                print("✅ Success detected in ToolMessage!")
                break
    
    if not success:
        print(f"❌ Success not detected.")
    
    execution_time = time.time() - start_time
    
    return ExperimentResult(
        condition=condition,
        seed=seed,
        task_id=task['task_id'],
        success=success,
        tool_calls=tool_calls,
        tokens_input=0, # Placeholder
        tokens_output=0,
        tokens_total=0,
        execution_time=execution_time,
        trace_id="langfuse_auto"
    )

async def main():
    """Run the full experiment."""
    print("\n" + "="*70)
    print("E16: Active Observability - Progressive Trace Summarization (ToolRuntime State Access)")
    print("="*70)
    print(f"Tasks: {len(HARD_TASKS)} hard HumanEval problems")
    print("COMPARISON: Baseline vs Treatment")
    print()
    
    n_seeds = 1
    results = []
    
    # Run experiment for each task
    for task_idx, task in enumerate(HARD_TASKS, 1):
        print(f"\n{'='*70}")
        print(f"TASK {task_idx}/{len(HARD_TASKS)}: {task['task_id']} ({task['difficulty']})")
        print(f"Problem: {task['entry_point']}")
        print(f"{'='*70}")
        
        # Run baseline (no summarization)
        print(f"\n📊 Running BASELINE condition (no summarization)...")
        for seed in range(1, n_seeds + 1):
            print(f"\n⚠️ Attempting baseline run {seed}...")
            result = None
            import traceback
            try:
                result = await run_single_experiment("baseline", seed, task, summarization_enabled=False)
                print(f"✅ Baseline run {seed} completed successfully")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"❌ Baseline run {seed} failed with error: {error_msg[:200]}")
                result = ExperimentResult(
                    condition="baseline",
                    seed=seed,
                    task_id=task['task_id'],
                    success=False,
                    tool_calls=0,
                    execution_time=0.0,
                    trace_id="error",
                    error=error_msg[:1000]
                )
            
            if result:
                results.append(result)
            await asyncio.sleep(1)
        
        # Run treatment (with summarization)
        print(f"\n📊 Running TREATMENT condition (with summarization)...")
        for seed in range(1, n_seeds + 1):
            print(f"\n⚠️ Attempting treatment run {seed}...")
            result = None
            import traceback
            try:
                result = await run_single_experiment("treatment", seed, task, summarization_enabled=True)
                print(f"✅ Treatment run {seed} completed successfully")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"❌ Treatment run {seed} failed with error: {error_msg[:200]}")
                result = ExperimentResult(
                    condition="treatment",
                    seed=seed,
                    task_id=task['task_id'],
                    success=False,
                    tool_calls=0,
                    execution_time=0.0,
                    trace_id="error",
                    error=error_msg[:1000]
                )
            
            if result:
                results.append(result)
            await asyncio.sleep(1)
    
    # Save results
    results_file = Path(__file__).parent / f"results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    baseline_results = [r for r in results if r.condition == "baseline"]
    treatment_results = [r for r in results if r.condition == "treatment"]
    
    def print_stats(name: str, results_list: List[ExperimentResult]):
        if not results_list:
            print(f"\n{name}: No runs completed.")
            return
        success_rate = sum(1 for r in results_list if r.success) / len(results_list) * 100
        avg_tool_calls = sum(r.tool_calls for r in results_list) / len(results_list)
        avg_time = sum(r.execution_time for r in results_list) / len(results_list)
        
        print(f"\n{name}:")
        print(f"  Success Rate: {success_rate:.1f}% ({sum(1 for r in results_list if r.success)}/{len(results_list)})")
        print(f"  Avg Tool Calls: {avg_tool_calls:.1f}")
        print(f"  Avg Execution Time: {avg_time:.2f}s")
        
        # Show errors if any
        errors = [r for r in results_list if r.error]
        if errors:
            print(f"  Errors: {len(errors)} run(s) had errors")
            for r in errors:
                print(f"    Seed {r.seed}: {r.error[:100]}...")
    
    print_stats("BASELINE", baseline_results)
    print_stats("TREATMENT", treatment_results)
    
    # Per-task breakdown
    if len(HARD_TASKS) > 1:
        print(f"\n📊 PER-TASK BREAKDOWN:")
        for task in HARD_TASKS:
            task_id = task['task_id']
            task_baseline = [r for r in baseline_results if r.task_id == task_id]
            task_treatment = [r for r in treatment_results if r.task_id == task_id]
            
            print(f"\n  {task_id}:")
            if task_baseline:
                baseline_success = sum(1 for r in task_baseline if r.success) / len(task_baseline) * 100
                print(f"    Baseline: {baseline_success:.1f}% success ({sum(1 for r in task_baseline if r.success)}/{len(task_baseline)})")
            if task_treatment:
                treatment_success = sum(1 for r in task_treatment if r.success) / len(task_treatment) * 100
                print(f"    Treatment: {treatment_success:.1f}% success ({sum(1 for r in task_treatment if r.success)}/{len(task_treatment)})")

if __name__ == "__main__":
    asyncio.run(main())
