#!/usr/bin/env python3
"""
Experiment E8: React vs Reflexion for Eval Agent Reflect Node
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from shared.config import config

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

NUM_RUNS_PER_CONDITION = 2
EVAL_AGENT_URL = os.getenv("EVAL_AGENT_URL", "http://127.0.0.1:8002")
RESULTS_DIR = Path(__file__).parent / "e8_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Test scenarios - simple and complex
TEST_SCENARIOS = [
    # Scenario 1: Simple - single GitHub action
    """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should be able to create a GitHub issue when requested. 
For example, when I ask it to "create an issue titled 'Test Issue'", it should create that issue in the repository.""",
    
    # Scenario 2: Complex - multi-step GitHub + Asana workflow
    """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should be able to handle complex multi-integration workflows. Specifically:
1. When a GitHub pull request is merged, the agent should:
   - Create a GitHub issue to track the changes
   - Create a corresponding Asana task in the project
   - Link the GitHub issue and Asana task together (mention the task in the issue, or vice versa)
2. The agent should handle edge cases like:
   - PRs with multiple commits
   - PRs that reference existing Asana tasks
   - Ensuring no duplicate tasks are created
   
For example, when I say "I just merged PR #42 that fixed the authentication bug. Please create a tracking issue and link it to an Asana task", the agent should create both the GitHub issue and Asana task, and link them appropriately."""
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not set in environment")
    sys.exit(1)

# Initialize Langfuse client for distributed tracing
LANGFUSE_CLIENT = Langfuse(
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_base_url
) if config.langfuse_secret_key else None

# ============================================================================
# Helper Functions
# ============================================================================

def load_checkpoint() -> Dict[str, Any]:
    """Load existing checkpoint if it exists."""
    checkpoint_file = RESULTS_DIR / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}", flush=True)
            return {"react_results": [], "reflexion_results": []}
    return {"react_results": [], "reflexion_results": []}

def save_checkpoint(react_results: List[Dict[str, Any]], reflexion_results: List[Dict[str, Any]]):
    """Save checkpoint with all results so far."""
    checkpoint_file = RESULTS_DIR / "checkpoint.json"
    checkpoint = {
        "experiment": "e8--react-vs-reflexion-eval-agent",
        "last_updated": datetime.now().isoformat(),
        "react_results": react_results,
        "reflexion_results": reflexion_results,
        "config": {
            "num_runs_per_condition": NUM_RUNS_PER_CONDITION,
            "eval_agent_url": EVAL_AGENT_URL
        }
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)
    print(f"💾 Checkpoint saved to {checkpoint_file}", flush=True)

def save_run_to_json(run_data: Dict[str, Any], run_num: int, condition: str):
    """Save run data to JSON file."""
    filename = RESULTS_DIR / f"run_{run_num:03d}_{condition}.json"
    with open(filename, 'w') as f:
        json.dump(run_data, f, indent=2, default=str)
    print(f"💾 Saved run {run_num} ({condition}) to {filename}", flush=True)

def extract_metrics_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant metrics from eval agent state."""
    # Handle both dict and Pydantic model serialization
    def safe_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default) if hasattr(obj, key) else default
    
    def safe_len(obj):
        if obj is None:
            return 0
        return len(obj) if hasattr(obj, "__len__") else 0
    
    latest_results = safe_get(state, "latest_results", [])
    dataset_examples = safe_get(state, "dataset_examples", [])
    
    # Count passed/failed tests
    passed_count = 0
    failed_count = 0
    test_ids = []
    test_inputs = []
    
    for result in latest_results:
        passed = safe_get(result, "passed", False)
        if isinstance(passed, bool):
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        
        # Extract test info
        dataset_example = safe_get(result, "dataset_example")
        if dataset_example:
            example_id = safe_get(dataset_example, "example_id", "")
            input_msg = safe_get(dataset_example, "input_message", "")
            if example_id:
                test_ids.append(str(example_id))
            if input_msg:
                test_inputs.append(str(input_msg)[:100])
    
    metrics = {
        "attempts": safe_get(state, "attempts", 0),
        "num_test_cases": safe_len(dataset_examples),
        "num_results": safe_len(latest_results),
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "test_ids": test_ids,
        "test_inputs": test_inputs,
    }
    
    return metrics

async def invoke_eval_agent(
    architecture: str,
    run_num: int,
    scenario: str,
    user_id: str = "experiment_user"
) -> Dict[str, Any]:
    """
    Invoke the eval agent with specified architecture.
    
    NOTE: The eval agent must be restarted with EVAL_AGENT_ARCHITECTURE env var
    set before running this experiment. The config is loaded at module import time.
    
    Args:
        architecture: 'react' or 'reflexion'
        run_num: Run number for tracking
        user_id: User ID for the eval agent
    
    Returns:
        Dict with run results and metrics
    """
    print(f"\n{'='*60}", flush=True)
    print(f"RUN {run_num}: {architecture.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    
    start_time = time.time()
    
    try:
        # Note: Environment variable must be set before eval agent starts
        # We'll verify it matches what we expect
        current_arch = os.environ.get("EVAL_AGENT_ARCHITECTURE", "reflexion")
        if current_arch.lower() != architecture.lower():
            print(f"⚠️  Warning: EVAL_AGENT_ARCHITECTURE={current_arch} but expected {architecture}")
            print(f"   Please restart eval agent with: EVAL_AGENT_ARCHITECTURE={architecture} python run.py")
        
        # Connect to eval agent with retry logic
        max_retries = 3
        retry_delay = 5
        sync_client = None
        remote_graph = None
        
        for attempt in range(max_retries):
            try:
                sync_client = get_sync_client(url=EVAL_AGENT_URL)
                # Test connection by creating a thread
                test_thread = await asyncio.to_thread(sync_client.threads.create)
                trace_id = None
                langfuse_handler = None
                if LANGFUSE_CLIENT:
                    trace_id = Langfuse.create_trace_id(seed=test_thread["thread_id"])
                    langfuse_handler = CallbackHandler()
                
                remote_graph = RemoteGraph(
                    "eval_agent",
                    sync_client=sync_client,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️  Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...", flush=True)
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to connect to eval agent after {max_retries} attempts: {str(e)}")
        
        # Create thread
        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_id = thread["thread_id"]
        thread_cfg = {"configurable": {"thread_id": thread_id}}
        
        # Setup Langfuse trace context for distributed tracing
        trace_id = None
        langfuse_handler = None
        if LANGFUSE_CLIENT:
            trace_id = Langfuse.create_trace_id(seed=thread_id)
            langfuse_handler = CallbackHandler()
            thread_cfg["metadata"] = thread_cfg.get("metadata", {})
            thread_cfg["metadata"]["langfuse_trace_id"] = trace_id
        
        print(f"📝 Thread ID: {thread_id}", flush=True)
        print(f"🚀 Invoking eval agent with architecture: {architecture}", flush=True)
        
        # Prepare input - eval agent will extract context from the message
        input_payload = {
            "messages": [{"role": "user", "content": scenario}]
        }
        
        # Invoke agent - use streaming with retry logic for connection resilience
        print(f"  📡 Using streaming for long-running operation...", flush=True)
        max_stream_retries = 2
        result = None
        
        for stream_attempt in range(max_stream_retries):
            try:
                # Use stream and collect final state - more resilient to connection issues
                def collect_stream():
                    """Collect all stream events and return final state."""
                    events = []
                    try:
                        # Add Langfuse callback handler if available
                        stream_cfg = thread_cfg.copy()
                        if langfuse_handler:
                            stream_cfg["callbacks"] = [langfuse_handler]
                        
                        for event in remote_graph.stream(input_payload, stream_cfg):
                            events.append(event)
                            # Print progress for long operations
                            if len(events) % 10 == 0:
                                print(f"    ... {len(events)} events received ...", flush=True)
                    except Exception as e:
                        # If we got some events before error, return them
                        if events:
                            print(f"    ⚠️  Stream interrupted after {len(events)} events: {e}", flush=True)
                        raise
                    return events
                
                stream_task = asyncio.to_thread(collect_stream)
                stream_results = await asyncio.wait_for(stream_task, timeout=600)  # 10 min timeout for full workflow
                
                # Get the last result from stream
                # Stream returns iterator of (node_name, state_dict) tuples
                if stream_results:
                    # Get the last event which should be the final state
                    last_event = stream_results[-1]
                    
                    # Handle tuple format: (node_name, state_dict)
                    if isinstance(last_event, tuple) and len(last_event) == 2:
                        node_name, state_dict = last_event
                        if isinstance(state_dict, dict):
                            result = state_dict
                            break  # Success!
                        else:
                            # If state_dict is not a dict, try to convert
                            result = dict(state_dict) if hasattr(state_dict, '__dict__') else {"messages": [], "latest_results": []}
                    elif isinstance(last_event, dict):
                        # Direct dict format
                        result = last_event
                        break  # Success!
                    else:
                        # Try to extract as dict
                        result = dict(last_event) if hasattr(last_event, '__dict__') else {"messages": [], "latest_results": []}
                        print(f"  ⚠️  Warning: Unexpected stream event format, extracted partial result", flush=True)
                else:
                    raise Exception("Stream returned no results")
                    
            except asyncio.TimeoutError:
                # Timeout - the operation took longer than 600 seconds
                raise Exception(f"Stream timed out after 600s. The eval agent may still be processing. Check thread_id: {thread_id}")
            except Exception as stream_error:
                error_str = str(stream_error)
                # If it's a broken pipe or connection error, retry if we have attempts left
                if ("Broken pipe" in error_str or "Connection" in error_str or "Errno 32" in error_str) and stream_attempt < max_stream_retries - 1:
                    print(f"  ⚠️  Connection lost (attempt {stream_attempt + 1}/{max_stream_retries}), retrying in 5s...", flush=True)
                    await asyncio.sleep(5)
                    # Reconnect
                    try:
                        sync_client = get_sync_client(url=EVAL_AGENT_URL)
                        # Recreate trace context for reconnection
                        if LANGFUSE_CLIENT:
                            trace_id = Langfuse.create_trace_id(seed=thread_id)
                            langfuse_handler = CallbackHandler()
                        remote_graph = RemoteGraph(
                            "eval_agent",
                            sync_client=sync_client,
                        )
                    except Exception as reconnect_error:
                        raise Exception(f"Failed to reconnect after connection loss: {reconnect_error}")
                    continue
                else:
                    # No more retries or different error
                    raise Exception(
                        f"Connection lost during streaming after {stream_attempt + 1} attempts. "
                        f"Error: {error_str}. "
                        f"Thread ID: {thread_id}. "
                        f"Check eval agent logs for crashes/restarts."
                    )
        
        if result is None:
            raise Exception("Failed to get result after all retry attempts")
        
        execution_time = time.time() - start_time
        
        # Extract metrics
        metrics = extract_metrics_from_state(result)
        metrics["execution_time_seconds"] = execution_time
        metrics["thread_id"] = thread_id
        
        print(f"\n📊 Results:", flush=True)
        print(f"  Execution Time: {execution_time:.2f}s", flush=True)
        print(f"  Attempts: {metrics['attempts']}", flush=True)
        print(f"  Test Cases Generated: {metrics['num_test_cases']}", flush=True)
        print(f"  Tests Passed: {metrics['passed_tests']}", flush=True)
        print(f"  Tests Failed: {metrics['failed_tests']}", flush=True)
        
        return {
            "run_num": run_num,
            "condition": architecture,
            "success": True,
            "metrics": metrics,
            "result_state": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        error_msg = f"Eval agent invocation timed out after {execution_time:.2f}s"
        print(f"  ❌ {error_msg}", flush=True)
        
        return {
            "run_num": run_num,
            "condition": architecture,
            "success": False,
            "error": error_msg,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Eval agent invocation failed: {str(e)}"
        print(f"  ❌ {error_msg}", flush=True)
        import traceback
        traceback.print_exc()
        
        return {
            "run_num": run_num,
            "condition": architecture,
            "success": False,
            "error": error_msg,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        }

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from results."""
    if not results:
        return {}
    
    successful_runs = [r for r in results if r.get("success", False)]
    if not successful_runs:
        return {
            "total_runs": len(results),
            "successful_runs": 0,
            "success_rate": 0.0
        }
    
    metrics_list = [r["metrics"] for r in successful_runs]
    
    stats = {
        "total_runs": len(results),
        "successful_runs": len(successful_runs),
        "success_rate": len(successful_runs) / len(results),
        "avg_execution_time": sum(m.get("execution_time_seconds", 0) for m in metrics_list) / len(metrics_list),
        "avg_attempts": sum(m.get("attempts", 0) for m in metrics_list) / len(metrics_list),
        "avg_test_cases": sum(m.get("num_test_cases", 0) for m in metrics_list) / len(metrics_list),
        "avg_passed_tests": sum(m.get("passed_tests", 0) for m in metrics_list) / len(metrics_list),
        "avg_failed_tests": sum(m.get("failed_tests", 0) for m in metrics_list) / len(metrics_list),
    }
    
    if metrics_list:
        total_tests = sum(m.get("num_test_cases", 0) for m in metrics_list)
        total_passed = sum(m.get("passed_tests", 0) for m in metrics_list)
        if total_tests > 0:
            stats["overall_pass_rate"] = total_passed / total_tests
        else:
            stats["overall_pass_rate"] = 0.0
    
    return stats

# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the experiment."""
    print("="*60)
    print("EXPERIMENT E8: REACT VS REFLEXION FOR EVAL AGENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Runs per condition: {NUM_RUNS_PER_CONDITION}")
    print(f"  Test scenarios: {len(TEST_SCENARIOS)} (simple + complex)")
    print(f"  Eval Agent URL: {EVAL_AGENT_URL}")
    print(f"  Results Directory: {RESULTS_DIR}")
    print(f"\n⚠️  IMPORTANT: The eval agent must be started with EVAL_AGENT_ARCHITECTURE")
    print(f"   environment variable set. Restart the eval agent between conditions:")
    print(f"   1. For React: EVAL_AGENT_ARCHITECTURE=react python run.py")
    print(f"   2. For Reflexion: EVAL_AGENT_ARCHITECTURE=reflexion python run.py")
    print("="*60)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    react_results = checkpoint.get("react_results", [])
    reflexion_results = checkpoint.get("reflexion_results", [])
    
    if react_results or reflexion_results:
        print(f"\n📂 Loaded checkpoint:")
        print(f"  React runs: {len(react_results)}")
        print(f"  Reflexion runs: {len(reflexion_results)}")
    
    # Check if eval agent is running (non-blocking check)
    # We'll let the actual invocation handle connection errors more gracefully
    print(f"📡 Eval agent URL: {EVAL_AGENT_URL}")
    print(f"   (Connection will be tested during first invocation)")
    
    # Run React Condition
    print("\n" + "="*60)
    print("RUNNING REACT CONDITION")
    print("="*60)
    react_start_num = len(react_results) + 1
    for run_num in range(react_start_num, NUM_RUNS_PER_CONDITION + 1):
        # Use different scenario for each run (cycle through scenarios)
        scenario_idx = (run_num - 1) % len(TEST_SCENARIOS)
        scenario = TEST_SCENARIOS[scenario_idx]
        scenario_type = "simple" if scenario_idx == 0 else "complex"
        print(f"\n📋 Using {scenario_type} scenario for run {run_num}", flush=True)
        
        result = await invoke_eval_agent("react", run_num, scenario)
        result["scenario_type"] = scenario_type
        react_results.append(result)
        save_run_to_json(result, run_num, "react")
        save_checkpoint(react_results, reflexion_results)
        # Small delay between runs
        await asyncio.sleep(5)
    
    print(f"\n✅ React condition complete: {len(react_results)} runs")
    
    # Run Reflexion Condition
    print("\n" + "="*60)
    print("RUNNING REFLEXION CONDITION")
    print("="*60)
    reflexion_start_num = len(reflexion_results) + 1
    for run_num in range(reflexion_start_num, NUM_RUNS_PER_CONDITION + 1):
        # Use different scenario for each run (cycle through scenarios)
        scenario_idx = (run_num - 1) % len(TEST_SCENARIOS)
        scenario = TEST_SCENARIOS[scenario_idx]
        scenario_type = "simple" if scenario_idx == 0 else "complex"
        print(f"\n📋 Using {scenario_type} scenario for run {run_num}", flush=True)
        
        result = await invoke_eval_agent("reflexion", run_num, scenario)
        result["scenario_type"] = scenario_type
        reflexion_results.append(result)
        save_run_to_json(result, run_num, "reflexion")
        save_checkpoint(react_results, reflexion_results)
        # Small delay between runs
        await asyncio.sleep(5)
    
    print(f"\n✅ Reflexion condition complete: {len(reflexion_results)} runs")
    
    # Calculate Statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    react_stats = calculate_statistics(react_results)
    reflexion_stats = calculate_statistics(reflexion_results)
    
    print(f"\n📊 REACT (n={react_stats.get('total_runs', 0)}):")
    print(f"  Success Rate: {react_stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Avg Execution Time: {react_stats.get('avg_execution_time', 0):.2f}s")
    print(f"  Avg Attempts: {react_stats.get('avg_attempts', 0):.1f}")
    print(f"  Avg Test Cases: {react_stats.get('avg_test_cases', 0):.1f}")
    print(f"  Avg Passed Tests: {react_stats.get('avg_passed_tests', 0):.1f}")
    print(f"  Avg Failed Tests: {react_stats.get('avg_failed_tests', 0):.1f}")
    print(f"  Overall Pass Rate: {react_stats.get('overall_pass_rate', 0)*100:.1f}%")
    
    print(f"\n📊 REFLEXION (n={reflexion_stats.get('total_runs', 0)}):")
    print(f"  Success Rate: {reflexion_stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Avg Execution Time: {reflexion_stats.get('avg_execution_time', 0):.2f}s")
    print(f"  Avg Attempts: {reflexion_stats.get('avg_attempts', 0):.1f}")
    print(f"  Avg Test Cases: {reflexion_stats.get('avg_test_cases', 0):.1f}")
    print(f"  Avg Passed Tests: {reflexion_stats.get('avg_passed_tests', 0):.1f}")
    print(f"  Avg Failed Tests: {reflexion_stats.get('avg_failed_tests', 0):.1f}")
    print(f"  Overall Pass Rate: {reflexion_stats.get('overall_pass_rate', 0)*100:.1f}%")
    
    # Comparison
    print(f"\n📈 COMPARISON:")
    if react_stats.get('success_rate', 0) > 0 and reflexion_stats.get('success_rate', 0) > 0:
        time_diff = reflexion_stats.get('avg_execution_time', 0) - react_stats.get('avg_execution_time', 0)
        pass_rate_diff = reflexion_stats.get('overall_pass_rate', 0) - react_stats.get('overall_pass_rate', 0)
        
        print(f"  Execution Time Difference: {time_diff:+.2f}s (Reflexion - React)")
        print(f"  Pass Rate Difference: {pass_rate_diff*100:+.1f}% (Reflexion - React)")
        
        if pass_rate_diff > 0.05:
            print(f"  ✅ Reflexion shows better pass rate")
        elif pass_rate_diff < -0.05:
            print(f"  ✅ React shows better pass rate")
        else:
            print(f"  ⚖️  Similar pass rates")
    
    # Save summary
    summary = {
        "experiment": "e8--react-vs-reflexion-eval-agent",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_runs_per_condition": NUM_RUNS_PER_CONDITION,
            "eval_agent_url": EVAL_AGENT_URL
        },
        "react_stats": react_stats,
        "reflexion_stats": reflexion_stats
    }
    
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save final checkpoint
    save_checkpoint(react_results, reflexion_results)
    
    print(f"\n✅ Experiment complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Checkpoint saved to: {RESULTS_DIR / 'checkpoint.json'}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())

