#!/usr/bin/env python3
"""
Experiment E9c: End-to-End Simplified

Full workflow validation with minimal complexity.
1 test case, 1 round, no Codex handoff.
"""

import os
import sys
import json
import asyncio
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from shared.config import config

# Configuration
EVAL_AGENT_PORT = 8004
EVAL_AGENT_URL = f"http://127.0.0.1:{EVAL_AGENT_PORT}"
RESULTS_FILE = Path(__file__).parent / "results.json"

LEVEL_NAMES = {0: "minimal", 1: "system_goal", 2: "system_goal_action", 3: "full_context"}

# TWO TEST SCENARIO VARIANTS
# Select which variant to use via --variant argument (default: 1)

TEST_SCENARIOS = {
    1: """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should be able to list GitHub issues.
When I ask it to "list all open issues", it should return a list of open issues.""",

    2: """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should be able to create a GitHub issue.
When I ask it to "create an issue", it should create a new issue with a title."""
}

# Default to variant 1 (simplest - read-only)
TEST_SCENARIO = TEST_SCENARIOS[1]

# ORIGINAL COMPLEX SCENARIO (commented out - all tests failing)
# TEST_SCENARIO = """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder
#
# The agent should sync Asana ticket updates when a GitHub PR is merged. 
# Whenever I merge a PR, it should search for related Asana tickets and update/close them."""

# Initialize Langfuse client for distributed tracing
LANGFUSE_CLIENT = Langfuse(
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_base_url
) if config.langfuse_secret_key else None

def kill_existing_agent():
    try:
        subprocess.run(["pkill", "-f", f"langgraph.*{EVAL_AGENT_PORT}"], check=False)
        time.sleep(2)
    except:
        pass

def start_agent(context_level: int):
    """Start eval agent with specified context level."""
    print(f"🚀 Starting Eval Agent with TARGET_AGENT_CONTEXT_LEVEL={context_level}...")
    
    env = os.environ.copy()
    env["TARGET_AGENT_CONTEXT_LEVEL"] = str(context_level)
    env["CODEX_HANDOFF_ENABLED"] = "false"
    env["EVAL_N_TEST_CASES"] = "2"  # Increased to 2 test cases for better signal
    env["EVAL_N_ROUNDS"] = "1"  # Just 1 round
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    
    import shutil
    langgraph_exe = shutil.which("langgraph")
    if not langgraph_exe:
        langgraph_exe = str(Path(sys.executable).parent / "langgraph")
        if not Path(langgraph_exe).exists():
            raise RuntimeError("Could not find langgraph executable")
    
    project_root = Path(__file__).parent.parent.parent
    cmd = [
        langgraph_exe, "dev",
        "--port", str(EVAL_AGENT_PORT),
        "--host", "127.0.0.1",
        "--no-browser",
        "--config", "agents/eval_agent/langgraph.json"
    ]
    
    log_file = open(f"/tmp/eval_agent_{EVAL_AGENT_PORT}_level_{context_level}.log", "w")
    process = subprocess.Popen(
        cmd, cwd=str(project_root), env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    # Wait for ready - increased timeout and better error handling
    print("⏳ Waiting for agent...")
    for i in range(60):  # Increased from 30 to 60 seconds
        try:
            sync_client = get_sync_client(url=EVAL_AGENT_URL)
            sync_client.threads.create()
            print("✅ Agent ready!")
            return process, log_file
        except Exception as e:
            if process.poll() is not None:
                # Process died, try to read error from log file
                error_msg = f"Agent failed to start. Exit code: {process.returncode}"
                try:
                    # Log file is write-only, read from disk instead
                    log_path = log_file.name
                    log_file.close()
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                        if "error" in log_content.lower() or "Error" in log_content:
                            error_msg += f"\nLast log entries:\n{log_content[-500:]}"
                except:
                    pass  # If we can't read log, just report the exit code
                raise RuntimeError(error_msg)
            if i % 5 == 0:  # Print progress every 5 seconds
                print(f"  Still waiting... ({i+1}/60)")
            time.sleep(1)
    
    raise RuntimeError(f"Agent timeout after 60 seconds. Check logs at {log_file.name}")

async def run_experiment(context_level: int):
    """Run full workflow for one context level."""
    print(f"\n{'='*60}")
    print(f"E9c: End-to-End - Level {context_level}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        sync_client = get_sync_client(url=EVAL_AGENT_URL)
        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_id = thread["thread_id"]
        thread_cfg = {"configurable": {"thread_id": thread_id}}
        
        input_payload = {"messages": [{"role": "user", "content": TEST_SCENARIO}]}
        
        # Setup Langfuse trace context for distributed tracing
        trace_id = None
        langfuse_handler = None
        if LANGFUSE_CLIENT:
            trace_id = Langfuse.create_trace_id(seed=thread_id)
            langfuse_handler = CallbackHandler()
            thread_cfg["metadata"] = thread_cfg.get("metadata", {})
            thread_cfg["metadata"]["langfuse_trace_id"] = trace_id
        
        remote_graph = RemoteGraph("eval_agent", sync_client=sync_client)
        
        print("🚀 Running full workflow...")
        # Wrap invocation with Langfuse trace context if available
        if LANGFUSE_CLIENT and trace_id:
            def invoke_with_tracing():
                with LANGFUSE_CLIENT.start_as_current_observation(
                    as_type="span",
                    name="eval-remote-invocation",
                    trace_context={"trace_id": trace_id}
                ) as span:
                    span.update_trace(input=input_payload)
                    invoke_cfg = {**thread_cfg, "callbacks": [langfuse_handler]} if langfuse_handler else thread_cfg
                    result = remote_graph.invoke(input_payload, invoke_cfg)
                    span.update_trace(output=result)
                    return result
            result = await asyncio.wait_for(
                asyncio.to_thread(invoke_with_tracing),
                timeout=600  # 10 min timeout
            )
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(remote_graph.invoke, input_payload, thread_cfg),
                timeout=600  # 10 min timeout
            )
        
        execution_time = time.time() - start_time
        
        # Extract metrics
        dataset_examples = result.get("dataset_examples", [])
        latest_results = result.get("latest_results", [])
        
        passed = sum(1 for r in latest_results if r.get("passed", False))
        failed = len(latest_results) - passed
        
        metrics = {
            "context_level": context_level,
            "execution_time": execution_time,
            "num_test_cases": len(dataset_examples),
            "num_results": len(latest_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(latest_results) if latest_results else 0.0,
        }
        
        print(f"\n📊 Results:")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Test Cases: {metrics['num_test_cases']}")
        print(f"  Passed: {passed} / {len(latest_results)}")
        
        return {"success": True, "metrics": metrics, "result": result}
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  ❌ Failed: {str(e)}")
        return {"success": False, "error": str(e), "execution_time": execution_time}

async def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=[1,2,3])
    parser.add_argument("--variant", type=int, default=1, choices=[1,2],
                        help="Test scenario variant: 1=read-only (list issues), 2=simple write (create issue)")
    args = parser.parse_args()
    
    level = args.level
    variant = args.variant
    global TEST_SCENARIO
    TEST_SCENARIO = TEST_SCENARIOS[variant]
    
    print(f"🧪 E9c: End-to-End - Level {level}, Variant {variant}")
    print(f"📋 Test Scenario: {TEST_SCENARIO[:100]}...")
    
    # Cleanup
    kill_existing_agent()
    
    # Start agent
    process, log_file = start_agent(level)
    
    try:
        # Run experiment
        result = await run_experiment(level)
        
        # Save results to consolidated JSON
        result["timestamp"] = datetime.now().isoformat()
        result["context_level"] = level
        result["test_variant"] = variant
        result["test_scenario"] = TEST_SCENARIO
        
        # Load existing consolidated results or create new structure
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                consolidated = json.load(f)
        else:
            consolidated = {
                "experiment": "E9c: Context Level Impact on End-to-End Evaluation",
                "variants": {
                    "1": "Read-only operations (List GitHub Issues)",
                    "2": "Simple write operations (Create GitHub Issue)"
                },
                "context_levels": {
                    "1": "System Goal (input_message + system goal description)",
                    "2": "System Goal + Action (Level 1 + expected action)",
                    "3": "Full Context (Level 2 + MCP services + resource hints)"
                },
                "results": {}
            }
        
        # Update consolidated results
        if str(level) not in consolidated["results"]:
            consolidated["results"][str(level)] = {}
        consolidated["results"][str(level)][str(variant)] = result
        
        # Save consolidated results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(consolidated, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {RESULTS_FILE} (Level {level}, Variant {variant})")
        
    finally:
        print("🧹 Stopping agent...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        log_file.close()
        print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(main())

