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
from langsmith import Client as LangSmithClient

# Configuration
EVAL_AGENT_PORT = 8004
EVAL_AGENT_URL = f"http://127.0.0.1:{EVAL_AGENT_PORT}"
RESULTS_DIR = Path(__file__).parent / "e9c_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

LEVEL_NAMES = {0: "minimal", 1: "system_goal", 2: "system_goal_action", 3: "full_context"}

TEST_SCENARIO = """Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should sync Asana ticket updates when a GitHub PR is merged. 
Whenever I merge a PR, it should search for related Asana tickets and update/close them."""

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = LangSmithClient(api_key=LANGSMITH_API_KEY) if LANGSMITH_API_KEY else None

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
    env["EVAL_N_TEST_CASES"] = "1"  # Just 1 test case
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
    
    # Wait for ready
    print("⏳ Waiting for agent...")
    for i in range(30):
        try:
            sync_client = get_sync_client(url=EVAL_AGENT_URL)
            sync_client.threads.create()
            print("✅ Agent ready!")
            return process, log_file
        except:
            if process.poll() is not None:
                raise RuntimeError("Agent failed to start")
            time.sleep(1)
    
    raise RuntimeError("Agent timeout")

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
        
        remote_graph = RemoteGraph("eval_agent", sync_client=sync_client, client=LANGSMITH_CLIENT)
        
        print("🚀 Running full workflow...")
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
    parser.add_argument("--level", type=int, required=True, choices=[0,1,2,3])
    args = parser.parse_args()
    
    level = args.level
    print(f"🧪 E9c: End-to-End - Level {level}")
    
    # Cleanup
    kill_existing_agent()
    
    # Start agent
    process, log_file = start_agent(level)
    
    try:
        # Run experiment
        result = await run_experiment(level)
        
        # Save results
        result["timestamp"] = datetime.now().isoformat()
        result["context_level"] = level
        
        filename = RESULTS_DIR / f"level_{level}_result.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
        
    finally:
        print("🧹 Stopping agent...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        log_file.close()
        print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(main())

