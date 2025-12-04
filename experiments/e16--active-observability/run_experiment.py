import os
import sys
import asyncio
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env from project root
project_root = Path(__file__).parents[2] # seer/
sys.path.append(str(project_root))

env_path = project_root / ".env"
print(f"📂 Loading env from: {env_path} (Exists: {env_path.exists()})")
load_dotenv(env_path)

# Check Keys
required_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
missing_keys = [k for k in required_keys if not os.getenv(k)]
if missing_keys:
    print(f"❌ Missing environment variables: {', '.join(missing_keys)}")
    print("⚠️  Proceeding with MOCK credentials if allowed, or failing...")
    # If user didn't provide them, we can't really talk to LangFuse. 
    # But for the sake of the experiment structure, we might fail later.

try:
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        from langfuse.callback import CallbackHandler
    from experiments.e16__active_observability.smart_trace_logic import get_smart_trace_summary
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- MOCK AGENT GRAPH ---
# We avoid importing the full Seer agent because it requires complex environment setup (MCP, Context7, etc)
# that might not be present in this feasibility test environment.
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class MockState(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

async def step_1_think(state):
    print("   🤖 Agent: Thinking...")
    return {"messages": ["Thinking about the task..."], "count": 1}

async def step_2_tool(state):
    print("   🛠️  Agent: Calling Tool (Mock GitHub)...")
    # Simulate tool call
    return {"messages": ["Tool: GitHub PR #123 fetched"], "count": 1}

async def step_3_analyze(state):
    print("   🧠 Agent: Analyzing failure...")
    return {"messages": ["Analysis: PR is missing label."], "count": 1}

async def step_fail(state):
    print("   ❌ Agent: Encountering error...")
    # Simulate an error trace
    raise ValueError("Simulated Failure: GitHub API 500 Error")

def build_mock_graph():
    workflow = StateGraph(MockState)
    workflow.add_node("think", step_1_think)
    workflow.add_node("tool", step_2_tool)
    workflow.add_node("analyze", step_3_analyze)
    
    workflow.set_entry_point("think")
    workflow.add_edge("think", "tool")
    workflow.add_edge("tool", "analyze")
    workflow.add_edge("analyze", END)
    return workflow.compile()

async def run_experiment():
    print("\n🧪 E16: Active Observability - Trace Summarization Experiment")
    print("=======================================================")
    
    thread_id = str(uuid.uuid4())
    print(f"🆔 Thread ID: {thread_id}")
    
    # Configure LangFuse Handler
    try:
        from langfuse.types import TraceContext
        
        # Initialize trace context for session_id
        trace_context = TraceContext(
            session_id=thread_id,
            user_id="experiment-user",
            tags=["experiment", "mock-agent"]
        )
        
        # CallbackHandler reads credentials from env vars automatically
        langfuse_handler = CallbackHandler(trace_context=trace_context)
        print(f"✅ LangFuse Handler initialized with session_id: {thread_id}")
    except Exception as e:
        print(f"❌ Failed to init LangFuse Handler: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run Mock Agent
    print("\n🏃 Running Mock Agent...")
    graph = build_mock_graph()
    
    try:
        await graph.ainvoke(
            {"messages": [], "count": 0}, 
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler]
            }
        )
        print("✅ Agent execution complete.")
        
        # Flush traces and get trace_id
        trace_id = None
        if hasattr(langfuse_handler, 'flush'):
            langfuse_handler.flush()
        if hasattr(langfuse_handler, 'get_trace_id'):
            trace_id = langfuse_handler.get_trace_id()
        elif hasattr(langfuse_handler, 'trace_id'):
            trace_id = langfuse_handler.trace_id
        
        print("✅ Traces flushed to LangFuse.")
        if trace_id:
            print(f"📌 Trace ID: {trace_id}")
        else:
            print("⚠️  Could not get trace_id from handler - will query by session_id")
        
    except Exception as e:
        print(f"❌ Agent execution failed (Expected?): {e}")
        # In a real test we might want to simulate failure to see the trace
    
    # Analyze - wait a bit for traces to be indexed, then fetch most recent trace
    print("\n📉 Applying Smart Compression...")
    print("--------------------------------")
    print("⏳ Waiting 3 seconds for traces to be indexed...")
    import asyncio
    await asyncio.sleep(3)
    
    try:
        # If we have trace_id, use it; otherwise fetch most recent trace
        if trace_id:
            print(f"🔍 Fetching trace by trace_id: {trace_id}")
            summary = get_smart_trace_summary(session_id='', trace_id=trace_id)
        else:
            # Fetch most recent trace (should be the one we just created)
            print(f"🔍 Fetching most recent trace...")
            import httpx
            from dotenv import load_dotenv
            from pathlib import Path
            project_root = Path(__file__).parents[2]
            load_dotenv(project_root / ".env", override=True)
            
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "http://host.docker.internal:3000"
            base_url = host.rstrip('/')
            api_base = f"{base_url}/api/public"
            
            url = f"{api_base}/traces?limit=1&orderBy=timestamp.desc&fields=core,observations"
            auth = (public_key, secret_key)
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, auth=auth)
                response.raise_for_status()
                data = response.json()
                traces = data.get("data", [])
                if traces:
                    latest_trace_id = traces[0].get("id")
                    print(f"📌 Found trace: {latest_trace_id}")
                    summary = get_smart_trace_summary(session_id='', trace_id=latest_trace_id)
                else:
                    summary = "❌ No traces found"
        
        print("\n=== SMART TRACE SUMMARY ===")
        print(summary)
        print("===========================")
    except Exception as e:
        print(f"❌ Failed to fetch/analyze traces: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_experiment())
