import os
import json
from typing import List, Dict, Any, Optional
import time
import httpx

MAX_IO_CHARS = 1000

def _short(obj: Any) -> str:
    """Truncate long strings/objects for display."""
    try:
        if isinstance(obj, str):
            s = obj
        else:
            s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    return (s[:MAX_IO_CHARS] + "...") if s and len(s) > MAX_IO_CHARS else s

def get_smart_trace_summary(session_id: str, trace_id: Optional[str] = None) -> str:
    """
    Fetches traces from LangFuse and applies 'Smart Compression'.
    
    Logic:
    1. Fetch observations for the trace/session.
    2. Group successful steps into a summary line.
    3. Expand failed steps with details.
    4. (Mock) Link to code.
    """
    # Load env vars (in case called directly)
    from pathlib import Path
    from dotenv import load_dotenv
    project_root = Path(__file__).parents[2]
    env_path = project_root / ".env"
    load_dotenv(env_path, override=True)
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # Support both LANGFUSE_HOST and LANGFUSE_BASE_URL
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "http://localhost:3000"
    
    if not public_key or not secret_key:
        return f"❌ LangFuse credentials not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
    
    # Parse host to get base_url
    base_url = host.rstrip('/')
    if not base_url.startswith('http'):
        base_url = f"http://{base_url}"
    
    api_base = f"{base_url}/api/public"
    
    print(f"🔍 Fetching traces for session: {session_id}...")
    
    # Give API a moment to index if we just ran it
    time.sleep(2)
    
    try:
        # Use HTTP client with basic auth
        auth = (public_key, secret_key)
        headers = {"Content-Type": "application/json"}
        
        if trace_id:
            # Fetch specific trace
            url = f"{api_base}/traces/{trace_id}?fields=core,observations"
        else:
            # Fetch by session_id - use trace list endpoint
            url = f"{api_base}/traces?sessionId={session_id}&limit=1&fields=core,observations"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, auth=auth, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if trace_id:
                trace_data = data
            else:
                # List endpoint returns {"data": [...]}
                if not data.get("data"):
                    return f"❌ No traces found for session {session_id}"
                trace_data = data["data"][0]
                trace_id = trace_data.get("id")
            
            # Get observations
            observations = trace_data.get("observations", [])
            
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP Error fetching traces: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        import traceback
        return f"❌ Error fetching traces from LangFuse: {e}\n{traceback.format_exc()}"

    if not observations:
        return f"⚠️  Trace found but no observations available. Trace ID: {trace_id}"
    
    # Sort by start time (observations are dicts from API)
    def get_start_time(obs):
        return obs.get("startTime") or obs.get("start_time") or 0
    
    steps = sorted(observations, key=get_start_time)
    
    compressed_log = []
    buffer = [] # Holds successful steps to be summarized
    
    def flush_buffer():
        if not buffer: return
        
        # If we have a chunk of successful steps
        if len(buffer) > 2:
            start_idx = buffer[0]['index']
            end_idx = buffer[-1]['index']
            names = ", ".join(set(s['name'] for s in buffer[:3]))
            if len(buffer) > 3: names += "..."
            
            compressed_log.append(f"✅ Steps {start_idx}-{end_idx}: {len(buffer)} successful operations ({names})")
        else:
            # If it's just 1 or 2, show them
            for s in buffer:
                compressed_log.append(f"✅ Step {s['index']}: {s['name']}")
        
        buffer.clear()

    print(f"📊 Analyzing {len(steps)} steps for Smart Compression...")
    start_analysis = time.time()

    for i, step in enumerate(steps):
        # Observations are dicts from API
        step_name = step.get("name") or step.get("type") or f"Step_{i+1}"
        step_level = step.get("level") or step.get("statusCode") or "DEFAULT"
        status_msg = step.get("statusMessage") or step.get("status_message") or ""
        step_input = step.get("input") or {}
        step_output = step.get("output") or {}
        
        # Basic mapping for internal buffer
        step_data = {'index': i+1, 'name': step_name}
        
        # Check for error
        is_error = step_level == "ERROR" or (status_msg and "fail" in status_msg.lower())
        
        if is_error:
            flush_buffer()
            
            error_msg = status_msg or "Unknown Error"
            input_snippet = _short(step_input)
            output_snippet = _short(step_output)
            
            compressed_log.append(f"\n🛑 CRITICAL FAILURE at Step {i+1} [{step_name}]")
            compressed_log.append(f"   Error: {error_msg}")
            compressed_log.append(f"   Input: {input_snippet}")
            compressed_log.append(f"   Output: {output_snippet}")
            
            # Gap C: Link to Code (Mock example)
            # In a real scenario, we'd look up step_name in registry
            compressed_log.append(f"   🔍 Debug: Check definition of '{step_name}' in agents/eval_agent/nodes/")
            
        else:
            buffer.append(step_data)
            
    flush_buffer()
    
    end_analysis = time.time()
    analysis_time_ms = (end_analysis - start_analysis) * 1000
    
    footer = f"\n(Analysis took {analysis_time_ms:.2f}ms)"
    return "\n".join(compressed_log) + footer

if __name__ == "__main__":
    # Test with a dummy session if run directly
    print("Run run_experiment.py to generate real traces first.")

