"""
Seer - Simplified Agent Evaluation UI
"""

import streamlit as st
import asyncio
import httpx
import json
import os
import uuid
import traceback
from pathlib import Path
import sys

# Add project root to path for imports BEFORE importing shared modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from seer.shared.config import get_seer_config
from seer.shared.a2a_utils import send_a2a_message

# Data service URL (separate from orchestrator)
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://127.0.0.1:8500")

# Configure page
st.set_page_config(
    page_title="Seer - Agent Evaluation",
    page_icon="🔮",
    layout="wide"
)

# Constants
# Configuration - will be loaded from centralized config
from seer.shared.config import get_assistant_id as _get_assistant_id, get_graph_name as _get_graph_name
ORCHESTRATOR_ASSISTANT_ID = _get_assistant_id("orchestrator")
ORCHESTRATOR_GRAPH_ID = _get_graph_name("orchestrator")


def init_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None  # Will be set after first message
    if "langgraph_thread_id" not in st.session_state:
        st.session_state.langgraph_thread_id = None
    if "threads_cache" not in st.session_state:
        st.session_state.threads_cache = []


async def send_message_to_orchestrator(content: str, thread_id: str = None) -> tuple[str, str]:
    """
    Send message to Orchestrator agent
    Returns: (response_text, langgraph_thread_id)
    """
    try:
        # Get orchestrator config
        config = get_seer_config()
        orchestrator_port = config.orchestrator_port
        
        # Use /runs/stream to get thread_id from LangGraph
        url = f"http://127.0.0.1:{orchestrator_port}/runs/stream"
        payload = {
            "assistant_id": ORCHESTRATOR_GRAPH_ID or "orchestrator",
            "input": {"messages": [{"role": "user", "content": content}]},
            "stream_mode": ["values"],
        }
        
        # Only include thread_id if we have one from a previous message
        if thread_id:
            payload["config"] = {"configurable": {"thread_id": thread_id}}
        
        async with httpx.AsyncClient(timeout=config.a2a_timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                return f"Error: HTTP {resp.status_code} {resp.text[:200]}", thread_id
            
            final_response = ""
            extracted_thread_id = thread_id  # Default to input thread_id
            
            for line in resp.text.strip().split('\n'):
                if line.startswith('data: '):
                    try:
                        obj = json.loads(line[6:])
                    except Exception:
                        continue
                    
                    # Extract thread_id from metadata
                    if isinstance(obj, dict):
                        metadata = obj.get("metadata", {})
                        if isinstance(metadata, dict):
                            tid = metadata.get("thread_id")
                            if tid:
                                extracted_thread_id = tid
                    
                    # Extract messages
                    msgs = None
                    if isinstance(obj, dict):
                        vals = obj.get("values", {}) if isinstance(obj.get("values"), dict) else {}
                        if not isinstance(vals, dict):
                            vals = obj.get("value", {}) if isinstance(obj.get("value"), dict) else {}
                        msgs = vals.get("messages") if isinstance(vals, dict) else None
                        if msgs is None:
                            msgs = obj.get("messages") if isinstance(obj.get("messages"), list) else None
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if isinstance(msg, dict) and (msg.get("type") == "ai" or msg.get("role") == "assistant"):
                                text = msg.get("content") or msg.get("text") or ""
                                if text:
                                    final_response = text
            
            return final_response or "No response received", extracted_thread_id
            
    except Exception as e:
        st.error(f"Failed to send message: {e}")
        if st.checkbox("Show debug info", key="debug_send"):
            st.code(traceback.format_exc())
        return f"Error: {str(e)}", thread_id


async def get_conversation_messages(thread_id: str) -> list:
    """Get conversation messages from Data Service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/threads/{thread_id}/messages")
            response.raise_for_status()
            data = response.json()
            return data.get("messages", [])
    except Exception as e:
        st.error(f"Failed to fetch messages: {e}")
        if st.checkbox("Show debug info", key="debug_messages"):
            st.code(traceback.format_exc())
        return []


async def list_threads() -> list:
    """List all conversation threads via Data Service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/threads")
            response.raise_for_status()
            data = response.json()
            return data.get("threads", [])
    except Exception as e:
        st.error(f"Failed to list threads: {e}")
        return []


async def get_eval_suites():
    """Get all eval suites from Data Service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/eval-suites")
            response.raise_for_status()
            data = response.json()
            return data.get("suites", [])
    except Exception as e:
        st.error(f"Failed to fetch eval suites: {e}")
        if st.checkbox("Show debug info", key="debug_eval_suites"):
            st.code(traceback.format_exc())
        return []


async def get_test_results(suite_id: str):
    """Get test results for a specific eval suite from Data Service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/test-results", params={"suite_id": suite_id})
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
    except Exception as e:
        st.error(f"Failed to fetch test results: {e}")
        if st.checkbox("Show debug info", key="debug_test_results"):
            st.code(traceback.format_exc())
        return []

def render_chat():
    """Render main chat interface"""
    st.title("💬 Chat with Seer")

    # Thread controls in chat tab
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("➕ New Thread", key="new_thread_btn"):
                st.session_state.thread_id = None  # Will be set by LangGraph on first message
                st.session_state.langgraph_thread_id = None
                st.session_state.messages = []
                st.rerun()

        with col2:
            threads = asyncio.run(list_threads())
            st.session_state.threads_cache = threads
            options = [t.get("thread_id") for t in threads if t.get("thread_id")]
            current = st.session_state.thread_id
            
            # Display thread selector
            if not current:
                # New thread, not yet created
                st.text("🆕 New Thread")
            elif options:
                # Existing threads available
                # Add current thread to options if it's not there
                if current not in options:
                    options = [current] + options
                
                default_index = options.index(current) if current in options else 0
                
                # Create display names
                display_options = [f"{opt[:8]}..." for opt in options]
                selected_display = st.selectbox("Thread", display_options, index=default_index, key="thread_select")
                
                # Map back to actual thread_id
                selected_idx = display_options.index(selected_display)
                selected = options[selected_idx]
                
                if selected and selected != current:
                    st.session_state.thread_id = selected
                    st.session_state.langgraph_thread_id = selected
                    msgs = asyncio.run(get_conversation_messages(selected))
                    st.session_state.messages = [
                        {"role": (m.get("role") or "assistant"), "content": (m.get("content") or "")}
                        for m in msgs
                    ]
                    st.rerun()
            else:
                # No threads in DB yet, show current
                st.text(f"Thread: {current[:8]}..." if current else "🆕 New Thread")

        with col3:
            if st.button("↻ Refresh", key="refresh_threads"):
                st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick start guide
    if len(st.session_state.messages) == 0:
        st.info("""
        ### 🚀 Quick Start

        Tell me about the agent you want to evaluate. For example:

        **"Evaluate my agent at localhost:2024 (ID: my_agent). It should remember user preferences and respond politely."**
        """)

    # Chat input
    if prompt := st.chat_input("Message Seer..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Send to Orchestrator - use langgraph_thread_id if available, otherwise None (LangGraph will create one)
        reply, langgraph_thread_id = asyncio.run(
            send_message_to_orchestrator(prompt, st.session_state.langgraph_thread_id)
        )
        
        # Update thread_id with LangGraph's thread_id (first message creates it)
        if langgraph_thread_id and not st.session_state.langgraph_thread_id:
            st.session_state.langgraph_thread_id = langgraph_thread_id
            st.session_state.thread_id = langgraph_thread_id
        
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Rerun to show new messages
        st.rerun()


def render_results():
    """Render combined eval suites and test results"""
    st.title("📊 Evaluation Results")

    # Get eval suites
    eval_suites = asyncio.run(get_eval_suites())

    if not eval_suites:
        st.info("No evaluation suites yet. Start an evaluation in the Chat tab!")
        return

    # Display eval suites table
    st.subheader("Evaluation Suites")

    suite_data = []
    for suite in eval_suites:
        test_cases = suite.get("test_cases", [])
        suite_data.append({
            "Suite ID": suite.get("suite_id", "N/A"),
            "Spec Name": suite.get("spec_name", "N/A"),
            "Target Agent": f"{suite.get('target_agent_url', 'N/A')}/{suite.get('target_agent_id', 'N/A')}",
            "Test Count": len(test_cases) if isinstance(test_cases, list) else 0,
            "Created": suite.get("created_at", "N/A")[:19] if suite.get("created_at") else "N/A"
        })

    st.dataframe(suite_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Select suite to view results
    st.subheader("Test Results")
    suite_options = {suite.get("suite_id"): suite for suite in eval_suites if suite.get("suite_id")}

    if not suite_options:
        st.warning("No valid eval suites found")
        return

    selected_suite_id = st.selectbox("Select Eval Suite", list(suite_options.keys()))

    if selected_suite_id:
        # Get test results for selected suite
        results = asyncio.run(get_test_results(selected_suite_id))

        if not results:
            st.info("No test results yet. Run the evaluation to see results.")
            return

        # Calculate summary
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        failed = total - passed
        score = (passed / total * 100) if total > 0 else 0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tests", total)
        with col2:
            st.metric("Passed", passed, delta=f"+{passed}" if passed > 0 else "0")
        with col3:
            st.metric("Failed", failed, delta=f"-{failed}" if failed > 0 else "0")
        with col4:
            st.metric("Score", f"{score:.0f}%")

        st.markdown("---")

        # Display results table
        results_data = []
        for r in results:
            results_data.append({
                "Status": "✅" if r.get("passed") else "❌",
                "Test ID": r.get("test_case_id", "N/A")[:30],
                "Input": r.get("input_sent", "N/A")[:50],
                "Score": f"{r.get('score', 0):.2f}"
            })

        st.dataframe(results_data, use_container_width=True, hide_index=True)


# Removed complex debugging functions - now simplified to just the essentials




def main():
    """Main app entry point"""
    init_session_state()

    # Simple 2-tab interface
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Results"])

    with tab1:
        render_chat()

    with tab2:
        render_results()

    # Simple footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>Seer - Orchestrator-Based Multi-Agent System</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

