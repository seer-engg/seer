"""
Seer - Simplified Agent Evaluation UI
"""

import streamlit as st
import asyncio
import httpx
import os
import traceback
from pathlib import Path
import sys

# Add project root to path for imports BEFORE importing shared modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Data service URL (separate from orchestrator)
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://127.0.0.1:8500")

# Configure page
st.set_page_config(
    page_title="Seer - Agent Evaluation",
    page_icon="🔮",
    layout="wide"
)

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
    Send message to Orchestrator via FastAPI backend (proxy pattern)
    Returns: (response_text, thread_id)
    """
    try:
        # Send through FastAPI backend for proper persistence
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{DATA_SERVICE_URL}/messages/send",
                json={
                    "content": content,
                    "thread_id": thread_id,
                    "target_agent": "orchestrator"
                }
            )
            
            if response.status_code >= 400:
                error_detail = response.json().get("detail", response.text[:200])
                return f"Error: {error_detail}", thread_id
            
            data = response.json()
            return data.get("response", "No response"), data.get("thread_id", thread_id)
            
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
            
            # Build options list
            # If we have a new thread (current=None or not in DB), add it to the list
            if current and current not in options:
                # Current thread exists but not in DB yet (new thread with messages)
                options = [current] + options
            elif not current and options:
                # No current thread but we have history - add "New Thread" placeholder
                options = ["__new__"] + options
            elif not current and not options:
                # Completely new, no history
                options = ["__new__"]
            
            # Create display options
            display_options = []
            for opt in options:
                if opt == "__new__":
                    display_options.append("🆕 New Thread")
                elif opt not in [t.get("thread_id") for t in threads]:
                    # Thread has messages but not in DB list yet
                    display_options.append(f"🆕 Current ({opt[:8]}...)")
                else:
                    # Existing thread from DB
                    display_options.append(f"💬 {opt[:8]}...")
            
            # Determine default index
            if not current:
                default_index = 0  # "New Thread"
            elif current in options:
                default_index = options.index(current)
            else:
                default_index = 0
            
            # Show selector
            if display_options:
                selected_display = st.selectbox("Thread", display_options, index=default_index, key="thread_select")
                selected_idx = display_options.index(selected_display)
                selected = options[selected_idx]
                
                # Handle selection
                if selected == "__new__":
                    # User selected "New Thread" - clear everything
                    if current is not None or st.session_state.messages:
                        st.session_state.thread_id = None
                        st.session_state.langgraph_thread_id = None
                        st.session_state.messages = []
                        st.rerun()
                elif selected != current:
                    # User selected a different existing thread
                    st.session_state.thread_id = selected
                    st.session_state.langgraph_thread_id = selected
                    msgs = asyncio.run(get_conversation_messages(selected))
                    st.session_state.messages = [
                        {"role": (m.get("role") or "assistant"), "content": (m.get("content") or "")}
                        for m in msgs
                    ]
                    st.rerun()

        with col3:
            if st.button("↻ Refresh", key="refresh_threads"):
                st.rerun()

    # Display thread debug header
    with st.container():
        tid = st.session_state.thread_id or "(new)"
        st.caption(f"Thread: {tid}")

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

        # Send via FastAPI backend (handles thread creation and persistence)
        reply, thread_id = asyncio.run(
            send_message_to_orchestrator(prompt, st.session_state.thread_id)
        )
        
        # Update thread_id (FastAPI backend creates it if needed)
        if thread_id:
            st.session_state.thread_id = thread_id
            st.session_state.langgraph_thread_id = thread_id
        
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

