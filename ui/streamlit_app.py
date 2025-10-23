"""
Seer - Simplified Agent Evaluation UI
Now uses LangGraph SDK directly - no separate data service needed!
"""

import streamlit as st
import asyncio
import os
import traceback
from pathlib import Path
import sys

# Add project root to path for imports BEFORE importing shared modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langgraph_sdk import get_client

# Orchestrator URL
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8001")
ORCHESTRATOR_ASSISTANT_ID = "orchestrator"

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
    if "processing_message" not in st.session_state:
        st.session_state.processing_message = False  # Lock during message processing
    if "auto_refresh_interval" not in st.session_state:
        st.session_state.auto_refresh_interval = 5  # Refresh thread info every 5 seconds


async def send_message_to_orchestrator(content: str, thread_id: str = None) -> tuple[str, str]:
    """
    Send message to Orchestrator via LangGraph SDK
    Returns: (response_text, thread_id)
    """
    try:
        client = get_client(url=ORCHESTRATOR_URL)
        
        # Create or get thread
        if not thread_id:
            thread_response = await client.threads.create()
            thread_id = thread_response["thread_id"]
        
        # Send message and stream response
        final_response = ""
        async for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=ORCHESTRATOR_ASSISTANT_ID,
            input={"messages": [{"role": "user", "content": content}]},
            stream_mode="values"
        ):
            if chunk.event == "values":
                # Extract assistant message from last message in state
                messages = chunk.data.get("messages", [])
                if messages and hasattr(messages[-1], "type"):
                    # LangChain message object
                    if messages[-1].type == "ai":
                        final_response = messages[-1].content
                elif messages and isinstance(messages[-1], dict):
                    # Dict format
                    if messages[-1].get("type") == "ai":
                        final_response = messages[-1].get("content", "")
        
        return final_response or "No response", thread_id
            
    except Exception as e:
        st.error(f"Failed to send message: {e}")
        if st.checkbox("Show debug info", key="debug_send"):
            st.code(traceback.format_exc())
        return f"Error: {str(e)}", thread_id


async def get_thread_state(thread_id: str) -> dict:
    """Get thread state from LangGraph - contains all data!"""
    try:
        client = get_client(url=ORCHESTRATOR_URL)
        state = await client.threads.get_state(thread_id=thread_id)
        return state["values"] if state else {}
    except Exception as e:
        st.error(f"Failed to fetch thread state: {e}")
        return {}


async def get_conversation_messages(thread_id: str) -> list:
    """Get conversation messages from thread state"""
    try:
        state = await get_thread_state(thread_id)
        messages = state.get("messages", [])
        
        # Convert LangChain messages to dict format
        result = []
        for msg in messages:
            if hasattr(msg, "type"):
                # LangChain message object
                role = "user" if msg.type == "human" else "assistant"
                result.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                # Already dict format
                msg_type = msg.get("type", "ai")
                role = "user" if msg_type == "human" else "assistant"
                result.append({"role": role, "content": msg.get("content", "")})
        
        return result
    except Exception as e:
        st.error(f"Failed to fetch messages: {e}")
        return []


async def list_threads() -> list:
    """List all conversation threads via LangGraph"""
    try:
        client = get_client(url=ORCHESTRATOR_URL)
        threads = await client.threads.search()
        
        # Convert to format expected by UI
        return [
            {
                "thread_id": t["thread_id"],
                "created_at": t.get("created_at"),
                "updated_at": t.get("updated_at")
            }
            for t in threads
        ]
    except Exception as e:
        st.error(f"Failed to list threads: {e}")
        return []


async def get_eval_suites():
    """Get all eval suites from all thread states"""
    try:
        # Get all threads
        threads = await list_threads()
        all_suites = []
        
        # Collect eval suites from each thread
        for thread in threads:
            state = await get_thread_state(thread["thread_id"])
            suites = state.get("eval_suites", [])
            all_suites.extend(suites)
        
        return all_suites
    except Exception as e:
        st.error(f"Failed to fetch eval suites: {e}")
        return []


async def get_test_results(suite_id: str):
    """Get test results for a specific eval suite from all threads"""
    try:
        # Get all threads
        threads = await list_threads()
        all_results = []
        
        # Collect test results from each thread
        for thread in threads:
            state = await get_thread_state(thread["thread_id"])
            results = state.get("test_results", [])
            # Filter by suite_id
            suite_results = [r for r in results if r.get("suite_id") == suite_id]
            all_results.extend(suite_results)
        
        return all_results
    except Exception as e:
        st.error(f"Failed to fetch test results: {e}")
        return []


async def get_thread_config(thread_id: str):
    """Get target agent configuration from thread state"""
    try:
        state = await get_thread_state(thread_id)
        return state.get("target_config")
    except Exception as e:
        return None


async def get_thread_expectations(thread_id: str):
    """Get target agent expectations from thread state"""
    try:
        state = await get_thread_state(thread_id)
        return state.get("target_expectations")
    except Exception as e:
        return None


async def get_thread_eval_suites(thread_id: str):
    """Get evaluation suites from thread state"""
    try:
        state = await get_thread_state(thread_id)
        return state.get("eval_suites", [])
    except Exception as e:
        return []


async def get_thread_test_results(thread_id: str):
    """Get test results from thread state"""
    try:
        state = await get_thread_state(thread_id)
        return state.get("test_results", [])
    except Exception as e:
        return []


def render_thread_info_panel(thread_id: str):
    """Render right panel with thread-specific information"""
    if not thread_id:
        st.info("💡 Start a conversation to see thread information")
        return
    
    st.subheader("🔍 Thread Info")
    st.caption(f"Thread: {thread_id[:16]}...")
    
    # Auto-refresh controls
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("↻ Refresh Now", key="refresh_thread_info", use_container_width=True):
            st.rerun()
    with col_b:
        # Toggle auto-refresh
        auto_refresh = st.checkbox("Auto", value=False, key="auto_refresh_toggle")
    
    # Auto-refresh implementation using st.rerun with delay
    if auto_refresh:
        import time
        st.caption(f"🔄 Auto-refreshing every {st.session_state.auto_refresh_interval}s")
        time.sleep(st.session_state.auto_refresh_interval)
        st.rerun()
    
    st.markdown("---")
    
    # Target Agent Configuration
    with st.expander("📋 Target Agent Config", expanded=True):
        config = asyncio.run(get_thread_config(thread_id))
        if config:
            st.markdown(f"**URL:** `{config.get('target_agent_url', 'N/A')}`")
            st.markdown(f"**Port:** `{config.get('target_agent_port', 'N/A')}`")
            st.markdown(f"**Assistant ID:** `{config.get('target_agent_assistant_id', 'N/A')}`")
            if config.get('target_agent_github_url'):
                st.markdown(f"**GitHub:** {config.get('target_agent_github_url')}")
        else:
            st.caption("_No configuration yet_")
    
    # Target Agent Expectations
    with st.expander("🎯 Expectations", expanded=True):
        expectations = asyncio.run(get_thread_expectations(thread_id))
        if expectations and expectations.get('expectations'):
            exp_list = expectations.get('expectations', [])
            if isinstance(exp_list, list):
                for i, exp in enumerate(exp_list, 1):
                    st.markdown(f"{i}. {exp}")
            else:
                st.json(exp_list)
        else:
            st.caption("_No expectations collected yet_")
    
    # Eval Suites
    with st.expander("📊 Eval Suites", expanded=False):
        suites = asyncio.run(get_thread_eval_suites(thread_id))
        if suites:
            for suite in suites:
                st.markdown(f"**{suite.get('spec_name', 'N/A')}**")
                st.caption(f"Suite ID: {suite.get('suite_id', 'N/A')[:16]}...")
                st.caption(f"Tests: {len(suite.get('test_cases', []))}")
                st.markdown("---")
        else:
            st.caption("_No eval suites yet_")
    
    # Test Results Summary
    with st.expander("✅ Test Results", expanded=False):
        results = asyncio.run(get_thread_test_results(thread_id))
        if results:
            total = len(results)
            passed = sum(1 for r in results if r.get("passed"))
            failed = total - passed
            score = (passed / total * 100) if total > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Passed", passed)
            with col2:
                st.metric("Failed", failed)
            
            st.metric("Score", f"{score:.0f}%")
            
            # Show latest results
            st.caption("**Latest Results:**")
            for r in results[-3:]:  # Show last 3 results
                status = "✅" if r.get("passed") else "❌"
                st.caption(f"{status} {r.get('test_case_id', 'N/A')[:30]}...")
        else:
            st.caption("_No test results yet_")


def render_chat():
    """Render main chat interface"""
    st.title("💬 Chat with Seer")
    
    # Create 2-column layout: chat (70%) and thread info panel (30%)
    chat_col, info_col = st.columns([7, 3])
    
    with chat_col:
        # Thread controls in chat tab
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("➕ New Thread", key="new_thread_btn", disabled=st.session_state.processing_message):
                st.session_state.thread_id = None  # Will be set by LangGraph on first message
                st.session_state.langgraph_thread_id = None
                st.session_state.messages = []
                st.rerun()

        with col2:
            # Only update threads cache if not processing a message
            if not st.session_state.processing_message:
                threads = asyncio.run(list_threads())
                st.session_state.threads_cache = threads
            else:
                threads = st.session_state.threads_cache
            
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
            
            # Show selector (disabled during message processing to preserve thread_id)
            if display_options:
                selected_display = st.selectbox(
                    "Thread", 
                    display_options, 
                    index=default_index, 
                    key="thread_select",
                    disabled=st.session_state.processing_message
                )
                
                # Only handle selection changes if not processing
                if not st.session_state.processing_message:
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
            status = "🔒 Processing..." if st.session_state.processing_message else "✅ Ready"
            st.caption(f"Thread: {tid} | {status}")

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
        if prompt := st.chat_input("Message Seer...", disabled=st.session_state.processing_message):
            # Lock to prevent thread switching
            st.session_state.processing_message = True
            
            # Preserve current thread_id before sending
            current_thread_id = st.session_state.thread_id
            
            # Add user message to UI
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                # Send via FastAPI backend (handles thread creation and persistence)
                reply, returned_thread_id = asyncio.run(
                    send_message_to_orchestrator(prompt, current_thread_id)
                )
                
                # Update thread_id ONLY if it was newly created (None -> UUID) or explicitly changed by backend
                if returned_thread_id:
                    if current_thread_id is None:
                        # First message in new thread - save the thread_id
                        st.session_state.thread_id = returned_thread_id
                        st.session_state.langgraph_thread_id = returned_thread_id
                    elif current_thread_id != returned_thread_id:
                        # Backend returned different thread_id - log warning but use it
                        st.warning(f"⚠️ Thread ID changed: {current_thread_id[:8]}... → {returned_thread_id[:8]}...")
                        st.session_state.thread_id = returned_thread_id
                        st.session_state.langgraph_thread_id = returned_thread_id
                    # else: same thread_id, no update needed
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
            
            finally:
                # Unlock processing
                st.session_state.processing_message = False

            # Rerun to show new messages
            st.rerun()
    
    # Right panel: Thread info
    with info_col:
        render_thread_info_panel(st.session_state.thread_id)


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

