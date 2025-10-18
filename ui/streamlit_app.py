"""
Seer - Simplified Chat UI with Debugging
"""

import streamlit as st
import asyncio
import httpx
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_bus.schemas import EventMessage


# Configure page
st.set_page_config(
    page_title="Seer - Agent Evaluation",
    page_icon="🔮",
    layout="wide"
)


# Constants
EVENT_BUS_URL = os.getenv("EVENT_BUS_URL", "http://127.0.0.1:8000")


def init_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    if "last_check_timestamp" not in st.session_state:
        st.session_state.last_check_timestamp = None
    if "seen_message_ids" not in st.session_state:
        st.session_state.seen_message_ids = set()
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    if "poll_count" not in st.session_state:
        st.session_state.poll_count = 0


async def send_message_to_agent(message: str, thread_id: str):
    """Send a message via Event Bus (doesn't wait for response)"""
    async with httpx.AsyncClient() as client:
        # Publish MessageFromUser to Event Bus
        event = {
            "event_type": "MessageFromUser",
            "sender": "user",
            "thread_id": thread_id,
            "payload": {
                "content": message,
                "user_id": "lokesh"
            }
        }
        
        await client.post(f"{EVENT_BUS_URL}/publish", json=event)


async def get_thread_messages(thread_id: str, since_timestamp: str = None) -> list:
    """Get all MessageToUser events for a thread"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {"thread_id": thread_id, "limit": 50}
        if since_timestamp:
            params["since"] = since_timestamp
        
        try:
            response = await client.get(f"{EVENT_BUS_URL}/history", params=params)
            response.raise_for_status()
            
            # Handle empty or invalid response
            if not response.content:
                return []
            
            data = response.json()
            messages = data.get("messages", [])
            
            # Filter for MessageToUser from agents
            agent_messages = []
            for msg in messages:
                if (msg.get("event_type") == "MessageToUser" and 
                    msg.get("thread_id") == thread_id and
                    msg.get("sender") != "user"):
                    payload = msg.get("payload", {})
                    agent_messages.append({
                        "content": payload.get("content", ""),
                        "sender": msg.get("sender", "unknown"),
                        "timestamp": msg.get("timestamp", "")
                    })
            
            return agent_messages
        except Exception as e:
            st.error(f"Failed to fetch messages: {e}")
            return []


async def get_event_history(limit: int = 50):
    """Get recent events from event bus"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{EVENT_BUS_URL}/history",
                params={"limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data["messages"]
    except Exception as e:
        st.error(f"Failed to fetch event history: {e}")
        return []


async def get_all_threads():
    """Get all unique thread IDs from event bus"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EVENT_BUS_URL}/threads")
            response.raise_for_status()
            data = response.json()
            return data["threads"]
    except Exception as e:
        st.error(f"Failed to fetch threads: {e}")
        return []


async def get_thread_history(thread_id: str, limit: int = 50):
    """Get all events for a specific thread"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{EVENT_BUS_URL}/history",
                params={"thread_id": thread_id, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data["messages"]
    except Exception as e:
        st.error(f"Failed to fetch thread history: {e}")
        return []


async def get_bus_status():
    """Check if event bus is running"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{EVENT_BUS_URL}/")
            response.raise_for_status()
            return response.json()
    except Exception:
        return None

def render_chat():
    """Render main chat interface"""
    st.title("💬 Chat")
    
    # Thread selector in sidebar
    with st.sidebar:
        st.markdown("### 💬 Conversations")
        
        # Button to start new conversation
        if st.button("➕ New Conversation", use_container_width=True):
            import uuid
            new_thread_id = str(uuid.uuid4())
            st.session_state.thread_id = new_thread_id
            st.session_state.messages = []
            st.session_state.seen_message_ids = set()
            st.session_state.waiting_for_response = False
            st.rerun()
        
        st.markdown("---")
        
        # Get all threads
        threads = asyncio.run(get_all_threads())
        
        if threads:
            # Create thread options - only show existing threads
            thread_options = {}
            for t in threads:
                thread_id = t['thread_id']
                msg_count = t.get('message_count', 0)
                last_msg = t.get('last_message', '')[:19] if t.get('last_message') else 'N/A'
                label = f"{thread_id[:8]}... ({msg_count} msgs)"
                thread_options[label] = thread_id
            
            # Find current thread label
            current_label = None
            for label, tid in thread_options.items():
                if tid == st.session_state.thread_id:
                    current_label = label
                    break
            
            # If current thread is not in the list (new thread), show it
            if current_label is None:
                current_thread_id = st.session_state.thread_id
                current_label = f"{current_thread_id[:8]}... (new)"
                thread_options[current_label] = current_thread_id
            
            selected_label = st.selectbox(
                "Select conversation",
                list(thread_options.keys()),
                index=list(thread_options.keys()).index(current_label) if current_label else 0,
                key="chat_thread_selector"
            )
            
            selected_thread_id = thread_options[selected_label]
            
            # Switch thread if changed
            if selected_thread_id != st.session_state.thread_id:
                # Load selected thread
                st.session_state.thread_id = selected_thread_id
                st.session_state.messages = []
                st.session_state.seen_message_ids = set()
                st.session_state.waiting_for_response = False
                
                # Load thread history from database
                all_messages = asyncio.run(get_thread_messages(selected_thread_id))
                for msg in all_messages:
                    msg_id = f"{msg['sender']}_{msg['timestamp']}"
                    if msg_id not in st.session_state.seen_message_ids:
                        st.session_state.messages.append({"role": "assistant", "content": msg["content"]})
                        st.session_state.seen_message_ids.add(msg_id)
                st.rerun()
        else:
            st.info("No conversations yet")
        
        st.markdown("---")
        st.markdown(f"**Current Thread:**\n`{st.session_state.thread_id[:16]}...`")
    
    # Quick start guide
    if len(st.session_state.messages) == 0:
        st.info("""
        ### 🚀 Quick Start
        
        Tell me about the agent you want to evaluate. For example:
        
        **"Evaluate my agent at localhost:2024 (ID: my_agent). It should remember user preferences and respond politely."**
        """)
    
    # Check for new agent messages
    new_messages = asyncio.run(get_thread_messages(st.session_state.thread_id))
    
    had_new_messages = False
    for msg in new_messages:
        msg_id = f"{msg['sender']}_{msg['timestamp']}"
        if msg_id not in st.session_state.seen_message_ids:
            st.session_state.messages.append({"role": "assistant", "content": msg["content"]})
            st.session_state.seen_message_ids.add(msg_id)
            had_new_messages = True
    
    # If we got new messages, stop waiting
    if had_new_messages:
        st.session_state.waiting_for_response = False
        st.session_state.poll_count = 0
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show waiting indicator
    if st.session_state.waiting_for_response:
        with st.chat_message("assistant"):
            st.markdown("_Thinking..._")
    
    # Chat input
    if prompt := st.chat_input("Message..."):
        # Add user message to UI immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Send to event bus (non-blocking)
        asyncio.run(send_message_to_agent(prompt, st.session_state.thread_id))
        
        # Start waiting for response
        st.session_state.waiting_for_response = True
        st.session_state.poll_count = 0
        
        # Rerun to show user message and start polling for responses
        st.rerun()
    
    # Smart polling: only when waiting, with timeout
    if st.session_state.waiting_for_response:
        st.session_state.poll_count += 1
        
        # Stop polling after 30 attempts (60 seconds)
        if st.session_state.poll_count > 30:
            st.session_state.waiting_for_response = False
            st.session_state.poll_count = 0
            st.warning("⏱️ Response timeout. The agents might be busy. Try refreshing the page.")
        else:
            import time
            time.sleep(2)  # Poll every 2 seconds
            st.rerun()


def render_agent_threads():
    """Render agent threads debugging interface"""
    st.title("🤖 Agent Threads")
    st.markdown("Debug what each agent sees and does")
    
    # Get all threads
    threads = asyncio.run(get_all_threads())
    
    if not threads:
        st.info("No conversation threads yet. Start a conversation in the Chat tab!")
        return
    
    # Sort threads by most recent activity
    threads = sorted(threads, key=lambda t: t.get("last_message", ""), reverse=True)
    
    # Thread selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        thread_options = {
            f"{t['thread_id'][:8]}... ({t['message_count']} msgs, last: {t['last_message'][:19]})": t['thread_id']
            for t in threads
        }
        
        if not thread_options:
            st.warning("No threads found")
            return
        
        selected_display = st.selectbox("Select Thread", list(thread_options.keys()), key="thread_selector")
        selected_thread_id = thread_options[selected_display]
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_threads"):
            st.rerun()
    
    st.markdown("---")
    
    # Get thread history
    events = asyncio.run(get_thread_history(selected_thread_id, limit=50))
    
    if not events:
        st.info("No events in this thread")
        return
    
    # Reconstruct agent conversations
    cs_messages = []
    eval_messages = []
    
    for event in events:
        timestamp = event.get("timestamp", "")[:19]
        event_type = event.get("event_type", "")
        sender = event.get("sender", "")
        payload = event.get("payload", {})
        
        # Track what CS agent sees
        if event_type == "MessageFromUser":
            cs_messages.append({
                "type": "received",
                "timestamp": timestamp,
                "content": f"👤 User: {payload.get('content', '')}",
                "details": payload
            })
        elif event_type == "UserConfirmationQuery" and sender == "eval_agent":
            cs_messages.append({
                "type": "received",
                "timestamp": timestamp,
                "content": f"📨 Relay Request: {payload.get('question', '')}",
                "details": payload
            })
        elif event_type == "TestResultsReady" and sender == "eval_agent":
            cs_messages.append({
                "type": "received",
                "timestamp": timestamp,
                "content": f"📊 Results to Relay: {payload.get('summary', '')[:100]}...",
                "details": payload
            })
        elif sender == "customer_success":
            if event_type == "InitialAgentQuery":
                cs_messages.append({
                    "type": "sent",
                    "timestamp": timestamp,
                    "content": f"🎯 Sent Eval Request → Eval Agent",
                    "details": payload
                })
            elif event_type == "UserConfirmation":
                cs_messages.append({
                    "type": "sent",
                    "timestamp": timestamp,
                    "content": f"✅ User Confirmation: {payload.get('confirmed', False)}",
                    "details": payload
                })
            elif event_type == "MessageToUser":
                cs_messages.append({
                    "type": "sent",
                    "timestamp": timestamp,
                    "content": f"💬 To User: {payload.get('content', '')[:100]}...",
                    "details": payload
                })
        
        # Track what Eval agent sees
        if event_type == "InitialAgentQuery" and sender == "customer_success":
            eval_messages.append({
                "type": "received",
                "timestamp": timestamp,
                "content": f"🎯 Eval Request from CS",
                "details": payload
            })
        elif event_type == "UserConfirmation" and sender == "customer_success":
            eval_messages.append({
                "type": "received",
                "timestamp": timestamp,
                "content": f"✅ User Confirmed: {payload.get('confirmed', False)}",
                "details": payload
            })
        elif sender == "eval_agent":
            if event_type == "UserConfirmationQuery":
                eval_messages.append({
                    "type": "sent",
                    "timestamp": timestamp,
                    "content": f"❓ Request Confirmation: {payload.get('question', '')}",
                    "details": payload
                })
            elif event_type == "TestResultsReady":
                eval_messages.append({
                    "type": "sent",
                    "timestamp": timestamp,
                    "content": f"📊 Test Results Ready",
                    "details": payload
                })
    
    # Display side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💼 Customer Success Agent")
        if not cs_messages:
            st.info("No activity yet")
        else:
            for msg in cs_messages:
                if msg["type"] == "received":
                    with st.container():
                        st.markdown(f"**🟦 Received** `{msg['timestamp']}`")
                        st.info(msg["content"])
                        with st.expander("Details"):
                            st.json(msg["details"])
                else:
                    with st.container():
                        st.markdown(f"**🟩 Sent** `{msg['timestamp']}`")
                        st.success(msg["content"])
                        with st.expander("Details"):
                            st.json(msg["details"])
                st.markdown("---")
    
    with col2:
        st.subheader("🔬 Eval Agent")
        if not eval_messages:
            st.info("No activity yet")
        else:
            for msg in eval_messages:
                if msg["type"] == "received":
                    with st.container():
                        st.markdown(f"**🟦 Received** `{msg['timestamp']}`")
                        st.info(msg["content"])
                        with st.expander("Details"):
                            st.json(msg["details"])
                else:
                    with st.container():
                        st.markdown(f"**🟩 Sent** `{msg['timestamp']}`")
                        st.success(msg["content"])
                        with st.expander("Details"):
                            st.json(msg["details"])
                st.markdown("---")


def render_event_bus():
    """Render event bus monitoring interface"""
    st.title("📡 Event Bus Monitor")
    st.markdown("Real-time view of agent communication")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_event_bus"):
            st.rerun()
        
        limit = st.selectbox("Show last", [10, 20, 50, 100], index=1, key="event_limit")
    
    with col2:
        status = asyncio.run(get_bus_status())
        if status:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Messages", status.get("total_messages", 0))
            with col_b:
                st.metric("Active Subscribers", status.get("active_subscribers", 0))
            with col_c:
                subscribers = status.get("subscribers", [])
                st.metric("Agents Online", len(subscribers))
    
    st.markdown("---")
    
    # Get event history
    events = asyncio.run(get_event_history(limit=limit))
    
    if not events:
        st.info("No events yet. Start a conversation to see activity!")
        return
    
    # Display events in reverse chronological order
    for event in reversed(events):
        event_type = event.get("event_type", "Unknown")
        sender = event.get("sender", "Unknown")
        timestamp = event.get("timestamp", "")
        payload = event.get("payload", {})
        thread_id = event.get("thread_id", "None")
        
        # Color code by event type
        if event_type == "MessageFromUser":
            icon = "👤"
            color = "blue"
        elif event_type == "MessageToUser":
            icon = "🤖"
            color = "green"
        elif event_type == "InitialAgentQuery":
            icon = "🎯"
            color = "orange"
        elif event_type == "UserConfirmationQuery":
            icon = "❓"
            color = "orange"
        elif event_type == "UserConfirmation":
            icon = "✅"
            color = "green"
        elif event_type == "TestResultsReady":
            icon = "📊"
            color = "green"
        elif event_type == "AgentStarted":
            icon = "🚀"
            color = "gray"
        else:
            icon = "📨"
            color = "gray"
        
        with st.expander(f"{icon} **{event_type}** from `{sender}` - {timestamp[:19]}", expanded=False):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"**Sender:** `{sender}`")
                if thread_id and thread_id != "None":
                    st.markdown(f"**Thread:** `{thread_id[:8]}...`")
                else:
                    st.markdown("**Thread:** None")
                st.markdown(f"**Time:** {timestamp[:19]}")
            
            with col2:
                st.markdown("**Payload:**")
                
                # Pretty print payload
                if event_type == "MessageFromUser":
                    st.info(payload.get("content", ""))
                elif event_type == "MessageToUser":
                    msg_type = payload.get("message_type", "info")
                    content = payload.get("content", "")
                    if msg_type == "success":
                        st.success(content)
                    elif msg_type == "error":
                        st.error(content)
                    elif msg_type == "question":
                        st.warning(content)
                    else:
                        st.info(content)
                elif event_type == "InitialAgentQuery":
                    st.markdown(f"**Target:** `{payload.get('target_agent_url', 'N/A')}`")
                    st.markdown(f"**Agent ID:** `{payload.get('target_agent_id', 'N/A')}`")
                    st.markdown(f"**Expectations:** {payload.get('expectations', 'N/A')[:100]}...")
                elif event_type == "TestResultsReady":
                    st.markdown(f"**Passed:** {payload.get('passed', 0)}/{payload.get('total_tests', 0)}")
                    st.markdown(f"**Score:** {payload.get('overall_score', 0):.0f}%")
                    st.markdown(f"**Summary:**\n{payload.get('summary', 'N/A')}")
                else:
                    st.json(payload)




def main():
    """Main app entry point"""
    init_session_state()
    
    # Navigation using tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🤖 Agent Threads", "📡 Event Bus"])
    
    with tab1:
        render_chat()
    
    with tab2:
        render_agent_threads()
    
    with tab3:
        render_event_bus()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
    <small>Seer - Event-Driven Multi-Agent System | 
    <a href='{EVENT_BUS_URL}/docs' target='_blank'>Event Bus API</a>
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

