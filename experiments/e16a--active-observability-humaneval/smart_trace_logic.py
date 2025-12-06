import os
import json
from typing import List, Any
import time
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI

MAX_IO_CHARS = 1000

def _short(obj: Any) -> str:
    """Truncate long strings/objects for display."""
    if isinstance(obj, str):
        s = obj
    else:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    
    return (s[:MAX_IO_CHARS] + "...") if s and len(s) > MAX_IO_CHARS else s

def get_semantic_summary(messages: List[BaseMessage]) -> str:
    """
    Uses a cheap LLM (gpt-5-mini) to semantically summarize the message history.
    Focuses on:
    1. What was attempted (Intent)
    2. What happened (Result/Error)
    3. Current Status
    """
    # Filter for relevant messages (Human prompts and Tool interactions)
    # Exclude previous progress summaries to avoid recursion/noise
    filtered_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_msgs.append(f"USER: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get('name')
                    if name != 'get_progress_summary':
                         args = tc.get('args')
                         filtered_msgs.append(f"AGENT TOOL CALL: {name}({_short(args)})")
            elif msg.content:
                filtered_msgs.append(f"AGENT THOUGHT: {msg.content}")
        elif isinstance(msg, ToolMessage):
            # Check if this is a response to get_progress_summary (skip it)
            if msg.name != 'get_progress_summary':
                content = str(msg.content)
                # Truncate output for summarizer context window efficiency
                filtered_msgs.append(f"TOOL OUTPUT: {_short(content)}")

    if not filtered_msgs:
        return "📝 No relevant history to summarize yet."

    # Prepare prompt for the summarizer
    history_text = "\n".join(filtered_msgs)
    
    prompt = f"""You are a technical execution log summarizer.
Analyze the following agent execution history and provide a concise, semantic summary.

RULES:
1. Group related actions (e.g. "Defined function X and ran tests").
2. Highlight FAILURES explicitly with "🛑".
3. Highlight SUCCESS explicitly with "✅".
4. Be specific about WHAT was implemented or tested (function names, logic).
5. Ignore previous summary requests.
6. Keep it under 5 bullet points.

HISTORY:
{history_text}

SUMMARY:"""

    # Use cheap model
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0, reasoning_effort="minimal")
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"⚠️ Failed to generate semantic summary: {e}"

def get_smart_summary_from_messages(messages: List[BaseMessage]) -> str:
    """
    Wrapper that now calls the semantic summarizer.
    """
    if not messages:
        return "⚠️ No messages available yet."
    
    print(f"📊 Generating SEMANTIC summary for {len(messages)} messages...")
    start_analysis = time.time()
    
    summary = get_semantic_summary(messages)
    
    end_analysis = time.time()
    analysis_time_ms = (end_analysis - start_analysis) * 1000
    
    footer = f"\n\n(Semantic Analysis took {analysis_time_ms:.2f}ms)"
    return summary + footer

if __name__ == "__main__":
    # Test with a dummy session if run directly
    print("Run run_experiment.py to generate real traces first.")
