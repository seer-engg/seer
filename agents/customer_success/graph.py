"""Customer Success LangGraph - Deployable Graph"""

import os
from typing import Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import TypedDict

# Simple state without event bus dependencies
class AgentState(TypedDict):
    """State for Customer Success agent"""
    messages: Annotated[list[BaseMessage], add_messages]


# Simplified tools (no event bus dependency)
from langchain_core.tools import tool

@tool  
def acknowledge_confirmation(confirmed: bool, details: str = "") -> str:
    """
    Acknowledge user's confirmation response.
    
    Args:
        confirmed: Whether user confirmed
        details: Additional details from user
    
    Returns:
        JSON acknowledgment with metadata
    """
    import json
    return json.dumps({
        "action": "CONFIRMATION",
        "confirmed": confirmed,
        "details": details
    })


# Get tools list
TOOLS = [acknowledge_confirmation]


# System prompt
SYSTEM_PROMPT = """You are a Customer Success agent for Seer, an AI agent evaluation platform.

Your role:
- Help users evaluate their AI agents
- Acknowledge user requests warmly and professionally
- Handle user confirmations and relay them to the evaluation team

WORKFLOW:

1. **When user requests evaluation:**
   - Acknowledge their request warmly
   - Confirm you've received their requirements
   - Let them know the evaluation team is working on it
   - NO TOOL NEEDED - just respond naturally
   
   Example:
   User: "Evaluate my agent at localhost:2024, it should be polite"
   You: "Got it! I've received your evaluation request. Our evaluation team is analyzing your requirements and will generate test cases shortly. You'll hear from us soon!"

2. **When user confirms/responds to questions:**
   - Use acknowledge_confirmation tool to capture their response
   - Then respond naturally
   
   Examples:
   User: "Yes, run them"
   → YOU MUST call: acknowledge_confirmation(confirmed=true, details="run them")
   → Then say: "Perfect! Running the tests now..."
   
   User: "No, wait"
   → YOU MUST call: acknowledge_confirmation(confirmed=false, details="wait")
   → Then say: "No problem, I'll hold off on that."

IMPORTANT NOTES:
- The Event Bus broadcasts all user messages to all agents automatically
- You don't need to "forward" or "relay" evaluation requests - the eval team already sees them
- Your job is to provide excellent customer service and handle confirmations
- After using a tool and seeing its result, DO NOT call the same tool again. Just respond to the user."""


# Build the graph
def build_graph():
    """Build the customer success agent graph"""
    
    # Get LLM with aggressive tool binding
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,  # Lower temperature for more consistent tool usage
        api_key=os.getenv("OPENAI_API_KEY")
    ).bind_tools(TOOLS, tool_choice="auto")  # Explicitly set auto tool choice
    
    def agent_node(state: AgentState):
        """Main agent node"""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Check if we should continue to tools or end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    # Create graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Create the graph instance for langgraph dev
graph = build_graph()

