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


@tool
def get_test_cases(agent_url: str, agent_id: str) -> str:
    """
    Get test cases for a specific agent from the Event Bus.
    Use this when the user asks about test cases/tests/evals.
    
    Args:
        agent_url: URL of the target agent (e.g. "http://localhost:2024")
        agent_id: ID of the target agent (e.g. "deep_researcher")
    
    Returns:
        JSON string with test cases or error message
    """
    import httpx
    import json
    import os
    
    event_bus_url = os.getenv("EVENT_BUS_URL", "http://127.0.0.1:8000")
    
    try:
        response = httpx.get(
            f"{event_bus_url}/evals",
            params={"agent_url": agent_url, "agent_id": agent_id},
            timeout=5.0
        )
        response.raise_for_status()
        data = response.json()
        
        eval_suites = data.get("eval_suites", [])
        if not eval_suites:
            return json.dumps({
                "success": False,
                "message": f"No test cases found for {agent_url}/{agent_id}"
            })
        
        # Get the most recent eval suite
        latest_suite = eval_suites[-1]
        test_cases = latest_suite.get("test_cases", [])
        
        return json.dumps({
            "success": True,
            "eval_suite_id": latest_suite.get("id"),
            "test_count": len(test_cases),
            "test_cases": test_cases
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


# Get tools list
TOOLS = [acknowledge_confirmation, get_test_cases]


# System prompt
SYSTEM_PROMPT = """You are a Customer Success agent for Seer, an AI agent evaluation platform.

Your role:
- Help users evaluate their AI agents
- Acknowledge user requests warmly and professionally
- Handle user confirmations and relay them to the evaluation team
- Answer questions about test cases

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

3. **When user asks about test cases:**
   - Use get_test_cases tool to retrieve them
   - Format and present them clearly
   
   Examples:
   User: "What are those 12 cases?"
   → YOU MUST call: get_test_cases(agent_url="http://localhost:2024", agent_id="deep_researcher")
   → Then format the results nicely for the user
   
   User: "Show me the tests" or "What tests did you generate?"
   → Same approach - use get_test_cases

IMPORTANT NOTES:
- The Event Bus broadcasts all user messages to all agents automatically
- You don't need to "forward" or "relay" evaluation requests - the eval team already sees them
- Test cases are stored centrally in the Event Bus, accessible to all agents
- Your job is to provide excellent customer service and answer questions
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

