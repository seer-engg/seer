"""Base Agent Class - Common patterns for all Seer agents"""

import os
import json
from abc import ABC
from typing import List, Dict, Any, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from .error_handling import create_error_response, create_success_response
from .config import get_seer_config
from .a2a_utils import send_a2a_message


# Standalone tool functions
@tool
def think_agent_tool(observation: str, decision: str, reasoning: str) -> str:
    """
    Think about what you observe and decide whether to act or ignore.
    ALWAYS use this tool FIRST when you receive any input or event.
    
    Args:
        observation: What you observed
        decision: What you decided to do (ACT/IGNORE)
        reasoning: Why you made this decision
    
    Returns:
        JSON with your thought process
    """
    return json.dumps({
        "action": "THINK",
        "observation": observation,
        "decision": decision,
        "reasoning": reasoning
    })


@tool
async def send_to_orchestrator_tool(action: str, payload: Dict[str, Any] | None = None, thread_id: str | None = None) -> str:
    """
    Send any action to the orchestrator agent.
    
    Args:
        action: Action type (e.g., "user_confirmed", "eval_question")
        payload: Data to send with the action
        thread_id: Optional thread ID for the conversation
    
    Returns:
        Response from the orchestrator
    """
    try:
        payload = payload or {}
        config = get_seer_config()
        response = await send_a2a_message(
            "orchestrator",
            config.orchestrator_port,
            json.dumps({
                "action": action,
                "payload": payload,
                "thread_id": thread_id
            }),
            thread_id=thread_id
        )
        return create_success_response({"response": response})
    except Exception as e:
        return create_error_response(f"Failed to send message to orchestrator: {str(e)}", e)


class BaseAgentState(TypedDict):
    """Base state for all agents"""
    messages: Annotated[list[BaseMessage], add_messages]
    context: Dict[str, Any]  # Flexible context storage


class BaseAgent(ABC):
    """Base class for all Seer agents with common patterns"""
    
    def __init__(self, agent_name: str, system_prompt: str, tools: List = None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = (tools or []) + [think_agent_tool, send_to_orchestrator_tool]
        self.config = get_seer_config()
    
    def build_graph(self):
        """Build the agent graph with standard pattern"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(self.tools)
        
        def agent_node(state: BaseAgentState):
            """Main agent node"""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: BaseAgentState):
            """Check if we should continue to tools or end"""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
        
        # Create graph
        workflow = StateGraph(BaseAgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    # Registration removed; orchestrator seeds registry from deployment-config
