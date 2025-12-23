"""
LangGraph agent for intelligent workflow editing via chat.

This agent understands workflow structure and can suggest edits.
"""
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json

from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from .chat_schema import WorkflowEdit

logger = get_logger("api.workflows.chat_agent")


class ChatState(TypedDict):
    """State for the workflow chat agent."""
    messages: List[BaseMessage]
    workflow_state: Dict[str, Any]
    suggested_edits: List[WorkflowEdit]
    response: str


@tool
async def analyze_workflow(workflow_state: Dict[str, Any]) -> str:
    """
    Analyze the current workflow structure.
    
    Returns a JSON string describing the workflow's blocks, connections, and configuration.
    """
    nodes = workflow_state.get("nodes", [])
    edges = workflow_state.get("edges", [])
    
    analysis = {
        "total_blocks": len(nodes),
        "total_connections": len(edges),
        "block_types": {},
        "blocks": [],
        "connections": [],
    }
    
    # Count block types
    for node in nodes:
        block_type = node.get("type", "unknown")
        analysis["block_types"][block_type] = analysis["block_types"].get(block_type, 0) + 1
        
        # Add block details
        analysis["blocks"].append({
            "id": node.get("id"),
            "type": block_type,
            "label": node.get("data", {}).get("label", ""),
            "config": node.get("data", {}).get("config", {}),
        })
    
    # Add connection details
    for edge in edges:
        analysis["connections"].append({
            "source": edge.get("source"),
            "target": edge.get("target"),
            "source_handle": edge.get("sourceHandle"),
            "target_handle": edge.get("targetHandle"),
        })
    
    return json.dumps(analysis, indent=2)


def create_workflow_chat_agent(model: str = "gpt-4o-mini"):
    """
    Create a LangGraph agent for workflow chat assistance.
    
    This is a simplified version that uses LangChain directly.
    For a full LangGraph implementation, we would use StateGraph.
    
    Args:
        model: Model name to use (e.g., 'gpt-5.2', 'gpt-5-mini')
    """
    llm = get_llm_without_responses_api(model=model, temperature=0.7, api_key=None)
    
    # System prompt for the workflow assistant
    system_prompt = """You are an intelligent workflow assistant that helps users build and edit workflows.

Your capabilities:
1. Analyze workflow structure (blocks, connections, configurations)
2. Suggest improvements and edits
3. Answer questions about workflow logic
4. Help debug workflow issues

When suggesting edits, be specific about:
- Which blocks to add/modify/remove
- What configurations to change
- How to connect blocks

Always provide clear explanations for your suggestions."""

    async def chat_agent(state: ChatState) -> ChatState:
        """Main agent function that processes chat messages."""
        # Handle both dict and TypedDict inputs
        if isinstance(state, dict):
            messages_list = state.get("messages", [])
            workflow_state = state.get("workflow_state", {})
        else:
            messages_list = []
            workflow_state = {}
        
        # Convert message dicts to BaseMessage objects if needed
        messages = []
        for msg in messages_list:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            elif isinstance(msg, BaseMessage):
                messages.append(msg)
        
        # Build prompt with system message and workflow context
        prompt_messages = [
            SystemMessage(content=system_prompt),
        ]
        
        # Add workflow analysis
        if workflow_state:
            try:
                workflow_analysis = await analyze_workflow.ainvoke({"workflow_state": workflow_state})
                prompt_messages.append(
                    SystemMessage(content=f"Current workflow state:\n{workflow_analysis}")
                )
            except Exception as e:
                logger.warning(f"Failed to analyze workflow: {e}")
        
        # Add conversation history
        prompt_messages.extend(messages)
        
        # Get response from LLM (use async ainvoke)
        try:
            response = await llm.ainvoke(prompt_messages)
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}", exc_info=True)
            response_text = "I apologize, but I encountered an error processing your request. Please try again."
        
        # Parse suggested edits from response (if any)
        suggested_edits = []
        
        # Try to extract JSON edits from response if present
        try:
            # Look for JSON in the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                edits_data = json.loads(json_str)
                if isinstance(edits_data, list):
                    suggested_edits = [WorkflowEdit(**edit) for edit in edits_data]
                elif isinstance(edits_data, dict) and "edits" in edits_data:
                    suggested_edits = [WorkflowEdit(**edit) for edit in edits_data["edits"]]
        except Exception as e:
            logger.debug(f"Could not parse edits from response: {e}")
        
        return {
            "messages": messages + [AIMessage(content=response_text)],
            "workflow_state": workflow_state,
            "suggested_edits": suggested_edits,
            "response": response_text,
        }
    
    return chat_agent

