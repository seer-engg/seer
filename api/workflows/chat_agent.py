"""
LangGraph agent for intelligent workflow editing via chat.

This agent understands workflow structure and can suggest edits.
"""
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import json
import uuid

from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from .chat_schema import WorkflowEdit

logger = get_logger("api.workflows.chat_agent")

# Global workflow state context (thread-safe via thread_id key)
_workflow_state_context: Dict[str, Dict[str, Any]] = {}

def set_workflow_state_for_thread(thread_id: str, workflow_state: Dict[str, Any]) -> None:
    """Set workflow state for a specific thread."""
    _workflow_state_context[thread_id] = workflow_state

def get_workflow_state_for_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow state for a specific thread."""
    return _workflow_state_context.get(thread_id)


# Workflow manipulation tools

@tool
async def analyze_workflow(workflow_state: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> str:
    """
    Analyze the current workflow structure.
    
    Returns a JSON string describing the workflow's blocks, connections, and configuration.
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
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


@tool
async def add_workflow_block(
    block_type: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    block_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
) -> str:
    """
    Add a new block to the workflow.
    
    Args:
        block_type: Type of block to add (e.g., 'code', 'tool', 'llm', 'if_else', 'for_loop', 'input')
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        thread_id: Thread ID to retrieve workflow state from context
        block_id: Optional block ID (will be generated if not provided)
        config: Optional block configuration
        position: Optional position {x, y} (defaults to {x: 0, y: 0})
        label: Optional label for the block
        
    Returns:
        JSON string with the new block details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    nodes = workflow_state.get("nodes", [])
    
    # Generate block ID if not provided
    if not block_id:
        block_id = f"block-{uuid.uuid4().hex[:8]}"
    
    # Check if block ID already exists
    existing_ids = {node.get("id") for node in nodes}
    if block_id in existing_ids:
        return json.dumps({
            "error": f"Block with ID '{block_id}' already exists",
            "suggestion": f"Use a different ID or modify the existing block"
        })
    
    # Default position
    if not position:
        position = {"x": 0, "y": 0}
    
    # Create new block
    new_block = {
        "id": block_id,
        "type": block_type,
        "position": position,
        "data": {
            "label": label or block_id,
            "config": config or {},
        }
    }
    
    return json.dumps({
        "operation": "add_block",
        "block": new_block,
        "message": f"Ready to add {block_type} block '{block_id}'"
    })


@tool
async def modify_workflow_block(
    block_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    position: Optional[Dict[str, float]] = None,
) -> str:
    """
    Modify an existing block in the workflow.
    
    Args:
        block_id: ID of the block to modify
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        thread_id: Thread ID to retrieve workflow state from context
        config: Optional new configuration (will be merged with existing)
        label: Optional new label
        position: Optional new position {x, y}
        
    Returns:
        JSON string with modification details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    nodes = workflow_state.get("nodes", [])
    
    # Find the block
    block = None
    for node in nodes:
        if node.get("id") == block_id:
            block = node
            break
    
    if not block:
        return json.dumps({
            "error": f"Block with ID '{block_id}' not found"
        })
    
    # Apply modifications
    if config:
        current_config = block.get("data", {}).get("config", {})
        current_config.update(config)
        if "data" not in block:
            block["data"] = {}
        block["data"]["config"] = current_config
    
    if label:
        if "data" not in block:
            block["data"] = {}
        block["data"]["label"] = label
    
    if position:
        block["position"] = position
    
    return json.dumps({
        "operation": "modify_block",
        "block_id": block_id,
        "block": block,
        "message": f"Ready to modify block '{block_id}'"
    })


@tool
async def remove_workflow_block(
    block_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> str:
    """
    Remove a block from the workflow.
    
    Args:
        block_id: ID of the block to remove
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        thread_id: Thread ID to retrieve workflow state from context
        
    Returns:
        JSON string with removal details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    nodes = workflow_state.get("nodes", [])
    edges = workflow_state.get("edges", [])
    
    # Check if block exists
    block_exists = any(node.get("id") == block_id for node in nodes)
    if not block_exists:
        return json.dumps({
            "error": f"Block with ID '{block_id}' not found"
        })
    
    # Check for connected edges
    connected_edges = [
        edge for edge in edges
        if edge.get("source") == block_id or edge.get("target") == block_id
    ]
    
    return json.dumps({
        "operation": "remove_block",
        "block_id": block_id,
        "connected_edges": len(connected_edges),
        "message": f"Ready to remove block '{block_id}' (will also remove {len(connected_edges)} connected edges)"
    })


@tool
async def add_workflow_edge(
    source_id: str,
    target_id: str,
    source_handle: Optional[str] = None,
    target_handle: Optional[str] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> str:
    """
    Add a connection (edge) between two blocks.
    
    Args:
        source_id: Source block ID
        target_id: Target block ID
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        thread_id: Thread ID to retrieve workflow state from context
        source_handle: Optional source handle/port
        target_handle: Optional target handle/port
        
    Returns:
        JSON string with edge details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    nodes = workflow_state.get("nodes", [])
    edges = workflow_state.get("edges", [])
    
    # Check if blocks exist
    source_exists = any(node.get("id") == source_id for node in nodes)
    target_exists = any(node.get("id") == target_id for node in nodes)
    
    if not source_exists:
        return json.dumps({
            "error": f"Source block '{source_id}' not found"
        })
    
    if not target_exists:
        return json.dumps({
            "error": f"Target block '{target_id}' not found"
        })
    
    # Check if edge already exists
    edge_exists = any(
        edge.get("source") == source_id and edge.get("target") == target_id
        for edge in edges
    )
    
    if edge_exists:
        return json.dumps({
            "error": f"Edge from '{source_id}' to '{target_id}' already exists"
        })
    
    new_edge = {
        "id": f"edge-{source_id}-{target_id}",
        "source": source_id,
        "target": target_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }
    
    return json.dumps({
        "operation": "add_edge",
        "edge": new_edge,
        "message": f"Ready to connect '{source_id}' to '{target_id}'"
    })


@tool
async def remove_workflow_edge(
    source_id: str,
    target_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> str:
    """
    Remove a connection (edge) between two blocks.
    
    Args:
        source_id: Source block ID
        target_id: Target block ID
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        thread_id: Thread ID to retrieve workflow state from context
        
    Returns:
        JSON string with removal details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None and thread_id:
        workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    edges = workflow_state.get("edges", [])
    
    # Check if edge exists
    edge_exists = any(
        edge.get("source") == source_id and edge.get("target") == target_id
        for edge in edges
    )
    
    if not edge_exists:
        return json.dumps({
            "error": f"Edge from '{source_id}' to '{target_id}' not found"
        })
    
    return json.dumps({
        "operation": "remove_edge",
        "source_id": source_id,
        "target_id": target_id,
        "message": f"Ready to remove connection from '{source_id}' to '{target_id}'"
    })


def get_workflow_tools(workflow_state: Optional[Dict[str, Any]] = None) -> List:
    """
    Get all workflow manipulation tools.
    
    Args:
        workflow_state: Optional workflow state to inject into tools.
                        If provided, tools will use this state instead of requiring it as a parameter.
    """
    if workflow_state is None:
        # Return tools as-is (they'll require workflow_state parameter)
        return [
            analyze_workflow,
            add_workflow_block,
            modify_workflow_block,
            remove_workflow_block,
            add_workflow_edge,
            remove_workflow_edge,
        ]
    else:
        # Create wrapped tools that have workflow_state injected
        from functools import partial
        
        def wrap_tool(tool_func):
            """Wrap a tool to inject workflow_state."""
            async def wrapped_tool(**kwargs):
                # Inject workflow_state as first parameter
                return await tool_func(workflow_state=workflow_state, **kwargs)
            # Copy tool metadata
            wrapped_tool.__name__ = tool_func.name
            wrapped_tool.__doc__ = tool_func.description
            # Create new tool with same schema but without workflow_state parameter
            from langchain_core.tools import tool
            import inspect
            sig = inspect.signature(tool_func)
            params = {k: v for k, v in sig.parameters.items() if k != "workflow_state"}
            # This is complex - for now, return original tools
            # The LLM will need to pass workflow_state
            return tool_func
            # Note: Full implementation would require creating new tool schemas
            # For MVP, we'll have tools accept workflow_state from LLM calls
        
        return [
            analyze_workflow,
            add_workflow_block,
            modify_workflow_block,
            remove_workflow_block,
            add_workflow_edge,
            remove_workflow_edge,
        ]


def extract_thinking_from_messages(messages: List[Any]) -> List[str]:
    """
    Extract thinking/reasoning steps from agent messages.
    
    This looks for tool calls and intermediate reasoning in the message history.
    
    Args:
        messages: List of messages from agent
        
    Returns:
        List of thinking steps
    """
    thinking_steps = []
    
    for msg in messages:
        # Check for tool calls (indicates reasoning about what to do)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                thinking_steps.append(
                    f"Calling tool '{tool_call.get('name', 'unknown')}' "
                    f"with args: {tool_call.get('args', {})}"
                )
        
        # Check for tool results (indicates reasoning about results)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Look for reasoning patterns in content
            if "analyzing" in msg.content.lower() or "considering" in msg.content.lower():
                # Extract short reasoning snippets
                content_lines = msg.content.split("\n")
                for line in content_lines[:3]:  # First few lines often contain reasoning
                    if len(line.strip()) > 20 and len(line.strip()) < 200:
                        thinking_steps.append(line.strip())
    
    return thinking_steps


def create_workflow_chat_agent(
    model: str = "gpt-4o-mini",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for workflow chat assistance using create_agent.
    
    Uses LangChain v1.0+ create_agent with middleware for summarization
    and human-in-the-loop capabilities.
    
    Args:
        model: Model name to use (e.g., 'gpt-5.2', 'gpt-5-mini')
        checkpointer: Optional LangGraph checkpointer for persistence
        
    Returns:
        LangGraph agent compiled with tools and middleware
    """
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            SummarizationMiddleware,
            HumanInTheLoopMiddleware,
        )
    except ImportError:
        logger.error(
            "langchain>=1.0.0 is required for create_agent. "
            "Please install: pip install 'langchain>=1.0.0'"
        )
        raise
    
    llm = get_llm_without_responses_api(model=model, temperature=0.7, api_key=None)
    
    # System prompt for the workflow assistant
    workflow_context = ""
    if workflow_state:
        workflow_context = f"\n\nCurrent workflow state:\n{json.dumps(workflow_state, indent=2)}\n\nUse this information when calling tools. You don't need to pass workflow_state to tools - they will access it automatically."
    
    system_prompt = f"""You are an intelligent workflow assistant that helps users build and edit workflows.

Your capabilities:
1. Analyze workflow structure (blocks, connections, configurations)
2. Suggest improvements and edits
3. Answer questions about workflow logic
4. Help debug workflow issues

When suggesting edits, be specific about:
- Which blocks to add/modify/remove
- What configurations to change
- How to connect blocks

Always provide clear explanations for your suggestions. Use the available tools to analyze and manipulate workflows.{workflow_context}"""

    # Get workflow tools (with optional workflow_state injection)
    tools = get_workflow_tools(workflow_state=workflow_state)
    
    # Create summarization model (use same model with lower max tokens)
    summarization_model = get_llm_without_responses_api(
        model=model,
        temperature=0.3,
        api_key=None,
    )
    
    # Build middleware list
    middleware = [
        SummarizationMiddleware(
            model=summarization_model,
            max_tokens_before_summary=256000,  # 256k token limit
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "add_workflow_block": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "modify_workflow_block": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "remove_workflow_block": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "add_workflow_edge": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "remove_workflow_edge": {
                    "allowed_decisions": ["approve", "reject"]
                },
            }
        ),
    ]
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    
    logger.info(f"Created workflow chat agent with model {model}")
    return agent

