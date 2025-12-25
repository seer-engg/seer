"""
LangGraph agent for intelligent workflow editing via chat.

This agent understands workflow structure and can suggest edits.
"""
from typing import Any, Dict, List, Optional
from contextvars import ContextVar
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import json
import uuid

from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from shared.tools.registry import get_tools_by_integration
from .chat_schema import WorkflowEdit

logger = get_logger("api.workflows.chat_agent")

# Context variable to track current thread_id in tool execution
_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

# Global workflow state context (thread-safe via thread_id key)
_workflow_state_context: Dict[str, Dict[str, Any]] = {}

def set_workflow_state_for_thread(thread_id: str, workflow_state: Dict[str, Any]) -> None:
    """Set workflow state for a specific thread."""
    _workflow_state_context[thread_id] = workflow_state

def get_workflow_state_for_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow state for a specific thread."""
    return _workflow_state_context.get(thread_id)

# Global planned edits context (thread-safe via thread_id key)
_planned_edits_context: Dict[str, Dict[str, Dict[str, Any]]] = {}

def set_planned_edits_for_thread(thread_id: str, planned_edits: Dict[str, Dict[str, Any]]) -> None:
    """Set planned edits for a specific thread."""
    _planned_edits_context[thread_id] = planned_edits

def get_planned_edits_for_thread(thread_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get planned edits for a specific thread."""
    return _planned_edits_context.get(thread_id)

def get_planned_edit(thread_id: str, edit_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific planned edit by ID."""
    planned_edits = get_planned_edits_for_thread(thread_id)
    if planned_edits:
        return planned_edits.get(edit_id)
    return None


# Workflow manipulation tools

@tool
async def analyze_workflow(
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Analyze the current workflow structure.
    
    Returns a JSON string describing the workflow's blocks, connections, and configuration.
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
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
            "branch": edge.get("data", {}).get("branch"),
        })
    
    return json.dumps(analysis, indent=2)


@tool
async def add_workflow_block(
    block_type: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    block_id: Optional[str] = None,
    block_config: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
) -> str:
    """
    Add a new block to the workflow.
    
    Args:
        block_type: Type of block to add (e.g., 'code', 'tool', 'llm', 'if_else', 'for_loop', 'input')
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        block_id: Optional block ID (will be generated if not provided)
        block_config: Optional block configuration
        position: Optional position {x, y} (defaults to {x: 0, y: 0})
        label: Optional label for the block
        
    Returns:
        JSON string with the new block details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
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
    
    # Process block_config: transform inputs to params for tool blocks
    processed_config = block_config or {}
    if block_type == "tool" and "inputs" in processed_config and "params" not in processed_config:
        processed_config = {
            **processed_config,
            "params": processed_config.pop("inputs"),
        }
    
    # Create new block
    new_block = {
        "id": block_id,
        "type": block_type,
        "position": position,
        "data": {
            "label": label or block_id,
            "config": processed_config,
        }
    }
    
    # Generate edit_id and store planned edit in thread context
    edit_id = f"edit-{uuid.uuid4().hex[:8]}"
    if thread_id:
        planned_edits = get_planned_edits_for_thread(thread_id) or {}
        planned_edits[edit_id] = {
            "operation": "add_block",
            "block": new_block,
            "timestamp": json.dumps({"iso": "now"}),  # Simple timestamp placeholder
        }
        set_planned_edits_for_thread(thread_id, planned_edits)
    
    return json.dumps({
        "operation": "add_block",
        "edit_id": edit_id,
        "block": new_block,
        "message": f"Ready to add {block_type} block '{block_id}'. Use apply_workflow_edit with edit_id '{edit_id}' to apply this change."
    })


@tool
async def modify_workflow_block(
    block_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
    block_config: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    position: Optional[Dict[str, float]] = None,
) -> str:
    """
    Modify an existing block in the workflow.
    
    Args:
        block_id: ID of the block to modify
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        block_config: Optional new configuration (will be merged with existing)
        label: Optional new label
        position: Optional new position {x, y}
        
    Returns:
        JSON string with modification details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
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
    
    # Create modified block (don't modify original yet)
    modified_block = block.copy()
    
    # Apply modifications to copy
    if block_config:
        current_config = modified_block.get("data", {}).get("config", {})
        current_config.update(block_config)
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["config"] = current_config
    
    if label:
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["label"] = label
    
    if position:
        modified_block["position"] = position
    
    # Generate edit_id and store planned edit in thread context
    edit_id = f"edit-{uuid.uuid4().hex[:8]}"
    if thread_id:
        planned_edits = get_planned_edits_for_thread(thread_id) or {}
        planned_edits[edit_id] = {
            "operation": "modify_block",
            "block_id": block_id,
            "block": modified_block,
            "timestamp": json.dumps({"iso": "now"}),
        }
        set_planned_edits_for_thread(thread_id, planned_edits)
    
    return json.dumps({
        "operation": "modify_block",
        "edit_id": edit_id,
        "block_id": block_id,
        "block": modified_block,
        "message": f"Ready to modify block '{block_id}'. Use apply_workflow_edit with edit_id '{edit_id}' to apply this change."
    })


@tool
async def remove_workflow_block(
    block_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Remove a block from the workflow.
    
    Args:
        block_id: ID of the block to remove
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        
    Returns:
        JSON string with removal details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
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
    
    # Generate edit_id and store planned edit in thread context
    edit_id = f"edit-{uuid.uuid4().hex[:8]}"
    if thread_id:
        planned_edits = get_planned_edits_for_thread(thread_id) or {}
        planned_edits[edit_id] = {
            "operation": "remove_block",
            "block_id": block_id,
            "connected_edges": len(connected_edges),
            "timestamp": json.dumps({"iso": "now"}),
        }
        set_planned_edits_for_thread(thread_id, planned_edits)
    
    return json.dumps({
        "operation": "remove_block",
        "edit_id": edit_id,
        "block_id": block_id,
        "connected_edges": len(connected_edges),
        "message": f"Ready to remove block '{block_id}' (will also remove {len(connected_edges)} connected edges). Use apply_workflow_edit with edit_id '{edit_id}' to apply this change."
    })


@tool
async def add_workflow_edge(
    source_id: str,
    target_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Add a connection (edge) between two blocks.
    
    Args:
        source_id: Source block ID
        target_id: Target block ID
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        
    Returns:
        JSON string with edge details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
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
    
    branch = None
    source_node = next((node for node in nodes if node.get("id") == source_id), None)
    if source_node and source_node.get("type") == "if_else":
        existing_edges = [edge for edge in edges if edge.get("source") == source_id]
        existing_branches = {
            edge.get("data", {}).get("branch") for edge in existing_edges
        }
        if "true" not in existing_branches:
            branch = "true"
        elif "false" not in existing_branches:
            branch = "false"
        else:
            return json.dumps({
                "error": f"If/Else block '{source_id}' already has true/false branches defined"
            })
    
    new_edge = {
        "id": f"edge-{source_id}-{target_id}",
        "source": source_id,
        "target": target_id,
        **({"data": {"branch": branch}} if branch else {}),
    }
    
    # Generate edit_id and store planned edit in thread context
    edit_id = f"edit-{uuid.uuid4().hex[:8]}"
    if thread_id:
        planned_edits = get_planned_edits_for_thread(thread_id) or {}
        planned_edits[edit_id] = {
            "operation": "add_edge",
            "edge": new_edge,
            "timestamp": json.dumps({"iso": "now"}),
        }
        set_planned_edits_for_thread(thread_id, planned_edits)
    
    return json.dumps({
        "operation": "add_edge",
        "edit_id": edit_id,
        "edge": new_edge,
        "message": f"Ready to connect '{source_id}' to '{target_id}'. Use apply_workflow_edit with edit_id '{edit_id}' to apply this change."
    })


@tool
async def remove_workflow_edge(
    source_id: str,
    target_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Remove a connection (edge) between two blocks.
    
    Args:
        source_id: Source block ID
        target_id: Target block ID
        workflow_state: Current workflow state (nodes and edges) - optional, will be retrieved from context if not provided
        
    Returns:
        JSON string with removal details
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
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
    
    # Generate edit_id and store planned edit in thread context
    edit_id = f"edit-{uuid.uuid4().hex[:8]}"
    if thread_id:
        planned_edits = get_planned_edits_for_thread(thread_id) or {}
        planned_edits[edit_id] = {
            "operation": "remove_edge",
            "source_id": source_id,
            "target_id": target_id,
            "timestamp": json.dumps({"iso": "now"}),
        }
        set_planned_edits_for_thread(thread_id, planned_edits)
    
    return json.dumps({
        "operation": "remove_edge",
        "edit_id": edit_id,
        "source_id": source_id,
        "target_id": target_id,
        "message": f"Ready to remove connection from '{source_id}' to '{target_id}'. Use apply_workflow_edit with edit_id '{edit_id}' to apply this change."
    })


@tool
async def apply_workflow_edit(
    edit_id: str,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Apply a previously planned workflow edit.
    
    This tool actually modifies the workflow state. Use this AFTER the user has approved
    a plan that was shown via suggested_edits.
    
    Args:
        edit_id: ID of the edit to apply (from a previous plan)
        workflow_state: Current workflow state - optional, will be retrieved from context if not provided
        
    Returns:
        Confirmation message with details of what was applied
    """
    # Get workflow_state from parameter or thread context
    if workflow_state is None:
        thread_id = _current_thread_id.get()
        if thread_id:
            workflow_state = get_workflow_state_for_thread(thread_id)
    if workflow_state is None:
        return json.dumps({"error": "Workflow state not available"})
    
    thread_id = _current_thread_id.get()
    
    # Retrieve the planned edit
    planned_edit = get_planned_edit(thread_id, edit_id) if thread_id else None
    if not planned_edit:
        return json.dumps({
            "error": f"Planned edit with ID '{edit_id}' not found. It may have expired or been cleared."
        })
    
    operation = planned_edit.get("operation")
    nodes = workflow_state.get("nodes", [])
    edges = workflow_state.get("edges", [])
    
    # Apply the edit based on operation type
    if operation == "add_block":
        block = planned_edit.get("block")
        if block:
            nodes.append(block)
            workflow_state["nodes"] = nodes
            # Update thread context with modified workflow state
            if thread_id:
                set_workflow_state_for_thread(thread_id, workflow_state)
            return json.dumps({
                "success": True,
                "operation": "add_block",
                "block_id": block.get("id"),
                "message": f"Successfully added {block.get('type')} block '{block.get('id')}' to the workflow."
            })
    
    elif operation == "modify_block":
        block_id = planned_edit.get("block_id")
        modified_block = planned_edit.get("block")
        if block_id and modified_block:
            # Find and replace the block
            for i, node in enumerate(nodes):
                if node.get("id") == block_id:
                    nodes[i] = modified_block
                    workflow_state["nodes"] = nodes
                    # Update thread context with modified workflow state
                    if thread_id:
                        set_workflow_state_for_thread(thread_id, workflow_state)
                    return json.dumps({
                        "success": True,
                        "operation": "modify_block",
                        "block_id": block_id,
                        "message": f"Successfully modified block '{block_id}'."
                    })
            return json.dumps({
                "error": f"Block '{block_id}' not found in workflow"
            })
    
    elif operation == "remove_block":
        block_id = planned_edit.get("block_id")
        if block_id:
            # Remove the block
            nodes = [node for node in nodes if node.get("id") != block_id]
            workflow_state["nodes"] = nodes
            # Also remove connected edges
            edges = [edge for edge in edges if edge.get("source") != block_id and edge.get("target") != block_id]
            workflow_state["edges"] = edges
            # Update thread context with modified workflow state
            if thread_id:
                set_workflow_state_for_thread(thread_id, workflow_state)
            return json.dumps({
                "success": True,
                "operation": "remove_block",
                "block_id": block_id,
                "message": f"Successfully removed block '{block_id}' and its connected edges."
            })
    
    elif operation == "add_edge":
        edge = planned_edit.get("edge")
        if edge:
            edges.append(edge)
            workflow_state["edges"] = edges
            # Update thread context with modified workflow state
            if thread_id:
                set_workflow_state_for_thread(thread_id, workflow_state)
            return json.dumps({
                "success": True,
                "operation": "add_edge",
                "source_id": edge.get("source"),
                "target_id": edge.get("target"),
                "message": f"Successfully connected '{edge.get('source')}' to '{edge.get('target')}'."
            })
    
    elif operation == "remove_edge":
        source_id = planned_edit.get("source_id")
        target_id = planned_edit.get("target_id")
        if source_id and target_id:
            # Remove the edge
            edges = [edge for edge in edges if not (edge.get("source") == source_id and edge.get("target") == target_id)]
            workflow_state["edges"] = edges
            # Update thread context with modified workflow state
            if thread_id:
                set_workflow_state_for_thread(thread_id, workflow_state)
            return json.dumps({
                "success": True,
                "operation": "remove_edge",
                "source_id": source_id,
                "target_id": target_id,
                "message": f"Successfully removed connection from '{source_id}' to '{target_id}'."
            })
    
    return json.dumps({
        "error": f"Unknown operation '{operation}'"
    })


# Dynamic tool discovery tools

async def _search_tools_local(
    query: str,
    integration_name: Optional[List[str]] = None,
    top_k: int = 5
) -> List[dict]:
    """
    Search tools from local Chroma vector store using semantic search.
    
    Args:
        query: Search query string
        integration_name: Optional list of integration names to restrict search (e.g., ["github", "asana"])
        top_k: Number of results to return
    
    Returns:
        List of tool dictionaries
    """
    try:
        from shared.tool_hub.singleton import get_toolhub_instance
        toolhub = get_toolhub_instance()
        if toolhub is None:
            raise ValueError("LocalToolHub not available")
        
        results = await toolhub.query(
            query=query,
            integration_name=integration_name,
            top_k=top_k
        )
        return results
    except Exception as e:
        logger.warning(f"Local tool search not available: {e}")
        # Fallback to registry-based search
        all_tools = get_tools_by_integration()
        # Simple keyword matching fallback
        query_lower = query.lower()
        matching_tools = []
        for tool_meta in all_tools:
            tool_name = tool_meta.get("name", "").lower()
            tool_desc = tool_meta.get("description", "").lower()
            if query_lower in tool_name or query_lower in tool_desc:
                matching_tools.append(tool_meta)
                if len(matching_tools) >= top_k:
                    break
        return matching_tools[:top_k]


@tool
async def search_tools(
    query: str,
    reasoning: str = "",
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Search for available tools/actions using semantic search.
    
    Use this tool when you need to discover what tools are available for a specific capability.
    For example, if the user wants to "search emails" or "find Gmail messages", use this tool
    to discover the relevant Gmail tools.
    
    **QUERY GUIDELINES:**
    - Search for CAPABILITIES, not specific data values
    - Use specific, action-oriented queries
    - GOOD: "search emails", "find Gmail messages", "create Asana task", "list GitHub pull requests"
    - BAD: "Gmail", "GitHub", "search emails with subject 'test'" (includes actual data)
    
    Args:
        query: Search query describing the capability/action needed (e.g., "search emails", "create task")
        reasoning: Optional explanation of why you need this tool and what you're trying to accomplish
        integration_filter: Optional list of integration names to restrict search (e.g., ["gmail", "github"])
    
    Returns:
        JSON string with list of matching tools, their descriptions, and parameters
    """
    try:
        results = await _search_tools_local(
            query=query,
            integration_name=integration_filter,
            top_k=5
        )
        
        if not results:
            return json.dumps({
                "tools": [],
                "message": f"No tools found matching query: {query}. Try a different search term or use list_available_tools to see all tools."
            })
        
        # Format results
        tools_list = []
        for tool_data in results:
            tools_list.append({
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                "parameters": tool_data.get("parameters", {}),
                "integration": tool_data.get("service", tool_data.get("integration_type", ""))
            })
        
        return json.dumps({
            "tools": tools_list,
            "query": query,
            "reasoning": reasoning or "Searching for tools to fulfill user request"
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error searching tools: {e}")
        return json.dumps({
            "tools": [],
            "error": str(e),
            "message": "Tool search failed. Try using list_available_tools to see all available tools."
        })


@tool
async def list_available_tools(integration_type: Optional[str] = None) -> str:
    """
    List all available tools from the registry.
    
    Use this tool when you need to see what tools are available, especially when search_tools
    doesn't return what you need. You can filter by integration type (e.g., "gmail", "github").
    
    Args:
        integration_type: Optional integration type to filter by (e.g., "gmail", "github", "asana")
    
    Returns:
        JSON string with list of all available tools and their metadata
    """
    try:
        tools = get_tools_by_integration(integration_type=integration_type)
        
        tools_list = []
        for tool_meta in tools:
            tools_list.append({
                "name": tool_meta.get("name", ""),
                "description": tool_meta.get("description", ""),
                "parameters": tool_meta.get("parameters", {}),
                "integration_type": tool_meta.get("integration_type", ""),
            })
        
        return json.dumps({
            "tools": tools_list,
            "total": len(tools_list),
            "integration_filter": integration_type or "all"
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error listing tools: {e}")
        return json.dumps({
            "tools": [],
            "error": str(e)
        })


def get_workflow_tools(workflow_state: Optional[Dict[str, Any]] = None) -> List:
    """
    Get all workflow manipulation tools and dynamic discovery tools.
    
    Args:
        workflow_state: Optional workflow state to inject into tools.
                        If provided, tools will use this state instead of requiring it as a parameter.
    """
    # Base tools that are always available
    base_tools = [
        # Workflow manipulation tools
        analyze_workflow,
        add_workflow_block,
        modify_workflow_block,
        remove_workflow_block,
        add_workflow_edge,
        remove_workflow_edge,
        apply_workflow_edit,
        # Dynamic tool discovery tools
        search_tools,
        list_available_tools,
    ]
    
    if workflow_state is None:
        # Return tools as-is (they'll require workflow_state parameter)
        return base_tools
    else:
        # Note: Discovery tools don't need workflow_state, so we return them as-is
        # Workflow manipulation tools will need workflow_state passed by LLM
        return base_tools


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
        workflow_context = f"\n\nCurrent workflow state:\n{json.dumps(workflow_state, indent=2)}\n\nUse this information when calling tools. Tools automatically access workflow state from thread context via runtime configuration."
    
    system_prompt = f"""You are an intelligent workflow assistant that helps users build and edit workflows. Your role is to understand user intent, discover appropriate tools, and create workflows that achieve their goals.

**Core Principles:**
- Focus on what users want to achieve, not technical implementation details
- Ask questions in everyday language - avoid jargon and technical terms
- Always plan changes first, get approval, then apply - never modify workflows directly
- Use search_tools to discover tools dynamically - let the LLM reason about tool selection

**Creating Workflows:**
1. Understand user intent - what outcome do they want?
2. If unclear, ask 1-2 clarifying questions in plain language (e.g., "What should happen if no emails are found?")
3. Search for relevant tools using search_tools(query, reasoning)
4. Build the workflow step-by-step: add blocks, connect them, validate intent

**Modifying Workflows:**
Follow PLAN → WAIT → APPLY:
- **PLAN**: Call planning tools (add_workflow_block, modify_workflow_block, etc.) - they return a plan with edit_id
- **WAIT**: User approves with "yes", "approve", "apply", etc. - if rejected, create a new plan
- **APPLY**: Call apply_workflow_edit(edit_id) once approved

**Asking Questions:**
- Ask only when essential information is missing - make reasonable assumptions otherwise
- Limit to 1-2 questions at a time to avoid overwhelming users
- Adapt language to user's apparent technical level
- Ask about goals and outcomes, not technical details

**Error Handling:**
- If tools fail, explain the error in plain terms and suggest alternatives
- If workflow state is invalid, identify what's missing and ask for clarification
- When multiple tools could work, choose the simplest that meets the requirement

**Tool Discovery:**
- Use search_tools(query, reasoning) for semantic search by capability
- Use list_available_tools(integration_type) to see all tools, optionally filtered
- Explain why selected tools are appropriate for the user's request

Always think through your reasoning and provide clear explanations for suggestions.{workflow_context}"""

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
    ]
    
    # Verify checkpointer is provided (required for persistence)
    if checkpointer is None:
        logger.warning("No checkpointer provided to create_workflow_chat_agent - traces will not be persisted")
    else:
        logger.debug(f"Creating workflow chat agent with checkpointer: {type(checkpointer).__name__}")
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    
    logger.info(f"Created workflow chat agent with model {model}, checkpointer={'enabled' if checkpointer else 'disabled'}")
    return agent

