from typing import Optional, Dict, Any
import json
from langchain_core.tools import tool
from agents.workflow_agent.context import (
    get_workflow_state_for_thread,
    set_workflow_state_for_thread,
    _current_thread_id,
    append_patch_op_for_thread,
)
import uuid
from shared.logger import get_logger
from workflow_core.validation import with_block_config_defaults, validate_block_config
from workflow_core.alias_utils import extract_block_alias_info, refresh_workflow_state_aliases

logger = get_logger(__name__)


def _record_patch_op(thread_id: Optional[str], patch_op: Dict[str, Any]) -> None:
    """Record a tool patch op so the router can build proposals without log parsing."""
    if not thread_id or not isinstance(patch_op, dict):
        return
    if "op" not in patch_op:
        return
    append_patch_op_for_thread(thread_id, patch_op)


def _normalize_tool_block_config(block_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Rename legacy tool config keys to the canonical schema."""
    config = block_config.copy() if block_config else {}
    legacy_params = config.pop("tool_params", None)
    if legacy_params is not None and "params" not in config:
        config["params"] = legacy_params
    legacy_inputs = config.pop("inputs", None)
    if legacy_inputs is not None and "params" not in config:
        config["params"] = legacy_inputs
    return config


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
    
    if workflow_state.get("block_aliases"):
        analysis["block_aliases"] = workflow_state["block_aliases"]
    if workflow_state.get("template_reference_examples"):
        analysis["template_reference_examples"] = workflow_state["template_reference_examples"]
    if workflow_state.get("input_variables"):
        analysis["input_variables"] = workflow_state["input_variables"]
    
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
        block_type: Type of block to add (e.g., 'tool', 'llm', 'if_else', 'for_loop', 'input')
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
    normalized_config = block_config.copy() if block_config else {}
    if block_type == "tool":
        normalized_config = _normalize_tool_block_config(normalized_config)
    processed_config = with_block_config_defaults(block_type, normalized_config)
    
    # Validate block configuration before creating the block
    validation_error = validate_block_config(block_type, processed_config, block_id)
    if validation_error:
        return json.dumps({
            "error": validation_error,
            "block_type": block_type,
            "block_id": block_id,
            "suggestion": "Please provide the required configuration fields. For tool blocks, you must specify 'tool_name' (e.g., use search_tools to find available tools)."
        })
    
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
    
    # Mutate workflow_state contextually so subsequent tool calls can reference the block
    nodes.append(new_block)
    workflow_state["nodes"] = nodes
    
    # Extract alias info for the new block
    existing_aliases = set()
    for existing_node in nodes:
        if existing_node.get("id") != block_id:
            existing_info = extract_block_alias_info(existing_node, existing_aliases)
            if existing_info.get("aliases"):
                existing_aliases.update(existing_info["aliases"])
    
    alias_info = extract_block_alias_info(new_block, existing_aliases)
    
    # Refresh workflow_state alias maps to include the new block
    refresh_workflow_state_aliases(workflow_state)
    
    thread_id = _current_thread_id.get()
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    response_data = {
        "op": "add_node",
        "description": f"Add {block_type} block '{block_id}'",
        "node_id": block_id,
        "node": new_block,
    }
    
    # Include alias info so agent can immediately use it in prompts
    if alias_info.get("alias"):
        response_data["alias"] = alias_info["alias"]
    
    _record_patch_op(thread_id, response_data)
    return json.dumps(response_data)


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
        if modified_block.get("type") == "tool":
            current_config = _normalize_tool_block_config(current_config)
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["config"] = current_config
    
    if label:
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["label"] = label
    
    if position:
        modified_block["position"] = position
    
    # Ensure config defaults (important for validation)
    block_type = modified_block.get("type")
    if block_type:
        updated_config = with_block_config_defaults(block_type, modified_block.get("data", {}).get("config"))
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["config"] = updated_config
        
        # Validate block configuration before applying modification
        validation_error = validate_block_config(block_type, updated_config, block_id)
        if validation_error:
            return json.dumps({
                "error": validation_error,
                "block_type": block_type,
                "block_id": block_id,
                "suggestion": "Please provide the required configuration fields."
            })
    
    # Update workflow_state so further reasoning uses modified block
    for idx, node in enumerate(nodes):
        if node.get("id") == block_id:
            nodes[idx] = modified_block
            break
    workflow_state["nodes"] = nodes
    
    # Extract alias info for the modified block (may have changed if label/tool_name changed)
    existing_aliases = set()
    for existing_node in nodes:
        if existing_node.get("id") != block_id:
            existing_info = extract_block_alias_info(existing_node, existing_aliases)
            if existing_info.get("aliases"):
                existing_aliases.update(existing_info["aliases"])
    
    alias_info = extract_block_alias_info(modified_block, existing_aliases)
    
    # Refresh workflow_state alias maps to reflect any changes
    refresh_workflow_state_aliases(workflow_state)
    
    thread_id = _current_thread_id.get()
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    response_data = {
        "op": "update_node",
        "description": f"Modify block '{block_id}'",
        "node_id": block_id,
        "node": modified_block,
    }
    
    # Include updated alias info
    if alias_info.get("alias"):
        response_data["alias"] = alias_info["alias"]
    
    _record_patch_op(thread_id, response_data)
    return json.dumps(response_data)


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
    
    nodes = workflow_state.get("nodes", [])
    edges = workflow_state.get("edges", [])
    
    # Check if block exists
    block_exists = any(node.get("id") == block_id for node in nodes)
    if not block_exists:
        return json.dumps({
            "error": f"Block with ID '{block_id}' not found"
        })
    
    connected_edges = [
        edge for edge in edges
        if edge.get("source") == block_id or edge.get("target") == block_id
    ]
    
    # Mutate workflow_state so subsequent tool calls respect the removal
    workflow_state["nodes"] = [node for node in nodes if node.get("id") != block_id]
    workflow_state["edges"] = [
        edge for edge in edges
        if edge.get("source") != block_id and edge.get("target") != block_id
    ]
    
    # Refresh workflow_state alias maps after removal
    refresh_workflow_state_aliases(workflow_state)
    
    thread_id = _current_thread_id.get()
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    response_data = {
        "op": "remove_node",
        "description": f"Remove block '{block_id}' and {len(connected_edges)} connected edges",
        "node_id": block_id,
    }
    
    _record_patch_op(thread_id, response_data)
    return json.dumps(response_data)


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
    
    # Update workflow_state
    edges.append(new_edge)
    workflow_state["edges"] = edges
    thread_id = _current_thread_id.get()
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    response_data = {
        "op": "add_edge",
        "description": f"Connect '{source_id}' to '{target_id}'",
        "edge_id": new_edge.get("id"),
        "edge": new_edge,
    }
    
    _record_patch_op(thread_id, response_data)
    return json.dumps(response_data)


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
    
    workflow_state["edges"] = [
        edge for edge in edges
        if not (edge.get("source") == source_id and edge.get("target") == target_id)
    ]
    thread_id = _current_thread_id.get()
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    response_data = {
        "op": "remove_edge",
        "description": f"Remove connection from '{source_id}' to '{target_id}'",
        "edge": {
            "source": source_id,
            "target": target_id,
        },
    }
    
    _record_patch_op(thread_id, response_data)
    return json.dumps(response_data)

