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
from .schema import BlockDefinition, BlockType
from .alias_utils import extract_block_alias_info, refresh_workflow_state_aliases
import mlflow
mlflow.langchain.autolog()

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

# Workflow manipulation tools


def _with_block_config_defaults(
    block_type: Optional[str],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ensure required config defaults exist for specific block types."""
    config = config.copy() if config else {}
    if block_type == "for_loop":
        # Loop blocks require array_var/item_var for validation/execution
        config.setdefault("array_var", "items")
        config.setdefault("item_var", "item")
    return config


def _validate_block_config(
    block_type: str,
    config: Dict[str, Any],
    block_id: str = "validation-dummy",
) -> Optional[str]:
    """
    Validate block configuration using BlockDefinition schema.
    
    Args:
        block_type: Block type (e.g., 'tool', 'llm', 'if_else', 'for_loop')
        config: Block configuration dictionary
        block_id: Optional block ID for error messages (defaults to dummy)
        
    Returns:
        Error message string if validation fails, None if valid
    """
    try:
        # Convert block_type string to BlockType enum
        try:
            block_type_enum = BlockType(block_type.lower().replace('_', '_'))
        except ValueError:
            return f"Invalid block type: {block_type}"
        
        # Create a BlockDefinition with dummy position to validate config
        BlockDefinition(
            id=block_id,
            type=block_type_enum,
            config=config,
            position={"x": 0, "y": 0},
        )
        return None
    except ValueError as e:
        # Extract the validation error message
        error_msg = str(e)
        # Make error message more helpful for the agent
        if "tool_name is required" in error_msg:
            return f"Tool blocks require 'tool_name' in config. Please specify which tool to use (e.g., 'gmail_read_emails', 'github_create_issue')."
        elif "user_prompt is required" in error_msg:
            return f"LLM blocks require 'user_prompt' in config. Please provide the prompt text."
        elif "condition is required" in error_msg:
            return f"If/else blocks require 'condition' in config. Please provide a condition expression."
        elif "array_var" in error_msg or "item_var" in error_msg:
            return f"For loop blocks require 'array_var' and 'item_var' in config."
        else:
            return f"Invalid block configuration: {error_msg}"
    except Exception as e:
        logger.warning(f"Unexpected error validating block config: {e}")
        return f"Validation error: {str(e)}"

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
    processed_config = _with_block_config_defaults(block_type, block_config)
    if block_type == "tool" and "inputs" in processed_config and "params" not in processed_config:
        processed_config = {
            **processed_config,
            "params": processed_config.pop("inputs"),
        }
    
    # Validate block configuration before creating the block
    validation_error = _validate_block_config(block_type, processed_config, block_id)
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
        # response_data["aliases"] = alias_info["aliases"]
        # response_data["variable_references"] = alias_info["references"]
        # response_data["hint"] = f"Use {', '.join(alias_info['references'])} to reference this block's output in other blocks."
    
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
        updated_config = _with_block_config_defaults(block_type, modified_block.get("data", {}).get("config"))
        if "data" not in modified_block:
            modified_block["data"] = {}
        modified_block["data"]["config"] = updated_config
        
        # Validate block configuration before applying modification
        validation_error = _validate_block_config(block_type, updated_config, block_id)
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
        # response_data["aliases"] = alias_info["aliases"]
        # response_data["variable_references"] = alias_info["references"]
        # response_data["hint"] = f"Use {', '.join(alias_info['references'])} to reference this block's output in other blocks."
    
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
    
    return json.dumps({
        "op": "remove_node",
        "description": f"Remove block '{block_id}' and {len(connected_edges)} connected edges",
        "node_id": block_id,
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
    
    return json.dumps({
        "op": "add_edge",
        "description": f"Connect '{source_id}' to '{target_id}'",
        "edge_id": new_edge.get("id"),
        "edge": new_edge,
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
    
    return json.dumps({
        "op": "remove_edge",
        "description": f"Remove connection from '{source_id}' to '{target_id}'",
        "edge": {
            "source": source_id,
            "target": target_id,
        },
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
            top_k=3
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
                "required_scopes": tool_meta.get("required_scopes", [])
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
        # Dynamic tool discovery tools
        search_tools,
        # list_available_tools,
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
    template_hint_section = ""
    if workflow_state:
        workflow_context = f"\n\nCurrent workflow state:\n{json.dumps(workflow_state, indent=2)}\n\nUse this information when calling tools. Tools automatically access workflow state from thread context via runtime configuration."
        alias_examples = workflow_state.get("template_reference_examples") or {}
        if alias_examples:
            alias_lines = []
            for block_id, examples in alias_examples.items():
                if not examples:
                    continue
                alias_lines.append(f"- {block_id}: {', '.join(examples)}")
            if alias_lines:
                template_hint_section = "\nTemplate reference hints (use these names when writing {{alias.output}} expressions):\n" + "\n".join(alias_lines)
    
    system_prompt = f"""You are an intelligent workflow assistant that helps users build and edit workflows. Your role is to understand user intent, discover appropriate tools, and create workflows that achieve their goals.
**Core Principles:**
- Focus on what users want to achieve, not technical implementation details
- Ask questions in everyday language - avoid jargon and technical terms
- Always plan changes first, get approval, then apply - never modify workflows directly
- Use search_tools to discover tools dynamically - let the LLM reason about tool selection
- When referencing other blocks, use the alias hints returned by add_workflow_block/modify_workflow_block tools
- Each tool response includes "variable_references" showing the exact {{alias.output}} format to use
- Always use the aliases from tool responses - never invent template variable names
- Remember alias names are always between double curly braces like {{alias.output}} 
**Creating Workflows:**
1. Understand user intent - what outcome do they want?
2. If unclear, ask 1-2 clarifying questions in plain language (e.g., "What should happen if no emails are found?")
3. Search for relevant tools using search_tools(query, reasoning)
4. Build the workflow step-by-step: add blocks, connect them via tool add_workflow_edge, validate intent
5. remember to connect blocks to each other via tool add_workflow_edge after you have added all blocks
**Modifying Workflows:**
- Call planning tools (add_workflow_block, modify_workflow_block, etc.) to build a complete proposal
- Each tool call must describe the change as a patch operation (add_node, update_node, remove_node, add_edge, remove_edge)
- Never apply edits yourself; simply describe the full change-set so the user can accept or reject it
- Include a concise natural-language justification for the proposal
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
Always think through your reasoning and provide clear explanations for suggestions.{template_hint_section}{workflow_context}"""

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

