"""
Node functions for workflow blocks.

Each block type has a corresponding node function that executes the block
and updates the workflow state.
"""
from typing import Any, Dict, Optional
from datetime import datetime

from shared.logger import get_logger
from shared.llm import get_llm
from shared.tools.executor import execute_tool as execute_tool_with_oauth
from langchain_core.messages import HumanMessage, SystemMessage

from .state import WorkflowState
from .schema import BlockDefinition, BlockType
from .code_executor import execute_code_block, CodeExecutionError
from .models import WorkflowExecution, BlockExecution, WorkflowBlock
from shared.database.models import User

logger = get_logger("api.workflows.nodes")


def build_variable_map(state: WorkflowState) -> Dict[str, Any]:
    """
    Build a comprehensive variable map from workflow state for template resolution.
    
    Extracts variables from:
    1. input_data (direct input)
    2. All block outputs (including tool outputs with structured data)
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary mapping variable names to values
    """
    input_data = state.get("input_data", {})
    block_outputs = state.get("block_outputs", {})
    
    variable_map = {}
    
    # 1. Add variables from input_data (direct input)
    # Flatten nested structures if needed
    for key, value in input_data.items():
        if isinstance(value, dict):
            # If it's a dict, merge its keys into variable_map
            variable_map.update(value)
        else:
            variable_map[key] = value
    
    # 2. Add variables from all block outputs
    # For each block, extract all output handles
    for block_id, block_output in block_outputs.items():
        if isinstance(block_output, dict):
            # Extract all keys from block_output as variables
            # This includes both "output" and any structured output keys
            for key, value in block_output.items():
                # Use block_id.key as the variable name for explicit reference
                # Also add just the key if it's unique
                var_name_with_block = f"{block_id}.{key}"
                variable_map[var_name_with_block] = value
                
                # Add simple key if not already present (allows {{key}} if unique)
                if key not in variable_map:
                    variable_map[key] = value
                elif isinstance(value, dict) and isinstance(variable_map.get(key), dict):
                    # If both are dicts, merge them
                    variable_map[key].update(value)
    
    return variable_map


def resolve_template_variables(text: str, variable_map: Dict[str, Any]) -> str:
    """
    Resolve {{variable_name}} template variables in text.
    
    Supports:
    - Simple names: {{email}} - resolves to variable_map["email"]
    - Dot notation: {{block_id.handle_id}} - resolves to variable_map["block_id.handle_id"]
    - Falls back to empty string if variable not found
    
    Args:
        text: Text containing template variables
        variable_map: Dictionary mapping variable names to values
        
    Returns:
        Text with template variables resolved
    """
    import re
    
    if not text or "{{" not in text:
        return text
    
    def replace_template_var(match):
        var_name = match.group(1)
        # Try variable_map first, then fallback to empty string
        value = variable_map.get(var_name, "")
        logger.debug(f"Resolving {{{{{var_name}}}}}: {value}")
        return str(value) if value is not None else ""
    
    resolved = re.sub(r'\{\{(\w+(?:\.\w+)?)\}\}', replace_template_var, text)
    logger.debug(f"Resolved template: {text[:50]}... -> {resolved[:50]}...")
    return resolved


async def resolve_inputs(
    state: WorkflowState,
    input_resolution: Dict[str, Dict[str, Any]],
    block: BlockDefinition,
) -> Dict[str, Any]:
    """
    Resolve inputs for a block using hybrid approach:
    1. Explicit connections (via edges)
    2. Global state references (via block.config.input_refs)
    
    Args:
        state: Current workflow state
        input_resolution: Pre-computed input resolution map (handle_id -> ref_info)
        block: Block definition
    
    Returns:
        Dictionary of resolved input values
    """
    resolved_inputs = {}
    
    # 1. Resolve explicit connections
    for handle_id, ref_info in input_resolution.items():
        source_block_id = ref_info["source_block"]
        source_handle = ref_info["source_handle"]
        
        # Get from global state
        block_output = state["block_outputs"].get(source_block_id, {})
        resolved_inputs[handle_id] = block_output.get(source_handle)
    
    # 2. Resolve global state references from block.config.input_refs
    if "input_refs" in block.config:
        for handle_id, ref in block.config["input_refs"].items():
            # Parse reference: "block_a.email" or "block_a.output"
            if "." in ref:
                source_block_id, source_handle = ref.split(".", 1)
                block_output = state["block_outputs"].get(source_block_id, {})
                resolved_inputs[handle_id] = block_output.get(source_handle)
    
    return resolved_inputs


async def input_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """Input block: provides workflow input data."""
    # Get input_data from state
    input_data = state.get("input_data", {})
    
    # Handle fields array (new format)
    fields = block.config.get("fields", [])
    if isinstance(fields, list) and len(fields) > 0:
        # Extract field values based on field names
        output = {}
        for field in fields:
            if isinstance(field, dict) and "name" in field:
                field_name = field["name"]
                # Get value from input_data, use field name as key
                if field_name in input_data:
                    output[field_name] = input_data[field_name]
                # Also support accessing via block label if available
                block_label = block.config.get("label", block.id)
                if block_label in input_data and isinstance(input_data[block_label], dict):
                    if field_name in input_data[block_label]:
                        output[field_name] = input_data[block_label][field_name]
        # If no fields matched, return empty dict (or all input_data for backward compatibility)
        if not output:
            output = input_data
    # Handle variable_name (legacy format)
    elif block.config.get("variable_name"):
        variable_name = block.config.get("variable_name")
        if variable_name in input_data:
            # Return the specific variable value
            output = {variable_name: input_data[variable_name]}
        else:
            # Return all input_data for backward compatibility
            output = input_data
    else:
        # Return all input_data (for backward compatibility)
        output = input_data
    
    return {
        "block_outputs": {
            block.id: {"output": output}
        }
    }


async def code_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """Code block: execute Python code."""
    code = block.python_code or ""
    inputs = await resolve_inputs(state, input_resolution, block)
    
    # Prepare code context
    code_context = {
        **inputs,
        "_inputs": inputs,
    }
    
    result = await execute_code_block(code, code_context)
    
    if result.get("error"):
        raise CodeExecutionError(result["error"])
    
    output = result.get("output")
    
    return {
        "block_outputs": {
            block.id: {"output": output}
        }
    }


async def tool_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    user_id: Optional[str] = None,
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """Tool block: execute external tool."""
    tool_name = block.config.get("tool_name")
    connection_id = block.config.get("connection_id")
    
    # Get tool schema to determine parameters
    from shared.tools.base import get_tool
    try:
        tool = get_tool(tool_name)
        tool_schema = tool.get_parameters_schema()
    except Exception as e:
        raise ValueError(f"Tool '{tool_name}' not found: {e}")
    
    # Resolve inputs for each parameter
    inputs = await resolve_inputs(state, input_resolution, block)
    tool_params = {}
    
    # Build variable map for template resolution
    variable_map = build_variable_map(state)
    
    # Get parameter names from schema
    param_properties = tool_schema.get("properties", {})
    
    for param_name in param_properties.keys():
        param_value = None
        param_schema = param_properties.get(param_name, {})
        
        # Check if there's an explicit connection for this parameter
        if param_name in inputs:
            param_value = inputs[param_name]
        # Otherwise, check block.config.params for default/static values
        elif param_name in block.config.get("params", {}):
            param_value = block.config["params"][param_name]
        # Use schema default if available
        elif "default" in param_schema:
            param_value = param_schema["default"]
        
        # Resolve template variables in parameter value if it's a string
        if param_value is not None:
            if isinstance(param_value, str) and "{{" in param_value:
                param_value = resolve_template_variables(param_value, variable_map)
            tool_params[param_name] = param_value
    
    # Execute tool with OAuth token management

    user = await User.get(user_id=user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    
    try:
        result = await execute_tool_with_oauth(
            tool_name=tool_name,
            user=user,
            connection_id=connection_id,
            arguments=tool_params
        )
    except Exception as e:
        raise ValueError(f"Tool execution failed: {str(e)}")
    
    # Store output with parameter-level granularity if structured
    output_data = {"output": result}
    if isinstance(result, dict):
        output_data.update(result)
    
    return {
        "block_outputs": {
            block.id: output_data
        }
    }


async def llm_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """LLM block: execute LLM call with optional structured output."""
    import json
    import re
    
    system_prompt = block.config.get("system_prompt", "")
    model = block.config.get("model", "gpt-5-mini")
    temperature = block.config.get("temperature", 0.2)
    output_schema = block.config.get("output_schema")  # Optional JSON schema for structured output
    
    # Resolve template variables in system prompt (e.g., {{variable_name}})
    variable_map = build_variable_map(state)
    system_prompt = resolve_template_variables(system_prompt, variable_map)
    
    inputs = await resolve_inputs(state, input_resolution, block)
    user_message = inputs.get("input", "")
    
    llm = get_llm(model=model, temperature=temperature)
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=str(user_message)))
    
    # If output_schema provided, use structured output
    if output_schema:
        logger.info(f"Running LLM with structured output schema")
        try:
            structured_llm = llm.with_structured_output(
                output_schema,
                method="json_schema"
            )
            result = await structured_llm.ainvoke(messages)
            
            # Convert result to dict if it's a Pydantic model
            if hasattr(result, "model_dump"):
                structured_output = result.model_dump()
            elif isinstance(result, dict):
                structured_output = result
            else:
                structured_output = {"result": str(result)}
            
            return {
                "block_outputs": {
                    block.id: {
                        "output": json.dumps(structured_output, indent=2),
                        "structured_output": structured_output
                    }
                }
            }
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to text: {e}")
            # Fall back to regular text generation
            response = await llm.ainvoke(messages)
            output = response.content
            return {
                "block_outputs": {
                    block.id: {"output": output}
                }
            }
    else:
        # Regular text generation
        response = await llm.ainvoke(messages)
        output = response.content
        
        return {
            "block_outputs": {
                block.id: {"output": output}
            }
        }


async def if_else_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """If/Else block: evaluate condition and set routing flag."""
    condition = block.config.get("condition", "")
    inputs = await resolve_inputs(state, input_resolution, block)
    
    # Evaluate condition in context
    eval_context = {
        **inputs,
    }
    
    condition_code = f"_result = {condition}"
    result = await execute_code_block(condition_code, eval_context)
    
    if result.get("error"):
        raise CodeExecutionError(f"Condition evaluation error: {result['error']}")
    
    condition_result = result.get("output")
    is_true = bool(condition_result)
    
    # Store routing information in state for conditional edges
    return {
        "block_outputs": {
            block.id: {
                "output": condition_result,
                "condition_result": is_true,
                "route": "true" if is_true else "false",
            }
        }
    }


async def for_loop_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    execution: Optional[WorkflowExecution] = None,
) -> WorkflowState:
    """For loop block: prepare loop state for iteration."""
    array_var = block.config.get("array_var")
    item_var = block.config.get("item_var", "item")
    
    inputs = await resolve_inputs(state, input_resolution, block)
    
    # Get array from inputs
    array = inputs.get(array_var) or inputs.get("input", [])
    
    if not isinstance(array, (list, tuple)):
        raise ValueError(f"For loop array '{array_var}' is not iterable")
    
    # Store loop state
    loop_state = {
        "array_var": array_var,
        "item_var": item_var,
        "current_index": 0,
        "items": list(array),
        "results": [],
    }
    
    return {
        "loop_state": loop_state,
        "block_outputs": {
            block.id: {
                "output": {"items": list(array), "count": len(array)}
            }
        }
    }


# Node function registry
NODE_FUNCTIONS = {
    BlockType.INPUT: input_node,
    BlockType.CODE: code_node,
    BlockType.TOOL: tool_node,
    BlockType.LLM: llm_node,
    BlockType.IF_ELSE: if_else_node,
    BlockType.FOR_LOOP: for_loop_node,
}


def get_node_function(block_type: BlockType):
    """Get node function for a block type."""
    return NODE_FUNCTIONS.get(block_type)


__all__ = [
    "resolve_inputs",
    "build_variable_map",
    "resolve_template_variables",
    "input_node",
    "code_node",
    "tool_node",
    "llm_node",
    "if_else_node",
    "for_loop_node",
    "get_node_function",
    "NODE_FUNCTIONS",
]

