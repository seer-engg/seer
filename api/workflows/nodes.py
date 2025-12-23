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

logger = get_logger("api.workflows.nodes")


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
    
    # If this input block has a variable_name config, extract that specific variable
    variable_name = block.config.get("variable_name")
    if variable_name and variable_name in input_data:
        # Return the specific variable value
        output = {variable_name: input_data[variable_name]}
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
    
    # Get parameter names from schema
    param_properties = tool_schema.get("properties", {})
    
    for param_name in param_properties.keys():
        # Check if there's an explicit connection for this parameter
        if param_name in inputs:
            tool_params[param_name] = inputs[param_name]
        # Otherwise, check block.config.params for default/static values
        elif param_name in block.config.get("params", {}):
            tool_params[param_name] = block.config["params"][param_name]
        # Check if there's a generic "input" handle
        elif "input" in inputs and param_name not in tool_params:
            # Use generic input if available
            tool_params[param_name] = inputs["input"]
    
    # Execute tool with OAuth token management
    try:
        result = await execute_tool_with_oauth(
            tool_name=tool_name,
            user_id=user_id,
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
    # First, collect all available variables from input_data and input block outputs
    input_data = state.get("input_data", {})
    block_outputs = state.get("block_outputs", {})
    
    # Build a comprehensive variable map
    variable_map = {}
    
    # 1. Add variables from input_data (direct input)
    # Flatten nested structures if needed
    for key, value in input_data.items():
        if isinstance(value, dict):
            # If it's a dict, merge its keys into variable_map
            variable_map.update(value)
        else:
            variable_map[key] = value
    
    # 2. Add variables from input block outputs
    # Input blocks store their output under block_outputs[block_id].output
    for block_id, block_output in block_outputs.items():
        if isinstance(block_output, dict) and "output" in block_output:
            output_value = block_output["output"]
            if isinstance(output_value, dict):
                # Merge dict output directly
                variable_map.update(output_value)
            elif output_value is not None:
                # If it's a single value, try to use block_id or find variable_name
                # For now, we'll skip non-dict outputs here as they're handled by input_data
                pass
    
    # Log for debugging
    logger.debug(f"Template variable map: {variable_map}")
    
    # Resolve template variables
    if system_prompt and "{{" in system_prompt:
        def replace_template_var(match):
            var_name = match.group(1)
            # Try variable_map first, then fallback to empty string
            value = variable_map.get(var_name, "")
            logger.debug(f"Resolving {{{{{var_name}}}}}: {value}")
            return str(value) if value is not None else ""
        system_prompt = re.sub(r'\{\{(\w+)\}\}', replace_template_var, system_prompt)
        logger.debug(f"Resolved system prompt: {system_prompt}")
    
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
    "input_node",
    "code_node",
    "tool_node",
    "llm_node",
    "if_else_node",
    "for_loop_node",
    "get_node_function",
    "NODE_FUNCTIONS",
]

