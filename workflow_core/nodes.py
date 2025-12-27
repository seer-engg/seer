"""
Node functions for workflow blocks.

Each block type has a corresponding node function that executes the block
and updates the workflow state.
"""
from typing import Any, Dict, Optional, List, Union

from shared.logger import get_logger
from shared.llm import  get_llm_without_responses_api
from shared.tools.executor import execute_tool as execute_tool_with_oauth
from langchain_core.messages import HumanMessage, SystemMessage

from .state import WorkflowState
from .schema import BlockDefinition, BlockType
from .code_executor import execute_code_block, CodeExecutionError
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
    block_aliases = state.get("block_aliases", {})
    
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
                
                # Register alias-based names (e.g., {{gmail_read_tool.output}})
                aliases = block_aliases.get(block_id, [])
                for alias in aliases:
                    alias_key = f"{alias}.{key}"
                    if alias_key not in variable_map:
                        variable_map[alias_key] = value
                
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
    - Deep dot notation: {{block_id.structured_output.field}} - resolves nested properties
    - Falls back to empty string if variable not found
    
    Args:
        text: Text containing template variables
        variable_map: Dictionary mapping variable names to values
        
    Returns:
        Text with template variables resolved
    """
    import re
    import json

    logger.info(f"Resolving template variables: {text[:50]}...")
    logger.info(f"Variable map: {variable_map.keys()}...")
    
    if not text or "{{" not in text:
        return text
    
    def resolve_nested_value(var_name: str) -> Any:
        """Resolve a variable name, supporting nested dot notation."""
        # First try exact match in variable_map
        if var_name in variable_map:
            return variable_map[var_name]
        
        parts = var_name.split(".")
        if len(parts) < 2:
            return ""
        
        block_id = parts[0]
        first_handle = parts[1]
        remaining_path = parts[2:]
        
        def get_handle_value(handle_name: str) -> Any:
            return variable_map.get(f"{block_id}.{handle_name}")
        
        # Prefer explicit handle references such as {{alias.item.*}}
        current_value = get_handle_value(first_handle)
        if current_value is None:
            # Fall back to default output/structured_output handles
            remaining_path = parts[1:]
            current_value = get_handle_value("output")
            if current_value is not None and remaining_path and remaining_path[0] == "output":
                remaining_path = remaining_path[1:]
            elif current_value is None:
                current_value = get_handle_value("structured_output")
                if current_value is not None and remaining_path and remaining_path[0] == "structured_output":
                    remaining_path = remaining_path[1:]
        
        if current_value is None:
            return ""
        
        # If we still have a nested path to resolve, try to coerce strings
        # containing JSON or reuse structured_output for legacy {{block.output.*}} refs.
        if remaining_path:
            parsed_value = None
            if isinstance(current_value, str):
                try:
                    parsed_value = json.loads(current_value)
                except Exception:
                    parsed_value = None
            if parsed_value is not None:
                current_value = parsed_value
            elif first_handle == "output":
                structured_value = get_handle_value("structured_output")
                if structured_value is not None:
                    current_value = structured_value
        
        # Traverse any remaining nested keys (dicts or list indexes)
        for key in remaining_path:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            elif isinstance(current_value, list):
                try:
                    index = int(key)
                    current_value = current_value[index]
                except (ValueError, IndexError):
                    return ""
            else:
                return ""
        
        return current_value
    
    def replace_template_var(match):
        var_name = match.group(1)
        value = resolve_nested_value(var_name)
        logger.debug(f"Resolving {{{{{var_name}}}}}: {value}")
        # Convert dict/list to JSON string for readability
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value) if value is not None else ""
    
    # Updated regex to support multiple levels of dot notation
    resolved = re.sub(r'\{\{(\w+(?:\.\w+)*)\}\}', replace_template_var, text)
    logger.debug(f"Resolved template: {text[:50]}... -> {resolved[:50]}...")
    return resolved


def resolve_template_payload(payload: Any, variable_map: Dict[str, Any]) -> Any:
    """
    Resolve template variables within arbitrary payloads.
    
    Supports nested lists/dicts so tool parameters like ["{{node.handle}}"]
    are resolved just like plain strings.
    """
    if isinstance(payload, str):
        if "{{" in payload:
            return resolve_template_variables(payload, variable_map)
        return payload
    if isinstance(payload, list):
        return [resolve_template_payload(item, variable_map) for item in payload]
    if isinstance(payload, dict):
        return {
            key: resolve_template_payload(value, variable_map)
            for key, value in payload.items()
        }
    return payload


async def resolve_inputs(
    state: WorkflowState,
    input_resolution: Dict[str, Dict[str, Any]],
    block: BlockDefinition,
) -> Dict[str, Any]:
    """
    Resolve inputs for a block using global state references.
    
    Args:
        state: Current workflow state
        input_resolution: Pre-computed input resolution map (handle_id -> ref_info)
        block: Block definition
    
    Returns:
        Dictionary of resolved input values
    """
    resolved_inputs = {}
    
    # Legacy explicit connections have been deprecated in favor of variable references.
    # Resolve global state references from block.config.input_refs
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
            if isinstance(field, dict):
                # Support both 'id' and 'name' fields (id is preferred)
                field_name = field.get("id") or field.get("name")
                if field_name:
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


async def variable_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
) -> WorkflowState:
    """Variable block: expose a literal value (string, number, or array) for reuse."""
    config = block.config or {}
    resolved_inputs = await resolve_inputs(state, input_resolution, block)
    value = resolved_inputs.get("input") if resolved_inputs else None
    if value is None:
        value = config.get("input")
    if value is None:
        raise ValueError(f"Variable block '{block.id}' requires an 'input' value.")

    input_type = str(config.get("input_type") or "string").lower()
    variable_map = build_variable_map(state)

    def coerce_number(raw: Any) -> Union[float, int]:
        resolved = resolve_template_payload(raw, variable_map)
        if isinstance(resolved, (int, float)):
            return resolved
        if resolved is None:
            raise ValueError(f"Variable block '{block.id}' input resolved to None, expected a number.")
        text = str(resolved).strip()
        if not text:
            raise ValueError(f"Variable block '{block.id}' input resolved to empty string, expected a number.")
        try:
            if any(char in text.lower() for char in ("e", ".")):
                return float(text)
            return int(text)
        except ValueError as exc:
            raise ValueError(f"Variable block '{block.id}' could not convert '{text}' to a number.") from exc

    if input_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"Variable block '{block.id}' expected a list when input_type='array'.")
        resolved_value = resolve_template_payload(value, variable_map)
    elif input_type == "number":
        resolved_value = coerce_number(value)
    else:
        string_value = value if isinstance(value, str) else str(value)
        resolved_value = resolve_template_payload(string_value, variable_map)

    return {
        "block_outputs": {
            block.id: {"output": resolved_value}
        }
    }


async def tool_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
    user_id: Optional[str] = None,
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
            resolved_param = resolve_template_payload(param_value, variable_map)
            tool_params[param_name] = resolved_param
    
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


def _prepare_structured_output_schema(schema: Dict[str, Any], block_id: str) -> Dict[str, Any]:
    """
    Prepare a JSON schema for OpenAI's structured output.
    
    OpenAI's structured output requires:
    - additionalProperties: false for strict mode
    - required array with all property names
    - A title/name for the schema
    
    Args:
        schema: Raw JSON schema from block config
        block_id: Block ID for generating unique schema name
        
    Returns:
        Properly formatted JSON schema for OpenAI
    """
    import copy
    
    if not schema or not isinstance(schema, dict):
        return schema
    
    # Create a deep copy to avoid mutating original
    prepared_schema = copy.deepcopy(schema)
    
    # Ensure it's an object type
    if prepared_schema.get("type") != "object":
        prepared_schema["type"] = "object"
    
    # Add title if not present (required by OpenAI for schema identification)
    if "title" not in prepared_schema:
        # Create a clean title from block_id
        clean_id = "".join(c if c.isalnum() else "_" for c in block_id)
        prepared_schema["title"] = f"Output_{clean_id}"
    
    # Add additionalProperties: false for strict mode
    if "additionalProperties" not in prepared_schema:
        prepared_schema["additionalProperties"] = False
    
    # Build required array from all properties if not already present
    properties = prepared_schema.get("properties", {})
    if properties and "required" not in prepared_schema:
        prepared_schema["required"] = list(properties.keys())
    
    # Ensure each property has a valid type and preserve description
    for prop_name, prop_def in properties.items():
        if isinstance(prop_def, dict):
            if "type" not in prop_def:
                prop_def["type"] = "string"  # Default to string if type is missing
            # Description is already preserved from the schema, no action needed
    
    return prepared_schema


def _extract_text_from_response(content: Any) -> str:
    """
    Extract text content from LLM response, handling both standard and Responses API formats.
    
    Args:
        content: Response content (could be string, list, or other format)
        
    Returns:
        Extracted text content as string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Responses API format: list of content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts) if text_parts else str(content)
    else:
        return str(content) if content else ""


async def llm_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
) -> WorkflowState:
    """LLM block: execute LLM call with optional structured output."""
    import json
    import re
    
    system_prompt = block.config.get("system_prompt", "")
    user_prompt = block.config.get("user_prompt", "")
    model = block.config.get("model", "gpt-5-mini")
    temperature = block.config.get("temperature", 0.2)
    output_schema = block.config.get("output_schema")  # Optional JSON schema for structured output
    
    # Validate that user_prompt is provided and not empty
    if not user_prompt or not user_prompt.strip():
        raise ValueError(f"LLM block '{block.id}' requires a non-empty 'user_prompt' in config")
    
    logger.info(f"LLM node executing with model={model}, has_output_schema={output_schema is not None}")
    
    # Resolve template variables in system prompt and user prompt (e.g., {{variable_name}})
    variable_map = build_variable_map(state)
    system_prompt = resolve_template_variables(system_prompt, variable_map)
    user_prompt = resolve_template_variables(user_prompt, variable_map)
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    
    logger.debug(f"LLM messages: system_prompt={system_prompt[:100] if system_prompt else 'None'}..., user_prompt={user_prompt[:100] if user_prompt else 'None'}...")
    
    # If output_schema provided, use structured output
    if output_schema and isinstance(output_schema, dict) and output_schema.get("properties"):
        logger.info(f"Running LLM with structured output schema: {output_schema}")
        
        # Prepare schema for OpenAI's structured output requirements
        prepared_schema = _prepare_structured_output_schema(output_schema, block.id)
        logger.info(f"Prepared schema for structured output: {prepared_schema}")
        
        try:
            # Use LLM without responses API for structured output
            # The responses API doesn't work correctly with with_structured_output
            llm_for_structured = get_llm_without_responses_api(model=model, temperature=temperature)
            logger.debug(f"Created LLM for structured output (without responses API)")
            
            structured_llm = llm_for_structured.with_structured_output(
                prepared_schema,
                method="json_schema"
            )
            logger.debug(f"Calling structured LLM...")
            result = await structured_llm.ainvoke(messages)
            logger.debug(f"Structured LLM result type: {type(result)}, value: {result}")
            
            # Convert result to dict if it's a Pydantic model
            if hasattr(result, "model_dump"):
                structured_output = result.model_dump()
            elif isinstance(result, dict):
                structured_output = result
            else:
                # Try to parse as JSON string
                try:
                    structured_output = json.loads(str(result))
                except (json.JSONDecodeError, TypeError):
                    structured_output = {"result": str(result)}
            
            logger.info(f"Structured output result: {structured_output}")
            
            # Return both the JSON string output and the structured output dict
            # Also include individual field values at the top level for easy access
            block_output = {
                "output": json.dumps(structured_output, indent=2),
                "structured_output": structured_output,
            }
            # Add each field from structured output to block output for easy variable access
            block_output.update(structured_output)
            
            return {
                "block_outputs": {
                    block.id: block_output
                }
            }
        except Exception as e:
            logger.error(f"Structured output failed: {e}", exc_info=True)
            # Fall back to regular text generation with error context
            # Use non-responses API for consistent output format
            llm = get_llm_without_responses_api(model=model, temperature=temperature)
            response = await llm.ainvoke(messages)
            output = _extract_text_from_response(response.content)
            return {
                "block_outputs": {
                    block.id: {
                        "output": output,
                        "structured_output_error": str(e)
                    }
                }
            }
    else:
        # Regular text generation - use non-responses API for consistent output format
        llm = get_llm_without_responses_api(model=model, temperature=temperature)
        response = await llm.ainvoke(messages)
        output = _extract_text_from_response(response.content)
        
        return {
            "block_outputs": {
                block.id: {"output": output}
            }
        }


async def if_else_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
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


def _resolve_loop_items(config: Dict[str, Any], state: WorkflowState) -> List[Any]:
    """Resolve the iterable for a for_loop block based on config."""
    import json
    
    array_mode = (config.get("array_mode") or "variable").lower()
    variable_map = build_variable_map(state)
    
    if array_mode == "literal":
        literal_items = config.get("array_literal", [])
        if literal_items is None:
            raise ValueError("array_literal must be provided when array_mode is 'literal'")
        if not isinstance(literal_items, list):
            raise ValueError("array_literal must be a list when array_mode is 'literal'")
        return list(literal_items)
    
    array_var = config.get("array_variable") or config.get("array_var")
    if not array_var:
        raise ValueError("array_variable is required when array_mode is 'variable'")
    
    normalized_var = array_var.strip()
    if normalized_var.startswith("{{") and normalized_var.endswith("}}"):
        normalized_var = normalized_var[2:-2].strip()
    
    array_value = variable_map.get(normalized_var)
    if array_value is None:
        raise ValueError(f"for_loop could not resolve variable '{normalized_var}'")
    
    # Attempt to parse JSON arrays stored as strings
    if isinstance(array_value, str):
        trimmed = array_value.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                array_value = json.loads(trimmed)
            except json.JSONDecodeError:
                pass
    
    if not isinstance(array_value, (list, tuple)):
        raise ValueError(f"for_loop expected '{normalized_var}' to be a list/array but received {type(array_value).__name__}")
    
    return list(array_value)


async def for_loop_node(
    state: WorkflowState,
    block: BlockDefinition,
    input_resolution: Dict[str, Dict[str, Any]],
) -> WorkflowState:
    """For loop block: iterate through items and expose the current item per execution."""
    config = block.config or {}
    item_var = (config.get("item_var") or "item").strip() or "item"
    
    loop_states = dict(state.get("loop_state") or {})
    loop_ctx = loop_states.get(block.id)
    
    if loop_ctx is None:
        items = _resolve_loop_items(config, state)
        loop_ctx = {
            "items": items,
            "current_index": 0,
            "item_var": item_var,
            "array_mode": config.get("array_mode", "variable"),
        }
    else:
        items = loop_ctx.get("items", [])
        loop_ctx["item_var"] = item_var
    
    total_items = len(items)
    current_index = loop_ctx.get("current_index", 0)
    
    if current_index < total_items:
        current_item = items[current_index]
        loop_ctx["current_index"] = current_index + 1
        loop_states[block.id] = loop_ctx
        block_output = {
            "output": current_item,
            item_var: current_item,
            "item_index": current_index,
            "count": total_items,
            "route": "loop",
        }
    else:
        # Loop complete
        loop_states.pop(block.id, None)
        block_output = {
            "output": {"items": items, "count": total_items},
            "count": total_items,
            "route": "exit",
        }
    
    # Always expose the full collection for reference
    block_output.setdefault("items", items)
    
    return {
        "loop_state": loop_states,
        "block_outputs": {
            block.id: block_output
        }
    }


# Node function registry
NODE_FUNCTIONS = {
    BlockType.INPUT: input_node,
    BlockType.TOOL: tool_node,
    BlockType.LLM: llm_node,
    BlockType.IF_ELSE: if_else_node,
    BlockType.FOR_LOOP: for_loop_node,
    BlockType.VARIABLE: variable_node,
}


def get_node_function(block_type: BlockType):
    """Get node function for a block type."""
    return NODE_FUNCTIONS.get(block_type)


__all__ = [
    "resolve_inputs",
    "build_variable_map",
    "resolve_template_variables",
    "resolve_template_payload",
    "input_node",
    "variable_node",
    "tool_node",
    "llm_node",
    "if_else_node",
    "for_loop_node",
    "get_node_function",
    "NODE_FUNCTIONS",
]

