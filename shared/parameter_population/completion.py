"""
Layer 3: Aggressive Parameter Completion

Completes ActionStep parameters by filling ALL required fields or raising errors.
Uses schema-driven inference to populate missing parameters.

Design: Fail-fast with clear error messages when parameters can't be inferred.
"""
import json
from typing import Dict, Any, List, Optional

from shared.schema import ActionStep
from shared.tools import ToolEntry
from shared.parameter_population.parameter_inference import (
    infer_parameter_value,
    get_missing_required_parameters,
)
from shared.logger import get_logger

logger = get_logger("parameter_population.completion")


class ParameterCompletionError(Exception):
    """Raised when required parameters cannot be completed."""
    pass


def coerce_to_schema_type(value: Any, param_schema: Optional[Dict[str, Any]]) -> Any:
    """
    Coerce a value to match its JSON schema type definition.
    
    This function handles common type mismatches that occur when LLMs or
    inference systems generate parameters in native Python types but APIs
    expect different representations (e.g., boolean True vs string "true").
    
    Supported conversions:
    - boolean → string: True → "true", False → "false"
    - number → string: 123 → "123", 4.5 → "4.5"
    - string → boolean: "true"/"1"/"yes" → True
    - string → number: "123" → 123 or 123.0
    - null handling: preserves None values appropriately
    
    Args:
        value: The value to coerce (can be any type)
        param_schema: JSON schema for the parameter (containing 'type' field)
        
    Returns:
        Value coerced to match schema type, or original value if:
        - No schema provided
        - Value is None
        - Type already matches
        - Coercion not supported/failed
        
    Example:
        >>> schema = {"type": "string", "description": "Enable recursive mode"}
        >>> coerce_to_schema_type(True, schema)
        "true"
    """
    # Early returns for edge cases
    if not param_schema or value is None:
        return value
    
    schema_type = param_schema.get('type')
    if not schema_type:
        return value
    
    # Handle array types (union types with anyOf/oneOf)
    if isinstance(schema_type, list):
        # For union types like ["string", "null"], pick first non-null type
        non_null_types = [t for t in schema_type if t != 'null']
        schema_type = non_null_types[0] if non_null_types else 'string'
    
    current_type = type(value).__name__
    
    # Skip if already correct type
    if schema_type == 'string' and isinstance(value, str):
        return value
    if schema_type == 'boolean' and isinstance(value, bool):
        return value
    if schema_type == 'integer' and isinstance(value, int) and not isinstance(value, bool):
        return value
    if schema_type == 'number' and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    
    try:
        # Boolean → String (most common issue)
        if schema_type == 'string' and isinstance(value, bool):
            result = "true" if value else "false"
            logger.debug(f"Coerced boolean {value} → string '{result}'")
            return result
        
        # Number → String
        if schema_type == 'string' and isinstance(value, (int, float)):
            result = str(value)
            logger.debug(f"Coerced {current_type} {value} → string '{result}'")
            return result
        
        # String → Boolean
        if schema_type == 'boolean' and isinstance(value, str):
            result = value.lower() in ('true', '1', 'yes', 'y')
            logger.debug(f"Coerced string '{value}' → boolean {result}")
            return result
        
        # String → Integer
        if schema_type == 'integer' and isinstance(value, str):
            result = int(value)
            logger.debug(f"Coerced string '{value}' → integer {result}")
            return result
        
        # String → Number (float)
        if schema_type == 'number' and isinstance(value, str):
            result = float(value)
            logger.debug(f"Coerced string '{value}' → number {result}")
            return result
        
        # Integer → Number (int is valid for number type)
        if schema_type == 'number' and isinstance(value, int):
            return float(value)
        
        # Type already compatible or no coercion rule
        return value
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(
            f"Type coercion failed for value '{value}' ({current_type}) "
            f"to schema type '{schema_type}': {e}. Using original value."
        )
        return value


async def complete_action_parameters(
    action: ActionStep,
    tool_entries: Dict[str, ToolEntry],
    context_vars: Dict[str, Any],
    aggressive: bool = True,
) -> ActionStep:
    """
    Aggressively complete missing required parameters in an ActionStep.
    
    Process:
    1. Parse current parameters from action.params JSON string
    2. Get tool schema and identify required parameters
    3. For each missing required parameter, infer value from context
    4. If aggressive=True, FAIL if any required parameter can't be filled
    5. Return updated ActionStep with completed parameters
    
    Args:
        action: ActionStep with potentially incomplete parameters
        tool_entries: Dict of tool metadata with schemas
        context_vars: Available context variables for inference
        aggressive: If True, raise error on missing required params
        
    Returns:
        ActionStep with completed parameters
        
    Raises:
        ParameterCompletionError: If aggressive=True and required params missing
    """
    # Action.tool is already normalized by ActionStep validator
    tool_name = action.tool
    
    # Get tool entry and schema
    tool_entry = tool_entries.get(tool_name)
    if not tool_entry:
        logger.warning(f"Tool '{tool_name}' not found in tool_entries. Available: {list(tool_entries.keys())[:10]}")
        return action
    
    if not tool_entry.pydantic_schema:
        logger.warning(f"No schema available for tool '{tool_name}', skipping parameter completion")
        return action
    
    schema = tool_entry.pydantic_schema
    
    # Parse current parameters
    try:
        current_params = json.loads(action.params) if action.params else {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in action.params for {action.tool}: {action.params}")
        current_params = {}
    
    # ROBUSTNESS: Coerce existing parameters to match schema types
    # This catches type mismatches from LLM-generated parameters
    properties = schema.get('properties', {})
    coerced_params = {}
    for param_name, param_value in current_params.items():
        param_schema = properties.get(param_name)
        if param_schema:
            coerced_value = coerce_to_schema_type(param_value, param_schema)
            coerced_params[param_name] = coerced_value
            if coerced_value != param_value:
                logger.info(f"Coerced existing param '{param_name}': {param_value} → {coerced_value}")
        else:
            # No schema for this param, keep as-is
            coerced_params[param_name] = param_value
    
    current_params = coerced_params
    
    # Get required parameters from schema
    required_params = schema.get('required', [])
    if not required_params:
        logger.debug(f"No required parameters for tool '{action.tool}'")
        # Still return updated action with coerced params
        updated_action = action.model_copy()
        updated_action.params = json.dumps(current_params)
        return updated_action
    
    # Identify missing required parameters
    missing_params = get_missing_required_parameters(current_params, required_params)
    
    if not missing_params:
        logger.debug(f"All required parameters already present for '{action.tool}'")
        # Return updated action with coerced params
        updated_action = action.model_copy()
        updated_action.params = json.dumps(current_params)
        return updated_action
    
    logger.info(f"Attempting to complete {len(missing_params)} missing parameters for '{action.tool}': {missing_params}")
    
    # Attempt to infer each missing parameter
    completed_params = dict(current_params)
    still_missing = []
    
    for param_name in missing_params:
        param_schema = properties.get(param_name)
        
        inferred_value = await infer_parameter_value(
            param_name=param_name,
            tool_name=action.tool,
            context_vars=context_vars,
            param_schema=param_schema,
        )
        
        if inferred_value is not None:
            # Coerce inferred value to match schema type
            coerced_value = coerce_to_schema_type(inferred_value, param_schema)
            completed_params[param_name] = coerced_value
            if coerced_value != inferred_value:
                logger.info(f"  ✓ Completed & coerced {param_name}: {inferred_value} → {coerced_value}")
            else:
                logger.info(f"  ✓ Completed {param_name} = {coerced_value}")
        else:
            still_missing.append(param_name)
            logger.warning(f"  ✗ Could not infer {param_name}")
    
    # Aggressive mode: FAIL if any required parameters still missing
    if aggressive and still_missing:
        error_msg = (
            f"PARAMETER COMPLETION FAILED for tool '{action.tool}':\n"
            f"  Required parameters: {required_params}\n"
            f"  Missing parameters: {still_missing}\n"
            f"  Available context variables: {list(context_vars.keys())}\n"
            f"  Current params: {current_params}\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Ensure context extraction captures the needed variables\n"
            f"  2. Add parameter patterns to parameter_inference.py\n"
            f"  3. Provide the parameters explicitly in the LLM-generated action"
        )
        raise ParameterCompletionError(error_msg)
    
    # Update action with completed parameters
    updated_action = action.model_copy()
    updated_action.params = json.dumps(completed_params)
    
    return updated_action


async def complete_action_list(
    actions: List[ActionStep],
    tool_entries: Dict[str, ToolEntry],
    context_vars: Dict[str, Any],
    aggressive: bool = True,
) -> List[ActionStep]:
    """
    Complete parameters for a list of ActionSteps.
    
    Convenience function that applies complete_action_parameters to each action.
    
    Args:
        actions: List of ActionSteps to complete
        tool_entries: Tool metadata with schemas
        context_vars: Available context variables
        aggressive: Fail on missing required parameters
        
    Returns:
        List of completed ActionSteps
        
    Raises:
        ParameterCompletionError: If any action fails completion in aggressive mode
    """
    completed_actions = []
    
    for idx, action in enumerate(actions):
        try:
            completed_action = await complete_action_parameters(
                action=action,
                tool_entries=tool_entries,
                context_vars=context_vars,
                aggressive=aggressive,
            )
            completed_actions.append(completed_action)
        except ParameterCompletionError as e:
            # Add context about which action failed
            raise ParameterCompletionError(
                f"Action #{idx+1} ({action.tool}) failed completion:\n{str(e)}"
            ) from e
    
    return completed_actions

