from typing import Optional, Dict, Any
from shared.logger import get_logger
from workflow_core.schema import BlockType, BlockDefinition

logger = get_logger(__name__)

def with_block_config_defaults(
    block_type: Optional[str],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ensure required config defaults exist for specific block types."""
    config = config.copy() if config else {}
    if block_type == "tool":
        if "tool_params" in config:
            raise ValueError(
                "tool_params is no longer supported. Use 'params' to configure tool arguments."
            )
        if "inputs" in config and "params" not in config:
            config["params"] = config.pop("inputs")
    if block_type == "for_loop":
        legacy_array_var = config.pop("array_var", None)
        if legacy_array_var and "array_variable" not in config:
            config["array_variable"] = legacy_array_var
        array_mode = config.get("array_mode") or "variable"
        if array_mode not in ("variable", "literal"):
            array_mode = "variable"
        config["array_mode"] = array_mode
        if array_mode == "variable":
            config.setdefault("array_variable", "items")
        else:
            config.setdefault("array_literal", [])
        config.setdefault("item_var", "item")
    elif block_type == "variable":
        input_type = str(config.get("input_type") or "string").lower()
        if input_type not in ("string", "number", "array"):
            input_type = "string"
        config["input_type"] = input_type
        if "input" not in config:
            if input_type == "array":
                config["input"] = []
            elif input_type == "number":
                config["input"] = 0
            else:
                config["input"] = ""
        else:
            value = config["input"]
            if input_type == "array" and not isinstance(value, list):
                config["input"] = [value] if value is not None else []
            elif input_type == "number" and not isinstance(value, (int, float)):
                try:
                    config["input"] = float(value)
                except (TypeError, ValueError):
                    config["input"] = 0
            elif input_type == "string" and not isinstance(value, str):
                config["input"] = str(value) if value is not None else ""
    return config


def validate_block_config(
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
        elif "variable block" in error_msg.lower() or "input is required in config for variable blocks" in error_msg.lower():
            return "Variable blocks require an 'input' value (string, number, or array)."
        else:
            return f"Invalid block configuration: {error_msg}"
    except Exception as e:
        logger.warning(f"Unexpected error validating block config: {e}")
        return f"Validation error: {str(e)}"
