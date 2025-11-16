"""Utility for formatting tool schemas for LLM prompts."""
from typing import Dict, List
from shared.tools import ToolEntry
from shared.logger import get_logger

logger = get_logger("shared.tools.schema_formatter")


def format_tool_schemas_for_llm(
    tool_entries: Dict[str, ToolEntry], 
    available_tool_names: List[str]
) -> str:
    """
    Formats tool schemas into a string for LLM prompts.
    Shows all parameters and highlights required parameters.
    
    This is the 'explicit contract' for the LLM - it needs to see:
    1. What tools are available
    2. What parameters each tool accepts
    3. Which parameters are required vs optional
    
    Args:
        tool_entries: Dict mapping tool names (lowercase) to ToolEntry objects
        available_tool_names: List of tool names to format
        
    Returns:
        Formatted string with tool descriptions, params, and required params
        
    Example output:
        Tool: `GITHUB_LIST_PULL_REQUESTS`
          Description: Lists pull requests in a GitHub repository
          All Params: ['owner', 'repo', 'state', 'page']
          Required Params: ['owner', 'repo']
    """
    lines = []
    if not available_tool_names:
        return "No tools were selected."
    
    for name in available_tool_names:
        entry = tool_entries.get(name.lower())
        
        if not entry:
            logger.warning(f"Tool '{name}' was in available_tools but not in tool_entries dict.")
            continue

        # Check if the schema is a valid, non-empty dictionary
        if not entry.pydantic_schema or not isinstance(entry.pydantic_schema, dict):
            lines.append(f"Tool: `{name}` (No parameters)")
            continue
        
        schema = entry.pydantic_schema
        
        try:
            # Get all fields from the 'properties' key
            all_fields = list(schema.get('properties', {}).keys())
            # Get required fields from the 'required' key
            required_fields = schema.get('required', [])
        except Exception:
            # Fallback in case the schema is malformed
            logger.warning(f"Could not parse schema for tool: {name}", exc_info=True)
            all_fields = ["(Schema format un-parseable)"]
            required_fields = []

        lines.append(
            f"Tool: `{name}`\n"
            f"  Description: {entry.description}\n"
            f"  All Params: {all_fields or 'None'}\n"
            f"  Required Params: {required_fields or 'None'}"
        )
    return "\n".join(lines)

