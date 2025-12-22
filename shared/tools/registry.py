"""
Tool metadata registry.
Manages tool entries and metadata.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from shared.tools.base import list_tools, get_tool as get_tool_from_base


@dataclass
class ToolEntry:
    """Metadata for a single MCP tool."""
    name: str
    description: str
    service: str


def get_tools_by_integration(integration_type: Optional[str] = None) -> List[Dict]:
    """
    Get tools filtered by integration type.
    
    Args:
        integration_type: Optional integration type filter (e.g., 'gmail', 'github')
    
    Returns:
        List of tool metadata dicts
    """
    all_tools = list_tools()
    
    if integration_type:
        # Filter tools by integration type
        # Integration type can be inferred from tool name or required_scopes
        filtered = []
        for tool in all_tools:
            # Check if tool's scopes match the integration type
            # e.g., gmail tools have 'gmail.readonly' scope
            if integration_type.lower() in tool.name.lower():
                filtered.append(tool)
            elif tool.required_scopes:
                for scope in tool.required_scopes:
                    if integration_type.lower() in scope.lower():
                        filtered.append(tool)
                        break
        return [tool.get_metadata() for tool in filtered]
    
    return [tool.get_metadata() for tool in all_tools]
