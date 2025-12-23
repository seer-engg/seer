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
        integration_type_lower = integration_type.lower()
        filtered = []
        for tool in all_tools:
            # First check the integration_type property (most reliable)
            if tool.integration_type and tool.integration_type.lower() == integration_type_lower:
                filtered.append(tool)
                continue
            
            # Fallback: check if integration type is in tool name
            # e.g., "github_list_pull_requests" contains "github"
            if integration_type_lower in tool.name.lower():
                filtered.append(tool)
                continue
            
            # Last resort: check if integration type is in any scope
            # e.g., gmail tools have 'gmail.readonly' scope
            if tool.required_scopes:
                for scope in tool.required_scopes:
                    if integration_type_lower in scope.lower():
                        filtered.append(tool)
                        break
        
        return [tool.get_metadata() for tool in filtered]
    
    return [tool.get_metadata() for tool in all_tools]
