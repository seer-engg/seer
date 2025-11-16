"""
Tool metadata registry.
Manages tool entries and metadata.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ToolEntry:
    """Metadata for a single MCP tool."""
    name: str
    description: str
    service: str
    pydantic_schema: Optional[Dict[str, Any]] = None


def build_tool_name_set(entries: Dict[str, ToolEntry]) -> Dict[str, str]:
    """
    Return canonical tool keys mapped to their original names.
    
    Args:
        entries: Dict of tool entries keyed by canonical name
        
    Returns:
        Dict mapping canonical names to original names
    """
    from shared.tools.normalizer import canonicalize_tool_name
    
    return {
        canonicalize_tool_name(entry.name): entry.name
        for entry in entries.values()
    }

