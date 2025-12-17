"""
MCP tool loader.
Handles loading tools from MCP services.
"""
from typing import List

from shared.logger import get_logger

logger = get_logger("shared.tools.loader")



def resolve_mcp_services(requested_services: List[str]) -> List[str]:
    """
    Normalize and optionally augment requested services with defaults.
    
    Args:
        requested_services: List of service names requested
        
    Returns:
        Normalized list of service names to load
    """
    normalized: List[str] = []
    for service in requested_services or []:
        if not service:
            continue
        normalized_name = service.strip().lower()
        if normalized_name and normalized_name not in normalized:
            normalized.append(normalized_name)

    return normalized

