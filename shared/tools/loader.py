"""
MCP tool loader.
Handles loading tools from MCP services.
"""
from typing import Dict, List, Sequence
from langchain_core.tools import BaseTool

from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs
from shared.tools.registry import ToolEntry
from shared.config import EVAL_AGENT_LOAD_DEFAULT_MCPS

logger = get_logger("shared.tools.loader")

DEFAULT_MCP_SERVICES: Sequence[str] = ("asana", "github", "langchain_docs")


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

    if not EVAL_AGENT_LOAD_DEFAULT_MCPS:
        return normalized

    combined: List[str] = list(DEFAULT_MCP_SERVICES)
    for service in normalized:
        if service not in combined:
            combined.append(service)
    return combined


async def load_tool_entries(service_names: Sequence[str]) -> Dict[str, ToolEntry]:
    """
    Return lightweight metadata for the requested MCP tools keyed by name.
    
    Args:
        service_names: List of MCP service names to load tools from
        
    Returns:
        Dict of ToolEntry objects keyed by lowercase tool name
    """
    if not service_names:
        return {}

    entries: Dict[str, ToolEntry] = {}

    # Load tools PER service to correctly tag with originating service
    # and avoid defaulting to 'misc' when names are not prefixed.
    for service in service_names:
        if not service:
            continue
        mcp_client, _ = await get_mcp_client_and_configs([service])
        tools: List[BaseTool] = await mcp_client.get_tools()

        for tool in tools:
            # Extract JSON schema - handles both dict and Pydantic model formats
            args_schema = getattr(tool, "args_schema", None)
            json_schema = None
            if args_schema:
                if isinstance(args_schema, dict):
                    # Already a JSON schema dict (e.g., from MCP tools)
                    json_schema = args_schema
                elif hasattr(args_schema, "model_json_schema"):
                    # Pydantic model - extract schema
                    json_schema = args_schema.model_json_schema()

            entry = ToolEntry(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                service=str(service).strip().lower(),
                pydantic_schema=json_schema,
            )
            # Key by a stable, service-qualified key to prevent cross-service collisions
            entries[f"{entry.service}::{entry.name}".lower()] = entry

    logger.info(
        "Loaded %d MCP tool entries for services: %s",
        len(entries),
        ", ".join(service_names),
    )
    return entries

