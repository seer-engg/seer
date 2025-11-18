"""
Central client for managing Multi-Server MCP (Multi-Modal Communication Protocol) connections.

Reads a central mcp.json configuration file and provides functions
to get tools for specified services with caching support.
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from shared.logger import get_logger

logger = get_logger("shared.mcp_client")

# Path to the MCP service configuration
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp.json"

# Cache for service definitions
_service_definitions: List[Dict[str, Any]] = []

# Cache for MCP clients and tools (keyed by frozenset of service names)
_client_cache: Dict[frozenset, Tuple[MultiServerMCPClient, Dict[str, Dict[str, Any]]]] = {}
_tools_cache: Dict[frozenset, List[BaseTool]] = {}

async def _load_mcp_services() -> List[Dict[str, Any]]:
    """Loads MCP service definitions from mcp.json without blocking the event loop."""
    global _service_definitions
    if _service_definitions:
        return _service_definitions

    if not MCP_CONFIG_PATH.exists():
        logger.warning(f"MCP config file not found at {MCP_CONFIG_PATH}. No MCP services will be available.")
        return []

    try:
        def _read_config_sync() -> Dict[str, Any]:
            with open(MCP_CONFIG_PATH, "r") as f:
                return json.load(f)

        config = await asyncio.to_thread(_read_config_sync)
        _service_definitions = config.get("services", [])
        logger.info(f"Loaded {len(_service_definitions)} MCP service definitions.")
        return _service_definitions
    except Exception as e:
        logger.error(f"Failed to load or parse {MCP_CONFIG_PATH}: {e}")
        return []

async def _create_mcp_client_and_services(
    service_names: List[str],
) -> Tuple[MultiServerMCPClient, Dict[str, Dict[str, Any]]]:
    """
    Creates an MCP client for the specified services with caching.
    
    Returns:
        A tuple of (MultiServerMCPClient, service_config_dict)
    """
    # Check cache first
    service_key = frozenset(service_names)
    if service_key in _client_cache:
        logger.debug(f"Returning cached MCP client for services: {service_names}")
        return _client_cache[service_key]
    all_services = await _load_mcp_services()
    
    services_to_load = {}
    service_configs = {}
    
    for service_name in service_names:
        found = False
        for service_def in all_services:
            if service_def.get("name") == service_name:
                service_configs[service_name] = service_def
                services_to_load[service_name] = {
                    "transport": "streamable_http",
                    "url": service_def.get("url"),
                }
                found = True
                break
        if not found:
            logger.warning(f"Service '{service_name}' requested but not found in mcp.json")

    if not services_to_load:
        logger.info("No valid MCP services requested or configured. Returning empty client.")
        empty_result = (MultiServerMCPClient({}), {})
        _client_cache[service_key] = empty_result
        return empty_result

    mcp_client = MultiServerMCPClient(services_to_load)
    result = (mcp_client, service_configs)
    _client_cache[service_key] = result
    logger.info(f"Created and cached MCP client for services: {service_names}")
    return result


async def get_mcp_tools(service_names: List[str]) -> List[BaseTool]:
    """
    Get a list of LangChain tools for the specified MCP services with caching.
    """
    if not service_names:
        return []

    # Check cache first
    service_key = frozenset(service_names)
    if service_key in _tools_cache:
        logger.debug(f"Returning cached tools for services: {service_names}")
        return _tools_cache[service_key]

    logger.info(f"Initializing MCP tools for services: {service_names}")
    mcp_client, _ = await _create_mcp_client_and_services(service_names)
    
    try:
        tools = await mcp_client.get_tools()
        logger.info(f"Successfully fetched {len(tools)} tools for {service_names}")
        _tools_cache[service_key] = tools
        return tools
    except Exception as e:
        logger.error(f"Failed to get MCP tools for {service_names}: {e}")
        return []

async def get_mcp_client_and_configs(
    service_names: List[str],
) -> Tuple[MultiServerMCPClient, Dict[str, Dict[str, Any]]]:
    """
    Get the raw MCP client and the configs for the specified services.
    Used by runners that need to invoke tools directly.
    """
    if not service_names:
        return MultiServerMCPClient({}), {}
    
    return await _create_mcp_client_and_services(service_names)
