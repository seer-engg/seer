"""
Central client for managing MCP tool access via Composio.

Reads a central mcp.json configuration file to determine which services
to expose, then uses Composio (with the LangChain provider) to return
LangChain-compatible tools. Caches clients and tool lists per service set.
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.tools import BaseTool
from shared.logger import get_logger
from urllib.parse import urlparse, parse_qs
from composio import Composio
from composio_langchain import LangchainProvider
from shared.config import COMPOSIO_USER_ID

logger = get_logger("shared.mcp_client")

# Path to the MCP service configuration
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp.json"

# Cache for service definitions
_service_definitions: List[Dict[str, Any]] = []

# Cache for MCP clients and tools (keyed by frozenset of service names)
_client_cache: Dict[frozenset, Tuple[Any, Dict[str, Dict[str, Any]]]] = {}
_tools_cache: Dict[frozenset, List[BaseTool]] = {}

class ComposioMCPClient:
    """
    Lightweight shim that matches the `.get_tools()` interface expected by callers,
    backed by Composio's LangChain provider.
    """
    def __init__(self, service_names: List[str], user_id: str):
        self._service_names = [str(s or "").strip().lower() for s in service_names or []]
        self._user_id = user_id
        self._composio = Composio(provider=LangchainProvider())

    @staticmethod
    def _to_toolkits(service_names: List[str]) -> List[str]:
        toolkits: List[str] = []
        for name in service_names:
            canonical = str(name).replace("-", "_").upper()
            toolkits.append(canonical)
        return toolkits

    async def get_tools(self) -> List[BaseTool]:
        toolkits = self._to_toolkits(self._service_names)
        tools = await asyncio.to_thread(
            self._composio.tools.get,
            user_id=self._user_id or "default",
            toolkits=toolkits,
            limit=100
        )
        return tools

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

def _extract_user_id_from_services_config(services: List[Dict[str, Any]]) -> str | None:
    """
    Attempt to extract a user_id query parameter from any configured service URL.
    """
    for svc in services or []:
        url = (svc or {}).get("url")
        if not url:
            continue
        try:
            parsed = urlparse(url)
            q = parse_qs(parsed.query)
            user_id_values = q.get("user_id")
            if user_id_values:
                return user_id_values[0]
        except Exception:
            continue
    return None

async def _create_mcp_client_and_services(
    service_names: List[str],
) -> Tuple[ComposioMCPClient, Dict[str, Dict[str, Any]]]:
    """
    Creates a Composio-backed MCP client for the specified services with caching.
    
    Returns:
        A tuple of (ComposioMCPClient, service_config_dict)
    """
    # Check cache first
    service_key = frozenset(service_names)
    if service_key in _client_cache:
        logger.debug(f"Returning cached MCP client for services: {service_names}")
        return _client_cache[service_key]
    all_services = await _load_mcp_services()
    
    service_configs = {}
    selected_services: List[Dict[str, Any]] = []
    
    for service_name in service_names:
        found = False
        for service_def in all_services:
            if service_def.get("name") == service_name:
                service_configs[service_name] = service_def
                selected_services.append(service_def)
                found = True
                break
        if not found:
            logger.warning(f"Service '{service_name}' requested but not found in mcp.json")

    if not selected_services:
        logger.info("No valid MCP services requested or configured. Returning empty Composio client.")
        empty_result = (ComposioMCPClient([], user_id="default"), {})
        _client_cache[service_key] = empty_result
        return empty_result

    # Determine Composio user id preference: env/config first, then attempt from mcp.json URLs, fallback to 'default'
    user_id = COMPOSIO_USER_ID or _extract_user_id_from_services_config(selected_services) or "default"
    mcp_client = ComposioMCPClient(list(service_names), user_id=user_id)
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
) -> Tuple[ComposioMCPClient, Dict[str, Dict[str, Any]]]:
    """
    Get the raw MCP client and the configs for the specified services.
    Used by runners that need to invoke tools directly.
    """
    if not service_names:
        return ComposioMCPClient([], user_id="default"), {}
    
    return await _create_mcp_client_and_services(service_names)
