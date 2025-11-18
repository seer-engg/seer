"""
Centralized Tool Service for managing MCP tools.

This service provides a single interface for:
- Loading MCP tools
- Selecting relevant tools
- Managing tool lifecycle

Usage:
    from shared.tool_service import get_tool_service
    
    tool_service = get_tool_service()
    await tool_service.initialize(["asana", "github"])
    tools = tool_service.get_tools()
"""
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

from shared.logger import get_logger
from shared.mcp_client import get_mcp_tools
from shared.tools import (
    load_tool_entries,
    select_relevant_tools,
    canonicalize_tool_name,
    ToolEntry,
)

logger = get_logger("shared.tool_service")


class ToolService:
    """
    Centralized service for managing MCP tools.
    
    Provides a clean interface for tool loading, selection, and access.
    Handles caching automatically.
    """
    
    def __init__(self):
        self._tools_dict: Dict[str, BaseTool] = {}
        self._tool_entries: Dict[str, ToolEntry] = {}
        self._initialized_services: List[str] = []
        self._initialized = False
    
    async def initialize(self, services: List[str]) -> None:
        """
        Initialize tools for given MCP services.
        
        Args:
            services: List of MCP service names (e.g., ['asana', 'github'])
        """
        if not services:
            logger.warning("initialize called with empty services list")
            return
        
        # Check if we need to reload
        services_set = set(services)
        initialized_set = set(self._initialized_services)
        
        if self._initialized and services_set == initialized_set:
            logger.debug(f"Tools already initialized for services: {services}")
            return
        
        logger.info(f"Initializing MCP tools for services: {services}")
        
        # Load tool entries for metadata (service-qualified keys)
        self._tool_entries = await load_tool_entries(services)

        # Build tools dict keyed by canonical name INCLUDING the service hint.
        # Fetch tools per service to preserve origin and avoid name collisions.
        tools_dict: Dict[str, BaseTool] = {}
        for service in services:
            service = (service or "").strip().lower()
            if not service:
                continue
            service_tools = await get_mcp_tools([service])
            by_name = {t.name: t for t in service_tools}

            # Iterate only entries from this service to build canonical keys
            for entry_key, entry in self._tool_entries.items():
                if entry.service != service:
                    continue
                tool = by_name.get(entry.name)
                if not tool:
                    continue
                canonical_key = canonicalize_tool_name(entry.name, service_hint=service)
                tools_dict[canonical_key] = tool

        self._tools_dict = tools_dict
        
        self._initialized_services = list(services)
        self._initialized = True
        
        logger.info(f"Successfully initialized {len(self._tools_dict)} tools")
    
    def get_tools(self) -> Dict[str, BaseTool]:
        """
        Get all available tools as dict keyed by canonical name.
        
        Returns:
            Dict mapping canonical tool names to BaseTool instances
            
        Raises:
            RuntimeError: If initialize() hasn't been called yet
        """
        if not self._initialized:
            raise RuntimeError(
                "ToolService not initialized. Call await initialize(services) first."
            )
        return self._tools_dict
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a single tool by name (canonical or original).
        
        Args:
            tool_name: Tool name (will be canonicalized)
            
        Returns:
            BaseTool instance or None if not found
        """
        canonical_name = canonicalize_tool_name(tool_name)
        return self._tools_dict.get(canonical_name)
    
    def get_tool_entries(self) -> Dict[str, ToolEntry]:
        """
        Get tool entries (metadata) for all loaded tools.
        
        Returns:
            Dict mapping tool names to ToolEntry instances
        """
        if not self._initialized:
            raise RuntimeError(
                "ToolService not initialized. Call await initialize(services) first."
            )
        return self._tool_entries
    
    async def select_relevant_tools(
        self,
        context: str,
        max_total: int = 20,
        max_per_service: int = 5,
    ) -> List[BaseTool]:
        """
        Select relevant tools based on context.
        
        Uses semantic search (Neo4j) if available, falls back to keywords.
        
        Args:
            context: Context string for tool selection
            max_total: Maximum total tools to return
            max_per_service: Maximum tools per service
            
        Returns:
            List of selected BaseTool instances
        """
        if not self._initialized:
            raise RuntimeError(
                "ToolService not initialized. Call await initialize(services) first."
            )
        
        # Get relevant tool names
        tool_names = await select_relevant_tools(
            self._tool_entries,
            context,
            max_total=max_total,
            max_per_service=max_per_service,
        )
        
        # Convert names to tools
        selected_tools: List[BaseTool] = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                selected_tools.append(tool)
        
        logger.info(f"Selected {len(selected_tools)} relevant tools for context")
        return selected_tools


# ============================================================================
# Global Singleton
# ============================================================================

_tool_service: Optional[ToolService] = None


def get_tool_service() -> ToolService:
    """
    Get the global ToolService singleton.
    
    Returns:
        Global ToolService instance
    """
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service


def reset_tool_service() -> None:
    """
    Reset the global ToolService (primarily for testing).
    """
    global _tool_service
    _tool_service = None

