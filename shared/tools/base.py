"""
Base tool interface for custom tool system.

All tools inherit from BaseTool and implement the execute() method.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from shared.logger import get_logger

logger = get_logger("shared.tools.base")


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools must implement:
    - name: Tool identifier
    - description: Human-readable description
    - required_scopes: List of OAuth scopes needed (empty for non-OAuth tools)
    - integration_type: Integration type (gmail, github, googledrive, etc.)
    - provider: OAuth provider (google, github, etc.) - used for OAuth connections
    - execute(): Tool execution logic
    """
    
    name: str
    description: str
    required_scopes: List[str] = []
    integration_type: Optional[str] = None  # e.g., 'gmail', 'github', 'googledrive'
    provider: Optional[str] = None  # e.g., 'google', 'github' - OAuth provider for connections
    
    @abstractmethod
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool.
        
        Args:
            access_token: OAuth access token (None for non-OAuth tools)
            arguments: Tool-specific arguments
        
        Returns:
            Tool execution result (any serializable type)
        
        Raises:
            Exception: If tool execution fails
        """
        pass
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool parameters.
        
        Returns:
            JSON schema dict describing tool parameters
        
        Default implementation returns empty schema.
        Override in subclasses to provide parameter validation.
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata for API responses.
        
        Returns:
            Dict with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_scopes": self.required_scopes,
            "integration_type": self.integration_type,
            "provider": self.provider,
            "parameters": self.get_parameters_schema()
        }


# Tool registry
_TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """
    Register a tool in the global registry.
    
    Args:
        tool: Tool instance to register
    """
    if tool.name in _TOOL_REGISTRY:
        logger.warning(f"Tool '{tool.name}' is already registered. Overwriting.")
    _TOOL_REGISTRY[tool.name] = tool
    logger.info(f"Registered tool: {tool.name}")


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Get a tool by name from the registry.
    
    Args:
        name: Tool name
    
    Returns:
        Tool instance or None if not found
    """
    return _TOOL_REGISTRY.get(name)


def list_tools() -> List[BaseTool]:
    """
    List all registered tools.
    
    Returns:
        List of all registered tool instances
    """
    return list(_TOOL_REGISTRY.values())


def clear_registry() -> None:
    """Clear the tool registry (mainly for testing)."""
    _TOOL_REGISTRY.clear()

