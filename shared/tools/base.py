"""
Base tool interface for custom tool system.

All tools inherit from BaseTool and implement the execute() method.

Resource Picker System:
-----------------------
Tools can declare parameters that support resource browsing via the 
`x-resource-picker` schema extension. This allows the UI to render
a resource browser instead of a plain text input.

Example resource picker schema:
{
    "spreadsheet_id": {
        "type": "string",
        "description": "Google Sheets spreadsheet ID",
        "x-resource-picker": {
            "resource_type": "google_drive_file",
            "filter": {"mimeType": "application/vnd.google-apps.spreadsheet"},
            "display_field": "name",
            "value_field": "id",
            "search_enabled": True,
            "hierarchy": True  # Enables folder navigation
        }
    }
}

The frontend will detect `x-resource-picker` and render a ResourcePicker component
that calls /api/integrations/{provider}/resources/{resource_type} to list resources.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict
from shared.logger import get_logger

logger = get_logger("shared.tools.base")


class ResourcePickerConfig(TypedDict, total=False):
    """Configuration for resource picker UI component."""
    resource_type: str  # Type of resource to browse (e.g., 'google_drive_file', 'github_repo')
    filter: Dict[str, Any]  # Filters to apply when fetching resources
    display_field: str  # Field to display in the picker (default: 'name')
    value_field: str  # Field to use as the parameter value (default: 'id')
    search_enabled: bool  # Enable search functionality
    hierarchy: bool  # Enable folder/hierarchy navigation
    depends_on: str  # Another parameter this depends on (for nested resources)
    endpoint: str  # Custom endpoint override (optional)


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
    
    Optional:
    - get_resource_pickers(): Define which parameters support resource browsing
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
    
    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        """
        Get resource picker configurations for parameters that support browsing.
        
        Override this in subclasses to enable resource browsing for specific parameters.
        
        Returns:
            Dict mapping parameter names to ResourcePickerConfig
        
        Example:
            return {
                "spreadsheet_id": {
                    "resource_type": "google_drive_file",
                    "filter": {"mimeType": "application/vnd.google-apps.spreadsheet"},
                    "display_field": "name",
                    "value_field": "id",
                    "search_enabled": True,
                    "hierarchy": True
                }
            }
        """
        return {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata for API responses.
        
        Returns:
            Dict with tool metadata including resource picker configs
        """
        schema = self.get_parameters_schema()
        resource_pickers = self.get_resource_pickers()
        
        # Inject x-resource-picker into schema properties
        if resource_pickers and "properties" in schema:
            for param_name, picker_config in resource_pickers.items():
                if param_name in schema["properties"]:
                    schema["properties"][param_name]["x-resource-picker"] = picker_config
        
        return {
            "name": self.name,
            "description": self.description,
            "required_scopes": self.required_scopes,
            "integration_type": self.integration_type,
            "provider": self.provider,
            "parameters": schema,
            "resource_pickers": resource_pickers  # Also include separately for convenience
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

