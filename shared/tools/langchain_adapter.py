"""
Adapter to convert BaseTool instances to LangChain tools.
"""
from typing import Any, Dict, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model
from shared.tools.base import BaseTool
from shared.tools.executor import execute_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.langchain_adapter")


def _create_input_model(tool: BaseTool) -> type[BaseModel]:
    """
    Create a Pydantic model from tool's parameter schema.
    
    Args:
        tool: BaseTool instance
    
    Returns:
        Pydantic model class for tool inputs
    """
    schema = tool.get_parameters_schema()
    
    # Extract properties and required fields
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Create field definitions for Pydantic
    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        # Map JSON schema types to Python types
        if prop_type == "string":
            python_type = str
        elif prop_type == "integer":
            python_type = int
        elif prop_type == "number":
            python_type = float
        elif prop_type == "boolean":
            python_type = bool
        elif prop_type == "array":
            python_type = list
        elif prop_type == "object":
            python_type = dict
        else:
            python_type = str  # Default to string
        
        # Check if field is required
        if prop_name in required:
            field_definitions[prop_name] = (python_type, ...)
        else:
            field_definitions[prop_name] = (Optional[python_type], None)
    
    # Create model class
    model_name = f"{tool.name.title().replace('_', '')}Input"
    return create_model(model_name, **field_definitions)


def base_tool_to_langchain_tool(base_tool: BaseTool, user_id: str) -> StructuredTool:
    """
    Convert a BaseTool instance to a LangChain StructuredTool.
    
    Args:
        base_tool: BaseTool instance
        user_id: User ID for tool execution
    
    Returns:
        LangChain StructuredTool instance
    """
    # Create input model from tool's parameter schema
    try:
        input_model = _create_input_model(base_tool)
    except Exception as e:
        logger.warning(f"Failed to create input model for {base_tool.name}, using dict: {e}")
        input_model = Dict[str, Any]
    
    async def tool_func(**kwargs) -> str:
        """
        LangChain tool function that wraps BaseTool execution.
        
        Args:
            **kwargs: Tool arguments from LangChain
        
        Returns:
            Tool execution result as string
        """
        # Extract connection_id if provided (for OAuth tools)
        connection_id = kwargs.pop("connection_id", None)
        
        # Execute tool using executor
        try:
            result = await execute_tool(
                tool_name=base_tool.name,
                user_id=user_id,
                connection_id=connection_id,
                arguments=kwargs
            )
            
            # Convert result to string for LangChain
            if isinstance(result, (dict, list)):
                import json
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return f"Error: {str(e)}"
    
    # Create LangChain tool
    return StructuredTool.from_function(
        func=tool_func,
        name=base_tool.name,
        description=base_tool.description,
        args_schema=input_model if input_model != Dict[str, Any] else None,
    )


def get_langchain_tools_from_registry(user_id: str, integration_type: Optional[str] = None) -> list[StructuredTool]:
    """
    Get LangChain tools from the tool registry.
    
    Args:
        user_id: User ID for tool execution
        integration_type: Optional filter by integration type
    
    Returns:
        List of LangChain StructuredTool instances
    """
    from shared.tools.registry import get_tools_by_integration
    
    tools_meta = get_tools_by_integration(integration_type)
    langchain_tools = []
    
    for tool_meta in tools_meta:
        tool_name = tool_meta["name"]
        from shared.tools.base import get_tool
        base_tool = get_tool(tool_name)
        
        if base_tool:
            langchain_tool = base_tool_to_langchain_tool(base_tool, user_id)
            langchain_tools.append(langchain_tool)
        else:
            logger.warning(f"Tool {tool_name} not found in registry")
    
    return langchain_tools

