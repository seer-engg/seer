"""
Composio tool discovery and execution utilities.
Shared by both baseline and baseline delegate architectures.
"""
import os
import sys
import json
import uuid
from typing import Optional, Tuple, Annotated
from langchain_core.tools import tool
from composio import Composio
from composio_langchain import LangchainProvider
from pydantic import ValidationError
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

# Import ToolHub for semantic search
sys.path.insert(0, "/home/akshay/tool_hub")
from tool_hub import ToolHub

from models import ToolParameter, ToolDefinition

# Global ToolHub instance
_TOOLHUB_INSTANCE = None
_toolhub_index_dir = "/home/akshay/tool_hub/examples/github_tool_index"

def _get_toolhub():
    """Get or create ToolHub instance."""
    global _TOOLHUB_INSTANCE
    if _TOOLHUB_INSTANCE is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
            
        _TOOLHUB_INSTANCE = ToolHub(openai_api_key=openai_api_key)
        # Load existing index
        if os.path.exists(os.path.join(_toolhub_index_dir, "tools.index")):
            try:
                _TOOLHUB_INSTANCE.load(_toolhub_index_dir)
            except:
                pass  # Rebuild if needed
    return _TOOLHUB_INSTANCE

@tool
def search_tools(query: str, reasoning: str) -> str:
    """
    Search for available tools/actions using semantic search (RAG).
    
    **MANDATORY REASONING:**
    Before searching, explain:
    1. What capability/action you need (e.g., "search tasks", "create task", "find PR")
    2. Why you need it (context from your task/instructions)
    3. What you're trying to accomplish
    
    **QUERY GUIDELINES:**
    - Search for CAPABILITIES, not specific data values
    - Use specific, action-oriented queries
    - GOOD: "search Asana tasks by title", "find GitHub pull request", "create Asana task"
    - BAD: "Asana", "GitHub", "search Asana tasks by title 'Seer: Evaluate my agent'" (includes actual data)
    
    **EXAMPLES:**
    search_tools(
        query="search Asana tasks by title",
        reasoning="I need to find if a task already exists for this PR. The task title might match the PR title, so I need a tool that can search Asana tasks by title string."
    )
    
    Returns a JSON string containing the tool definitions with full parameter schemas.
    """
    try:
        # Log reasoning for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🔍 Search query: {query} | Reasoning: {reasoning}")
        
        hub = _get_toolhub()
        matched_tools = hub.query(query, top_k=3)  # Reduced to 3 tools to reduce noise
        
        client = Composio(provider=LangchainProvider())
        user_id = os.getenv("COMPOSIO_USER_ID", "default")
        
        # Fetch actual tool definitions from Composio to get parameters
        matches_dict_list = []
        tool_names = [tool_dict.get('name') for tool_dict in matched_tools]
        
        # Fetch actual tools from Composio to get parameter schemas
        tool_dict_by_name = {}
        try:
            actual_tools = client.tools.get(user_id=user_id, tools=tool_names)
            tool_dict_by_name = {tool.name: tool for tool in actual_tools}
        except Exception as e:
            print(f"Warning: Could not fetch all tools from Composio: {e}")
        
        for tool_dict in matched_tools:
            tool_name = tool_dict.get('name')
            tool_obj = tool_dict_by_name.get(tool_name)
            
            # Extract parameters from actual tool schema
            parameters = []
            if tool_obj:
                try:
                    # LangChain tools have args_schema which is a Pydantic BaseModel
                    if hasattr(tool_obj, 'args_schema') and tool_obj.args_schema:
                        schema = tool_obj.args_schema
                        # Get JSON schema from Pydantic model
                        if hasattr(schema, 'model_json_schema'):
                            schema_dict = schema.model_json_schema()
                        elif hasattr(schema, 'schema'):
                            schema_dict = schema.schema()
                        else:
                            schema_dict = {}
                        
                        properties = schema_dict.get('properties', {})
                        required = schema_dict.get('required', [])
                        
                        for param_name, param_info in properties.items():
                            # Handle different schema formats
                            param_type = param_info.get('type', 'string')
                            if isinstance(param_type, list):
                                param_type = param_type[0] if param_type else 'string'
                            
                            parameters.append(ToolParameter(
                                name=param_name,
                                type=param_type,
                                description=param_info.get('description', ''),
                                required=param_name in required
                            ))
                except Exception as e:
                    # If schema extraction fails, at least include tool name
                    print(f"Warning: Could not extract schema for {tool_name}: {e}")
                    parameters = []
            
            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_dict.get('description', '')[:300],
                parameters=parameters
            )
            matches_dict_list.append(tool_def.model_dump())
            
            # Store tool schema in runtime store (global, no worker ID)
            from tools.runtime_tool_store import _runtime_tool_store
            _runtime_tool_store.store_tool_schema(tool_def)
            
        # Return JSON string directly - no memory I/O needed
        json_content = json.dumps(matches_dict_list, indent=2)
        return json_content
        
    except Exception as e:
        return f"Error searching tools: {e}"

def _normalize_nested_json_strings(obj):
    """Recursively parse JSON strings within nested dictionaries."""
    if isinstance(obj, dict):
        return {k: _normalize_nested_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_nested_json_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Try to parse as JSON if it looks like JSON
        if obj.strip().startswith(('{', '[')):
            try:
                parsed = json.loads(obj)
                # Recursively normalize the parsed JSON
                return _normalize_nested_json_strings(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
    return obj

def _get_tool_schema_summary(tool_obj):
    """Get a summary of the tool's expected schema for error messages."""
    if not hasattr(tool_obj, 'args_schema') or not tool_obj.args_schema:
        return "No schema available"
    
    schema = tool_obj.args_schema
    try:
        if hasattr(schema, 'model_json_schema'):
            schema_dict = schema.model_json_schema()
        elif hasattr(schema, 'schema'):
            schema_dict = schema.schema()
        else:
            return "No schema available"
        
        properties = schema_dict.get('properties', {})
        required = schema_dict.get('required', [])
        
        summary = []
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'string')
            if isinstance(param_type, list):
                param_type = param_type[0] if param_type else 'string'
            required_marker = " (required)" if param_name in required else ""
            summary.append(f"  - {param_name}: {param_type}{required_marker}")
        
        return "\n".join(summary) if summary else "No parameters defined"
    except Exception as e:
        return f"Error extracting schema: {e}"

def _validate_tool_args(tool_obj, args: dict) -> Tuple[bool, Optional[str]]:
    """Validate arguments against tool's Pydantic schema. Returns (is_valid, error_message)."""
    if not hasattr(tool_obj, 'args_schema') or not tool_obj.args_schema:
        return True, None  # No schema to validate against
    
    schema = tool_obj.args_schema
    try:
        # Try to instantiate the schema with the provided args
        schema(**args)
        return True, None
    except ValidationError as e:
        # Format validation errors nicely
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error.get('loc', []))
            msg = error.get('msg', '')
            errors.append(f"{field}: {msg}")
        
        schema_summary = _get_tool_schema_summary(tool_obj)
        error_msg = (
            f"Tool input validation error:\n"
            f"Errors: {', '.join(errors)}\n"
            f"Expected schema:\n{schema_summary}"
        )
        return False, error_msg
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def _validate_required_params(tool_def: ToolDefinition, args: dict) -> Tuple[bool, str]:
    """Validate that required params are present and non-empty."""
    missing = []
    empty = []
    
    for param in tool_def.parameters:
        if param.required:
            if param.name not in args:
                missing.append(f"{param.name} ({param.type})")
            elif param.type == 'string' and isinstance(args[param.name], str) and args[param.name].strip() == "":
                empty.append(f"{param.name} ({param.type})")
    
    if missing or empty:
        error_msg = []
        if missing:
            error_msg.append(f"Missing required params: {', '.join(missing)}")
        if empty:
            error_msg.append(f"Empty required string params: {', '.join(empty)}")
        return False, "\n".join(error_msg)
    
    return True, ""

@tool
async def execute_tool(tool_name: str, params: str, store: Annotated[BaseStore, InjectedStore] = None) -> str:
    """
    Execute a specific tool by name.
    params must be a JSON string of arguments.
    
    **REQUIREMENT:** You MUST call think() before calling this tool to plan your execution.
    
    Returns the full tool output. Workers see all outputs in their isolated, ephemeral context.
    No memory saving needed - worker's context is destroyed after completion.
    """
    clean_name = tool_name.replace("functions.", "")
    try:
        args = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: params must be valid JSON string. Parse error: {str(e)}"
    
    # Normalize nested JSON strings (e.g., {"data": "{\"completed\":true}"} -> {"data": {"completed":true}})
    args = _normalize_nested_json_strings(args)
    
    # **VALIDATE AGAINST PLANNED EXECUTION AND SCHEMA:**
    from tools.runtime_tool_store import _runtime_tool_store
    
    planned_execution = _runtime_tool_store.get_planned_execution(clean_name)
    tool_schema = _runtime_tool_store.get_tool_schema(clean_name)
    
    # Validate required params are present and non-empty
    if tool_schema:
        is_valid, validation_error = _validate_required_params(tool_schema, args)
        if not is_valid:
            planned_info = ""
            if planned_execution:
                planned_info = f"\n\nPlanned execution reasoning: {planned_execution.reasoning}"
            return (
                f"❌ Parameter validation error:\n{validation_error}{planned_info}\n\n"
                f"Please review the tool schema and ensure all required parameters are provided with non-empty values."
            )
    
    # Clear planned execution after use
    _runtime_tool_store.clear_planned_execution(clean_name)
        
    client = Composio(provider=LangchainProvider())
    user_id = os.getenv("COMPOSIO_USER_ID", "default")
    
    try:
        tools = client.tools.get(user_id=user_id, tools=[clean_name])
        if not tools:
            return f"Error: Tool '{clean_name}' not found."
        
        tool_to_use = tools[0]
        
        # Validate arguments against tool schema (Pydantic validation)
        is_valid, validation_error = _validate_tool_args(tool_to_use, args)
        if not is_valid:
            return validation_error or "Tool input validation error"
        
        # Execute and return full output - worker's context is isolated and ephemeral
        # No need to save to memory here - worker will summarize in final response if needed
        result = tool_to_use.invoke(args)
        result_str = str(result)
        
        return result_str
        
    except Exception as e:
        return f"Error executing {clean_name}: {str(e)}"
