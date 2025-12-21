"""
Composio tool discovery and execution utilities.
Uses Pinecone vector store for semantic tool search.
"""
from shared.config import config
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from langchain_core.tools import tool
from composio import Composio
from composio_langchain import LangchainProvider
from pydantic import ValidationError

from tool_hub import ToolHub
from agents.supervisor.models import ToolParameter, ToolDefinition
from agents.supervisor.state import SupervisorState

import logging
logger = logging.getLogger(__name__)

# Global Pinecone store instance (lazy-loaded)
# Note: embedding_dimensions must match Pinecone index dimension (512)
_TOOLHUB_INSTANCE = None

def _get_toolhub_instance():
    """
    Lazy-load ToolHub instance.
    Uses config module which ensures all required environment variables are available.
    """
    global _TOOLHUB_INSTANCE
    if _TOOLHUB_INSTANCE is None:
        # Use config module (environment variables validated on import)
        _TOOLHUB_INSTANCE = ToolHub(
            openai_api_key=config.openai_api_key,
            pinecone_index_name=config.PINECONE_INDEX_NAME,
            pinecone_api_key=config.PINECONE_API_KEY,
            embedding_dimensions=512  # Must match Pinecone index dimension
        )
        logger.info("✅ ToolHub instance initialized")
    
    return _TOOLHUB_INSTANCE

def get_user_context_from_state(state: Optional[SupervisorState] = None) -> Dict[str, Any]:
    """
    Extract user context from SupervisorState.
    
    Returns:
        dict with:
        - user_id: str (from context or env var fallback)
        - connected_accounts: Dict[str, str] (maps integration_type -> connected_account_id)
    """
    if not state:
        return {
            "user_id": os.getenv("COMPOSIO_USER_ID", "default"),
            "connected_accounts": {}
        }
    
    context = state.get("context", {})
    integrations = context.get("integrations", {})
    
    # Extract user_id from context (could be passed from frontend)
    user_id = context.get("user_id") or context.get("user_email")
    
    # Extract connected account IDs from integrations
    # Structure: { "github": {"id": "conn_abc123", "name": "repo"}, ... }
    connected_accounts = {}
    for integration_type, selection in integrations.items():
        if isinstance(selection, dict) and selection.get("id"):
            # The "id" field should be the Composio connected_account_id
            connected_accounts[integration_type] = selection["id"]
    
    return {
        "user_id": user_id or os.getenv("COMPOSIO_USER_ID", "default"),
        "connected_accounts": connected_accounts
    }


def get_available_integrations() -> List[str]:
    """
    Get list of available integrations.
    
    TODO: Replace with Pinecone API namespace search to dynamically discover
    available integrations from the vector store.
    
    Returns:
        List of integration names in lowercase (e.g., ["github", "asana"])
    """
    # Hardcoded list for now - TODO: Query Pinecone namespaces dynamically
    return [
        "github",
        "asana",
        "slack",
        "gmail",
        "googlecalendar",
        "googledocs",
        "googlesheets",
        "telegram",
        "twitter",
    ]

async def _search_tools_in_pinecone(
    query: str,
    integration_name: Optional[List[str]] = None,
    top_k: int = 3
) -> List[dict]:
    """
    Search tools from Pinecone using semantic search.
    
    Args:
        query: Search query string
        integration_name: Optional list of integration names to restrict search (e.g., ["github", "asana"])
        top_k: Number of results to return
    
    Returns:
        List of tool dictionaries
    """
    
    # Lazy-load ToolHub instance (initializes on first use)
    toolhub = _get_toolhub_instance()
    
    results = await toolhub.query(
        query=query,
        integration_name=integration_name,
        top_k=top_k
    )
    return results


@tool
async def search_tools(
    query: str,
    reasoning: str,
    integration_filter: Optional[List[str]] = None
) -> str:
    """
    Search for available tools/actions using semantic search via Pinecone vector store.
    
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
    
    **INTEGRATION FILTERING:**
    - integration_filter: Optional list of integration names to search (e.g., ["github", "asana"])
    - If not specified, searches all namespaces (slower but comprehensive)
    - Multiple integrations are searched in parallel and results are merged by relevance score
    
    **EXAMPLES:**
    search_tools(
        query="search tasks by title",
        reasoning="I need to find if a task already exists. The task title might match, so I need a tool that can search tasks by title string.",
        integration_filter=["asana"]  # Single integration
    )
    
    search_tools(
        query="create task or issue",
        reasoning="I need to create a task or issue in a project management system.",
        integration_filter=["asana", "github"]  # Multiple integrations (searches both Asana and GitHub namespaces)
    )
    
    Returns a JSON string containing the tool definitions with full parameter schemas.
    """
    try:
        # Log reasoning for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🔍 Search query: {query} | Reasoning: {reasoning}")
        
        # Search tools from Pinecone
        matched_tools = await _search_tools_in_pinecone(
            query=query,
            integration_name=integration_filter,
            top_k=3
        )
        
        if not matched_tools:
            logger.warning(f"No tools found for query: {query} (integration: {integration_filter})")
            return json.dumps([], indent=2)
        
        # Initialize Composio client to get full parameter schemas
        composio_api_key = os.getenv("COMPOSIO_API_KEY")
        if composio_api_key:
            client = Composio(api_key=composio_api_key, provider=LangchainProvider())
        else:
            client = Composio(provider=LangchainProvider())
        
        # Get user_id from context store (user-specific, not env var)
        from tools.user_context_store import get_user_context_store
        user_context = get_user_context_store().get_user_context(thread_id="default")
        user_id = user_context["user_id"]
        
        # Fetch actual tool definitions from Composio to get parameters
        matches_dict_list = []
        tool_names = [tool_dict.get('name') for tool_dict in matched_tools]
        
        # Fetch actual tools from Composio to get parameter schemas (async-safe)
        tool_dict_by_name = {}
        try:
            # Wrap blocking call in asyncio.to_thread to avoid blocking event loop
            actual_tools = await asyncio.to_thread(
                client.tools.get,
                user_id=user_id,
                tools=tool_names
            )
            tool_dict_by_name = {tool.name: tool for tool in actual_tools}
        except Exception as e:
            logger.warning(f"Could not fetch all tools from Composio: {e}")
        
        def _extract_parameters_from_schema(schema_dict: dict) -> List[ToolParameter]:
            """Extract ToolParameter list from JSON schema dict."""
            parameters = []
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
            return parameters
        
        for tool_dict in matched_tools:
            tool_name = tool_dict.get('name')
            tool_obj = tool_dict_by_name.get(tool_name)
            
            # Extract parameters - prefer Composio schema, fallback to Pinecone metadata
            parameters = []
            
            # Try to get parameters from Composio tool object first
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
                        
                        if schema_dict:
                            parameters = _extract_parameters_from_schema(schema_dict)
                except Exception as e:
                    logger.warning(f"Could not extract schema from Composio tool {tool_name}: {e}")
            
            # Fallback: Use parameters from Pinecone metadata if Composio fetch failed
            if not parameters:
                pinecone_params = tool_dict.get('parameters', {})
                if pinecone_params and isinstance(pinecone_params, dict):
                    try:
                        # Pinecone stores parameters as JSON Schema format
                        # Handle both direct properties dict and nested schema format
                        if 'properties' in pinecone_params:
                            # Full JSON Schema format
                            parameters = _extract_parameters_from_schema(pinecone_params)
                        elif pinecone_params:
                            # Direct properties dict format
                            # Convert to JSON Schema format
                            schema_dict = {
                                'properties': pinecone_params,
                                'required': []  # We don't store required separately in this format
                            }
                            parameters = _extract_parameters_from_schema(schema_dict)
                    except Exception as e:
                        logger.warning(f"Could not extract parameters from Pinecone metadata for {tool_name}: {e}")
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
async def execute_tool(tool_name: str, params: str) -> str:
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
    
    # **CHECK FOR PLANNED EXECUTION (Enforcement):**
    from tools.runtime_tool_store import _runtime_tool_store
    
    # TODO: Extract thread_id from runtime context if available
    planned_execution = _runtime_tool_store.get_planned_execution(clean_name, thread_id="default")
    
    # Enforce that think() was called first (middleware should handle this, but double-check)
    if not planned_execution:
        return (
            f"❌ ERROR: You MUST call think() before calling execute_tool.\n"
            f"Call think(scratchpad, last_tool_call) first to plan execution with:\n"
            f"- Tool name: {clean_name}\n"
            f"- All required parameters with reasoning\n"
            f"Then call execute_tool with the planned parameters."
        )
    
    # If planned execution exists, use validated params from think()
    # Params are already validated by Pydantic in think(), so we can trust them
    # Still validate against Composio's actual tool schema (different validation)
    tool_schema = _runtime_tool_store.get_tool_schema(clean_name)
    
    # Clear planned execution after use
    _runtime_tool_store.clear_planned_execution(clean_name, thread_id="default")
    
    # Get user_id and connected_account_id from context store (user-specific, not env var)
    from tools.user_context_store import get_user_context_store
    user_context = get_user_context_store().get_user_context(thread_id="default")
    user_id = user_context["user_id"]
    connected_accounts = user_context["connected_accounts"]
        
    client = Composio(provider=LangchainProvider())
    
    # Determine which integration this tool belongs to (for connected_account_id)
    # Tool names typically follow pattern: INTEGRATION_ACTION (e.g., GITHUB_FIND_PULL_REQUESTS)
    integration_type = None
    tool_name_lower = clean_name.lower()
    for integration in ["github", "asana", "googledrive", "googledocs", "googlesheets", "gmail", "slack"]:
        if tool_name_lower.startswith(integration):
            integration_type = integration
            break
    
    # Use connected_account_id if available for this integration
    connected_account_id = connected_accounts.get(integration_type) if integration_type else None
    
    try:
        # CRITICAL: Use tools.execute() with connected_account_id to ensure we use the user's account
        # NOT the sandbox account. This matches the proxy pattern.
        # Wrap blocking call in asyncio.to_thread to avoid blocking event loop
        def _execute_tool():
            # Use tools.execute() which accepts connected_account_id parameter
            # This ensures we use the user's connected account, not the sandbox
            result = client.tools.execute(
                clean_name,
                user_id=user_id,
                # connected_account_id=connected_account_id,
                arguments=args,
                dangerously_skip_version_check=True,
            )
            return result
        
        # Debug logging
        logger.info("[execute_tool] Tool: %s, Integration: %s", clean_name, integration_type)
        logger.info("[execute_tool] user_id: %s", user_id)
        logger.info("[execute_tool] connected_accounts dict: %s", connected_accounts)
        logger.info("[execute_tool] connected_account_id for %s: %s", integration_type, connected_account_id)
        
        if connected_account_id:
            logger.info("Using connected_account_id %s for %s tool %s", connected_account_id, integration_type, clean_name)
        else:
            logger.warning("No connected_account_id for %s tool %s - will try with user_id only: %s", integration_type, clean_name, user_id)
        
        result = await asyncio.to_thread(_execute_tool)
        result_str = str(result)
        
        return result_str
        
    except Exception as e:
        import traceback
        return f"Error executing {clean_name}: {traceback.format_exc()}"

