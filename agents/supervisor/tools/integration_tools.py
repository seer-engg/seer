"""
Integration tool discovery and execution utilities.
Uses local Chroma vector store for semantic tool search and new tool executor.
"""
from shared.config import config
import os
import json
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from langchain_core.tools import tool

from shared.tool_hub import LocalToolHub
from agents.supervisor.models import ToolParameter, ToolDefinition
from agents.supervisor.state import SupervisorState
from shared.tools.registry import get_tools_by_integration
from shared.tools.executor import execute_tool as execute_tool_with_oauth
from shared.tools.base import get_tool

import logging
logger = logging.getLogger(__name__)

def get_user_context_from_state(state: Optional[SupervisorState] = None) -> Dict[str, Any]:
    """
    Extract user context from SupervisorState.
    
    Returns:
        dict with:
        - user_id: str (from context or env var fallback)
        - connected_accounts: Dict[str, str] (maps integration_type -> connection_id)
    """
    if not state:
        return {
            "user_id": os.getenv("USER_ID", "default"),
            "connected_accounts": {}
        }
    
    context = state.get("context", {})
    integrations = context.get("integrations", {})
    
    # Extract user_id from context (could be passed from frontend)
    user_id = context.get("user_id") or context.get("user_email")
    
    # Extract connected account IDs from integrations
    # Structure: { "github": {"id": "gmail:123", "name": "repo"}, ... }
    # ID format is now "provider:id" or just "id" (OAuthConnection.id)
    connected_accounts = {}
    for integration_type, selection in integrations.items():
        if isinstance(selection, dict) and selection.get("id"):
            # The "id" field is now OAuthConnection.id
            connected_accounts[integration_type] = selection["id"]
    
    return {
        "user_id": user_id or os.getenv("USER_ID", "default"),
        "connected_accounts": connected_accounts
    }


def get_available_integrations() -> List[str]:
    """
    Get list of available integrations from tool registry.
    
    Returns:
        List of integration names in lowercase (e.g., ["github", "asana", "gmail"])
    """
    # Query tool registry to get available integrations
    all_tools = get_tools_by_integration()
    
    # Extract unique integration types from tool names/scopes
    integrations = set()
    for tool_meta in all_tools:
        tool_name = tool_meta.get("name", "").lower()
        required_scopes = tool_meta.get("required_scopes", [])
        
        # Check tool name for integration type
        # Extract integration type from tool name (e.g., "gmail_read_emails" -> "gmail")
        if "_" in tool_name:
            potential_integration = tool_name.split("_")[0]
            if potential_integration:
                integrations.add(potential_integration)
        
        # Also check scopes for integration hints
        for scope in required_scopes:
            if "gmail" in scope.lower():
                integrations.add("gmail")
            elif "github" in scope.lower():
                integrations.add("github")
            elif "asana" in scope.lower():
                integrations.add("asana")
    
    # Return sorted list, or empty list if no integrations found
    return sorted(list(integrations))

async def _search_tools_local(
    query: str,
    integration_name: Optional[List[str]] = None,
    top_k: int = 3
) -> List[dict]:
    """
    Search tools from local Chroma vector store using semantic search.
    
    Args:
        query: Search query string
        integration_name: Optional list of integration names to restrict search (e.g., ["github", "asana"])
        top_k: Number of results to return
    
    Returns:
        List of tool dictionaries
    """
    
    # Use shared LocalToolHub singleton instance
    from shared.tool_hub.singleton import get_toolhub_instance
    toolhub = get_toolhub_instance()
    
    if toolhub is None:
        logger.warning("LocalToolHub not available, returning empty results")
        return []
    
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
    Search for available tools/actions using semantic search via local Chroma vector store.
    
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
    
    Args:
        query: Search query describing the capability/action needed
        reasoning: Explanation of why you need this tool and what you're trying to accomplish
        integration_filter: Optional list of integration names to restrict search (e.g., ["github", "asana"])
    
    Returns:
        JSON string with list of matching tools and their parameters
    """
    try:
        results = await _search_tools_local(
            query=query,
            integration_name=integration_filter,
            top_k=5
        )
        
        if not results:
            return json.dumps({
                "tools": [],
                "message": f"No tools found matching query: {query}"
            })
        
        # Format results for supervisor
        tools_list = []
        for tool_data in results:
            tools_list.append({
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                "parameters": tool_data.get("parameters", {}),
                "integration": tool_data.get("service", "")
            })
        
        return json.dumps({
            "tools": tools_list,
            "query": query,
            "reasoning": reasoning
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error searching tools: {e}")
        return json.dumps({
            "tools": [],
            "error": str(e)
        })


def _normalize_nested_json_strings(data: Any) -> Any:
    """
    Recursively normalize nested JSON strings to actual objects.
    
    Example: {"data": "{\"completed\":true}"} -> {"data": {"completed":true}}
    """
    if isinstance(data, dict):
        return {k: _normalize_nested_json_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_normalize_nested_json_strings(item) for item in data]
    elif isinstance(data, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(data)
            # Recursively normalize parsed JSON
            return _normalize_nested_json_strings(parsed)
        except (json.JSONDecodeError, ValueError):
            return data
    else:
        return data


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
    # Still validate against tool schema if available
    tool_schema = _runtime_tool_store.get_tool_schema(clean_name)
    
    # Clear planned execution after use
    _runtime_tool_store.clear_planned_execution(clean_name, thread_id="default")
    
    # Get user_id and connection_id from context store (user-specific, not env var)
    from tools.user_context_store import get_user_context_store
    user_context = get_user_context_store().get_user_context(thread_id="default")
    user_id = user_context["user_id"]
    connected_accounts = user_context["connected_accounts"]
    
    # Determine which integration this tool belongs to (for connection_id)
    # Tool names typically follow pattern: INTEGRATION_ACTION (e.g., gmail_read_emails)
    integration_type = None
    tool_name_lower = clean_name.lower()
    # Get available integrations dynamically from registry
    available_integrations = get_available_integrations()
    for integration in available_integrations:
        if integration in tool_name_lower:
            integration_type = integration
            break
    
    # Use connection_id if available for this integration
    connection_id = connected_accounts.get(integration_type) if integration_type else None
    
    try:
        # Execute tool using new tool executor
        logger.info("[execute_tool] Tool: %s, Integration: %s", clean_name, integration_type)
        logger.info("[execute_tool] user_id: %s", user_id)
        logger.info("[execute_tool] connected_accounts dict: %s", connected_accounts)
        logger.info("[execute_tool] connection_id for %s: %s", integration_type, connection_id)
        
        if connection_id:
            logger.info("Using connection_id %s for %s tool %s", connection_id, integration_type, clean_name)
        else:
            logger.warning("No connection_id for %s tool %s - tool may not require OAuth", integration_type, clean_name)
        
        result = await execute_tool_with_oauth(
            tool_name=clean_name,
            user_id=user_id,
            connection_id=connection_id,
            arguments=args
        )
        
        # Convert result to string for supervisor
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)
        
        return result_str
        
    except Exception as e:
        import traceback
        logger.exception(f"Error executing tool {clean_name}: {e}")
        return f"Error executing {clean_name}: {traceback.format_exc()}"

