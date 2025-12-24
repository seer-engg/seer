from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from typing import Optional, List

from agents.supervisor.tools.integration_tools import search_tools, execute_tool
from agents.supervisor.tools.think_tool import think
from agents.supervisor.tools.user_context_store import get_user_context_store
from agents.supervisor.prompts import PROMPT_GENERIC_WORKER

def create_generic_worker(
    role_name: str, 
    specific_instructions: str,
    integrations: Optional[List[str]] = None
):
    """
    Creates a Generic Worker Sub-Agent.
    
    This agent is NOT hardcoded with specific tools.
    Instead, it uses `search_tools` and `execute_tool` to dynamically find what it needs.
    
    Args:
        role_name: Name of the role (e.g. "GitHub Researcher")
        specific_instructions: Task-specific system prompt additions.
        integrations: Optional list of integration names (lowercase, e.g., ["github", "asana"]).
                     If None, searches all integrations (slower but comprehensive).
                     If provided, restricts tool search to specified integrations.
    """
    
    # 1. Generic Toolset
    # Workers can search for tools and execute them. All tool outputs are visible in worker's isolated context.
    tools = [
        think,
        search_tools, 
        execute_tool, 
    ]
    
    # Get resource IDs (workspace GID, project GID, etc.) from user context if available
    # These are passed from the frontend when user selects resources, so workers don't need to discover them
    resource_context = ""
    try:
        user_context = get_user_context_store().get_user_context()
        resource_ids = user_context.get("resource_ids", {})
        
        if resource_ids:
            resource_lines = []
            for key, value in sorted(resource_ids.items()):
                # Format: "asana_workspace_gid" -> "Asana Workspace GID"
                parts = key.split("_")
                formatted_key = " ".join(word.capitalize() for word in parts)
                resource_lines.append(f"- {formatted_key}: {value}")
            
            if resource_lines:
                resource_context = f"\n\n**AVAILABLE RESOURCE IDs (from user selection):**\n" + "\n".join(resource_lines) + "\n\n**IMPORTANT:** Use these IDs directly in tool parameters. Do NOT try to discover or list workspaces/projects - use the provided IDs."
    except Exception:
        # If context is not available, continue without resource IDs
        pass
    
    # Add integration context to system prompt if integrations specified
    integration_context = ""
    if integrations:
        integration_names = ", ".join([i.upper() for i in integrations])
        integration_list = ", ".join([f'"{i}"' for i in integrations])
        integration_context = f"\n\n**INTEGRATION DOMAIN:** You are working with {integration_names} tools. When calling `search_tools`, use the `integration_filter` parameter to restrict searches to these integrations: `integration_filter=[{integration_list}]`. This makes searches faster and more relevant."
    
    # 2. System Prompt (PRUNED - ~500 chars vs ~1200)
    system_prompt = PROMPT_GENERIC_WORKER.format(
        ROLE_NAME=role_name,
        SPECIFIC_INSTRUCTIONS=specific_instructions,
        SECRETS_CONTEXT=resource_context,  # Now contains resource IDs from user selection
        INTEGRATION_CONTEXT=integration_context
    )
    
    # 3. Model and Middleware
    # We use a capable model since it needs to reason about tool discovery
    # Use config module which ensures OPENAI_API_KEY is available
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )
    
    # Middleware: Model retry + Tool call limits
    # ModelRetryMiddleware: Retries model calls with exponential backoff (4 total attempts: initial + 3 retries)
    # Tool call limits middleware - DOUBLED LIMITS for better worker autonomy
    middleware = [
        ModelRetryMiddleware(
            max_retries=3,  # 3 retries (4 total attempts)
            backoff_factor=2.0,  # Exponential backoff: 2s, 4s, 8s
            initial_delay=2.0,  # Initial delay of 2 seconds
        ),
        ToolCallLimitMiddleware(thread_limit=40, run_limit=16),  # Doubled global limit
        ToolCallLimitMiddleware(tool_name="search_tools", thread_limit=10, run_limit=6),  # Doubled
        ToolCallLimitMiddleware(tool_name="execute_tool", thread_limit=20, run_limit=10),  # Doubled
    ]
    
    # 4. Create agent using create_agent - it returns a compiled graph
    # create_agent handles the entire agent loop internally and accepts system_prompt directly
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware
    )
    
    return agent

