"""
GitHub MCP Tool Wrappers

Wraps GitHub MCP server tools as Seer BaseTool instances with scope validation.
Uses the official GitHub MCP server with per-request OAuth token injection.
"""
from typing import Any, Dict, Optional, List
from fastapi import HTTPException
import httpx

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool as LangChainBaseTool

from shared.tools.base import BaseTool, register_tool
from shared.tools.mcp_client import _load_tools
from shared.config import config
from shared.logger import get_logger

logger = get_logger("shared.tools.github")

# Tool-to-scope mapping (matches frontend)
# This is a simplified mapping - backend trusts frontend's scope requests
GITHUB_TOOL_SCOPES: Dict[str, list[str]] = {
    # Read-only PR tools
    "pull_request_read:get": ["repo"],
    "pull_request_read:get_comments": ["repo"],
    "pull_request_read:get_reviews": ["repo"],
    "pull_request_read:get_status": ["repo"],
    "pull_request_read:get_files": ["repo"],
    "pull_request_read:get_diff": ["repo"],
    "pull_request_read:get_sub_issues": ["repo"],
    "list_pull_requests": ["repo"],
    "search_pull_requests": ["repo"],
    
    # Write PR tools
    "create_pull_request": ["repo"],
    "update_pull_request": ["repo"],
    "merge_pull_request": ["repo"],
    "pull_request_review_write": ["repo"],
    "update_pull_request_branch": ["repo"],
    "add_comment_to_pending_review": ["repo"],
    "request_copilot_review": ["repo"],
    
    # Issue tools
    "issue_read:get": ["repo"],
    "issue_read:get_comments": ["repo"],
    "issue_read:get_sub_issues": ["repo"],
    "create_issue": ["repo"],
    "update_issue": ["repo"],
    "add_issue_comment": ["repo"],
    "close_issue": ["repo"],
    "reopen_issue": ["repo"],
    
    # Repository tools
    "get_repository": ["repo"],
    "list_repositories": ["repo"],
    "search_repositories": ["repo"],
    "get_repository_contents": ["repo"],
    "get_repository_file": ["repo"],
    
    # Branch tools
    "create_branch": ["repo"],
    "get_branch": ["repo"],
    "list_branches": ["repo"],
    
    # Commit tools
    "get_commit": ["repo"],
    "list_commits": ["repo"],
    "create_commit": ["repo"],
    
    # Default fallback
    "default": ["repo"],
}


class GitHubMCPTool(BaseTool):
    """Base wrapper for GitHub MCP tools."""
    
    def __init__(self, mcp_tool_name: str, description: str, parameters_schema: Optional[Dict[str, Any]] = None, integration_type: Optional[str] = "github"):
        self.mcp_tool_name = mcp_tool_name
        self.name = f"github_{mcp_tool_name.replace(':', '_')}"
        self.description = description
        self.required_scopes = GITHUB_TOOL_SCOPES.get(
            mcp_tool_name,
            GITHUB_TOOL_SCOPES.get("default", ["repo"])
        )
        self.integration_type = integration_type
        self.provider = "github"
        self._parameters_schema = parameters_schema or {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return self._parameters_schema
    
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        """
        Execute GitHub tool via the official GitHub MCP server.
        
        Creates a new MCP client per-request with the user's OAuth token
        injected in the Authorization header.
        
        Args:
            access_token: OAuth access token (required for GitHub API)
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=f"GitHub tool '{self.mcp_tool_name}' requires OAuth access token"
            )
        
        try:
            # First, try to execute via GitHub MCP server with OAuth token
            if config.GITHUB_MCP_SERVER_URL:
                try:
                    result = await self._execute_via_mcp(access_token, arguments)
                    return result
                except Exception as mcp_error:
                    logger.warning(f"MCP execution failed, falling back to direct API: {mcp_error}")
            
            # Fallback to direct GitHub API calls
            result = await self._execute_via_github_api(access_token, arguments)
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"GitHub tool execution failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"GitHub tool execution failed: {str(e)}"
            )
    
    async def _execute_via_mcp(self, access_token: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool via GitHub MCP server with OAuth token injection.
        
        Creates a new MCP client with the user's OAuth token in the
        Authorization header, then invokes the tool.
        """
        # Create a new MCP client with the OAuth token in headers
        mcp_client = MultiServerMCPClient({
            "github": {
                "transport": "streamable_http",
                "url": config.GITHUB_MCP_SERVER_URL,
                "headers": {
                    "Authorization": f"Bearer {access_token}",
                },
            }
        })
        
        try:
            # Get tools from the MCP server (authenticated with user's token)
            tools: List[LangChainBaseTool] = await mcp_client.get_tools(server_name="github")
            
            if not tools:
                raise HTTPException(
                    status_code=404,
                    detail="No tools available from GitHub MCP server"
                )
            
            # Find the matching tool
            target_tool = None
            for tool in tools:
                if tool.name == self.mcp_tool_name:
                    target_tool = tool
                    break
            
            if not target_tool:
                raise HTTPException(
                    status_code=404,
                    detail=f"GitHub MCP tool '{self.mcp_tool_name}' not found. Available: {[t.name for t in tools]}"
                )
            
            # Invoke the tool
            logger.info(f"Invoking GitHub MCP tool '{self.mcp_tool_name}' with args: {arguments}")
            result = await target_tool.ainvoke(arguments)
            logger.info(f"GitHub MCP tool '{self.mcp_tool_name}' completed successfully")
            
            return result
            
        finally:
            # Clean up the client session
            try:
                await mcp_client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
    
    async def _execute_via_github_api(self, access_token: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool by calling GitHub API directly.
        
        This is a fallback when MCP server doesn't support per-request tokens.
        For POC, we'll implement basic PR listing.
        """
        # Map MCP tool names to GitHub API endpoints
        if self.mcp_tool_name == "list_pull_requests":
            owner = arguments.get("owner")
            repo = arguments.get("repo")
            state = arguments.get("state", "open")
            
            if not owner or not repo:
                raise HTTPException(
                    status_code=400,
                    detail="owner and repo are required"
                )
            
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            params = {"state": state}
            
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url, headers=headers, params=params)
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="GitHub API authentication failed. Token may be expired or invalid."
                    )
                response.raise_for_status()
                return response.json()
        
        elif self.mcp_tool_name == "pull_request_read:get":
            owner = arguments.get("owner")
            repo = arguments.get("repo")
            pull_number = arguments.get("pullNumber") or arguments.get("pull_number")
            method = arguments.get("method", "get")
            
            if not owner or not repo or not pull_number:
                raise HTTPException(
                    status_code=400,
                    detail="owner, repo, and pullNumber are required"
                )
            
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url, headers=headers)
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="GitHub API authentication failed. Token may be expired or invalid."
                    )
                response.raise_for_status()
                return response.json()
        
        else:
            # For other tools, try to use MCP client if available
            # Otherwise, raise not implemented
            raise HTTPException(
                status_code=501,
                detail=f"GitHub tool '{self.mcp_tool_name}' not yet implemented via direct API calls"
            )


def register_github_tools():
    """Load and register GitHub tools from MCP server."""
    from shared.config import config
    
    # Always register fallback tools (even when MCP server is not configured)
    # These are common GitHub tools that we implement directly
    common_tools = [
        GitHubMCPTool(
            mcp_tool_name="list_pull_requests",
            description="List pull requests for a repository",
            parameters_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "state": {"type": "string", "description": "PR state (open, closed, all)", "default": "open"}
                },
                "required": ["owner", "repo"]
            },
            integration_type="pull_request"
        ),
        GitHubMCPTool(
            mcp_tool_name="pull_request_read:get",
            description="Get pull request details",
            parameters_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pullNumber": {"type": "integer", "description": "Pull request number"},
                    "method": {"type": "string", "description": "Method (get, get_diff, etc.)", "default": "get"}
                },
                "required": ["owner", "repo", "pullNumber"]
            },
            integration_type="pull_request"
        ),
    ]
    
    for tool in common_tools:
        try:
            register_tool(tool)
            logger.info(f"Registered GitHub tool (fallback): {tool.name}")
        except Exception as e:
            logger.warning(f"Failed to register fallback GitHub tool {tool.name}: {e}")
    
    # Try to load additional tools from MCP server if configured
    try:
        if not config.GITHUB_MCP_SERVER_URL:
            logger.info("GitHub MCP server URL not configured, using fallback tools only")
            return
        
        # Load tools from GitHub MCP server
        tools = _load_tools("github")
        
        if not tools:
            logger.warning("No tools loaded from GitHub MCP server, using fallback tools only")
            return
        
        logger.info(f"Loaded {len(tools)} tools from GitHub MCP server")
        
        # Register each tool as a BaseTool wrapper
        for tool in tools:
            try:
                # Extract tool metadata
                tool_name = tool.name
                tool_description = tool.description or f"GitHub tool: {tool_name}"
                
                # Get parameter schema from tool if available
                parameters_schema = None
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    # Convert Pydantic model to JSON schema
                    try:
                        parameters_schema = tool.args_schema.model_json_schema()
                    except Exception as e:
                        logger.warning(f"Failed to extract schema for {tool_name}: {e}")
                
                # Create wrapper
                wrapper = GitHubMCPTool(
                    mcp_tool_name=tool_name,
                    description=tool_description,
                    parameters_schema=parameters_schema
                )
                
                # Register tool (may overwrite fallback tool with same name)
                register_tool(wrapper)
                logger.info(f"Registered GitHub tool (MCP): {wrapper.name}")
                
            except Exception as e:
                logger.error(f"Failed to register GitHub tool {tool.name}: {e}")
                continue
        
    except Exception as e:
        logger.warning(f"Failed to load GitHub tools from MCP server: {e}. Using fallback tools only.")

