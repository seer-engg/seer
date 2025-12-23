"""
GitHub MCP Tool Wrappers

Wraps GitHub MCP server tools as Seer BaseTool instances with scope validation.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException
import httpx

from shared.tools.base import BaseTool, register_tool
from shared.tools.mcp_client import client, _load_tools
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
    
    def __init__(self, mcp_tool_name: str, description: str, parameters_schema: Optional[Dict[str, Any]] = None):
        self.mcp_tool_name = mcp_tool_name
        self.name = f"github_{mcp_tool_name.replace(':', '_')}"
        self.description = description
        self.required_scopes = GITHUB_TOOL_SCOPES.get(
            mcp_tool_name,
            GITHUB_TOOL_SCOPES.get("default", ["repo"])
        )
        self.integration_type = "github"
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
        Execute GitHub MCP tool via MCP client.
        
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
            # Execute via MCP client
            # Note: The GitHub MCP server may need the token passed differently
            # For now, we'll pass it in headers if the MCP client supports it
            # The actual implementation may need adjustment based on how GitHub MCP server handles auth
            
            # Try to get tools from GitHub MCP server
            tools = _load_tools("github")
            
            # Find the matching tool
            tool = None
            for t in tools:
                if t.name == self.mcp_tool_name:
                    tool = t
                    break
            
            if not tool:
                raise HTTPException(
                    status_code=404,
                    detail=f"GitHub MCP tool '{self.mcp_tool_name}' not found"
                )
            
            # Execute tool via MCP client
            # The MCP client's run_tool method should handle the execution
            # We may need to pass the access_token in a way the GitHub MCP server understands
            # For now, assume the MCP server can use the token from environment or headers
            
            # Note: The GitHub MCP server typically expects GITHUB_TOKEN env var
            # For OAuth tokens, we may need to modify how we call the MCP server
            # This is a limitation we'll need to address - either:
            # 1. Modify GitHub MCP server to accept tokens per-request
            # 2. Use a proxy/wrapper that injects tokens
            # 3. Call GitHub API directly instead of via MCP server
            
            # For POC, we'll call GitHub API directly if MCP server doesn't support per-request tokens
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
            }
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
            }
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

