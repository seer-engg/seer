"""
GitHub Tool Wrappers

GitHub tools that call the GitHub API directly.
"""
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, register_tool

logger = get_logger("shared.tools.github")

# Tool-to-scope mapping (matches frontend)
GITHUB_TOOL_SCOPES: Dict[str, list[str]] = {
    "list_pull_requests": ["repo"],
    "pull_request_read:get": ["repo"],
    "default": ["repo"],
}


class GitHubTool(BaseTool):
    """GitHub tool that calls GitHub API directly."""

    def __init__(self, tool_name: str, description: str, parameters_schema: Optional[Dict[str, Any]] = None, integration_type: Optional[str] = "github"):
        self.tool_name = tool_name
        self.name = f"github_{tool_name.replace(':', '_')}"
        self.description = description
        self.required_scopes = GITHUB_TOOL_SCOPES.get(
            tool_name,
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
        Execute GitHub tool by calling GitHub API directly.

        Args:
            access_token: OAuth access token (required for GitHub API)
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=f"GitHub tool '{self.tool_name}' requires OAuth access token"
            )

        try:
            result = await self._execute_via_github_api(access_token, arguments)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("GitHub tool execution failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"GitHub tool execution failed: {str(e)}"
            )

    async def _execute_via_github_api(self, access_token: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool by calling GitHub API directly.
        """
        if self.tool_name == "list_pull_requests":
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

        elif self.tool_name == "pull_request_read:get":
            owner = arguments.get("owner")
            repo = arguments.get("repo")
            pull_number = arguments.get("pullNumber") or arguments.get("pull_number")
            arguments.get("method", "get")

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
            raise HTTPException(
                status_code=501,
                detail=f"GitHub tool '{self.tool_name}' not yet implemented"
            )


def register_github_tools():
    """Register GitHub tools that call GitHub API directly."""
    common_tools = [
        GitHubTool(
            tool_name="list_pull_requests",
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
        GitHubTool(
            tool_name="pull_request_read:get",
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
            logger.debug("Registered GitHub tool: %s", tool.name)
        except Exception:
            logger.warning("Failed to register GitHub tool %s", tool.name)
