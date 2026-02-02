"""
LinkedIn Tool Wrappers

LinkedIn tools that call the LinkedIn API directly.
"""
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, register_tool

logger = get_logger("shared.tools.linkedin")

# Tool-to-scope mapping
LINKEDIN_TOOL_SCOPES: Dict[str, list[str]] = {
    "get_profile": ["openid", "profile", "email"],
    "create_post": ["w_member_social"],
    "default": ["openid", "profile", "email"],
}


# pylint: disable=broad-exception-caught
# Reason: LinkedIn API errors should be caught and converted to HTTP exceptions
class LinkedInTool(BaseTool):
    """LinkedIn tool that calls LinkedIn API directly."""

    def __init__(
        self, tool_name: str, description: str,
        parameters_schema: Optional[Dict[str, Any]] = None, integration_type: Optional[str] = "linkedin"
    ):
        self.tool_name = tool_name
        self.name = f"linkedin_{tool_name.replace(':', '_')}"
        self.description = description
        self.required_scopes = LINKEDIN_TOOL_SCOPES.get(
            tool_name,
            LINKEDIN_TOOL_SCOPES.get("default", ["openid", "profile", "email"])
        )
        self.integration_type = integration_type
        self.provider = "linkedin"
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
        Execute LinkedIn tool by calling LinkedIn API directly.

        Args:
            access_token: OAuth access token (required for LinkedIn API)
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=f"LinkedIn tool '{self.tool_name}' requires OAuth access token"
            )

        try:
            result = await self._execute_via_linkedin_api(access_token, arguments)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LinkedIn tool execution failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"LinkedIn tool execution failed: {str(e)}"
            ) from e

    async def _execute_via_linkedin_api(self, access_token: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool by calling LinkedIn API directly.
        """
        if self.tool_name == "get_profile":
            # Get authenticated user's profile
            url = "https://api.linkedin.com/v2/userinfo"
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url, headers=headers, timeout=10.0)
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="LinkedIn API authentication failed. Token may be expired or invalid."
                    )
                response.raise_for_status()
                return response.json()

        elif self.tool_name == "create_post":
            # Create a post on LinkedIn
            text = arguments.get("text")
            visibility = arguments.get("visibility", "PUBLIC")  # PUBLIC or CONNECTIONS

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="text is required"
                )

            # First, get the user's person URN
            userinfo_url = "https://api.linkedin.com/v2/userinfo"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "LinkedIn-Version": "202401",
            }

            async with httpx.AsyncClient() as http_client:
                # Get user info to get the person URN
                userinfo_response = await http_client.get(userinfo_url, headers=headers, timeout=10.0)
                if userinfo_response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="LinkedIn API authentication failed. Token may be expired or invalid."
                    )
                userinfo_response.raise_for_status()
                user_data = userinfo_response.json()
                person_urn = user_data.get("sub")

                if not person_urn:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to get user person URN from LinkedIn"
                    )

                # Create the post using UGC Post API
                post_url = "https://api.linkedin.com/v2/ugcPosts"
                post_data = {
                    "author": f"urn:li:person:{person_urn}",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {
                                "text": text
                            },
                            "shareMediaCategory": "NONE"
                        }
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": visibility
                    }
                }

                post_response = await http_client.post(
                    post_url,
                    headers=headers,
                    json=post_data,
                    timeout=15.0
                )

                if post_response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="LinkedIn API authentication failed. Token may be expired or invalid."
                    )
                if post_response.status_code == 403:
                    raise HTTPException(
                        status_code=403,
                        detail="LinkedIn API permission denied. Ensure 'w_member_social' scope is granted."
                    )
                post_response.raise_for_status()
                return post_response.json()

        else:
            raise HTTPException(
                status_code=501,
                detail=f"LinkedIn tool '{self.tool_name}' not yet implemented"
            )


def register_linkedin_tools():
    """Register LinkedIn tools that call LinkedIn API directly."""
    common_tools = [
        LinkedInTool(
            tool_name="get_profile",
            description="Get the authenticated user's LinkedIn profile information",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            integration_type="linkedin"
        ),
        LinkedInTool(
            tool_name="create_post",
            description="Create a post on LinkedIn on behalf of the authenticated user",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content of the post"
                    },
                    "visibility": {
                        "type": "string",
                        "description": "Post visibility (PUBLIC or CONNECTIONS)",
                        "enum": ["PUBLIC", "CONNECTIONS"],
                        "default": "PUBLIC"
                    }
                },
                "required": ["text"]
            },
            integration_type="linkedin"
        ),
    ]

    for tool in common_tools:
        try:
            register_tool(tool)
            logger.debug("Registered LinkedIn tool: %s", tool.name)
        except Exception:
            logger.warning("Failed to register LinkedIn tool %s", tool.name)
