"""
LinkedIn Tool Wrappers

LinkedIn tools that call the LinkedIn API directly.

Supports:
- get_profile: Fetch authenticated user's profile
- create_post: Create posts with optional image attachments (up to 9 images)

Image Upload Flow:
1. POST /rest/images?action=initializeUpload → get uploadUrl + image URN
2. PUT binary image bytes to uploadUrl
3. Reference image URN in post's content.media.id or content.multiImage.images[]
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.core.files.resolver import resolve_file_input
from seer.core.files.schemas import get_file_array_input_property
from seer.logger import get_logger
from seer.tools.base import BaseTool, register_tool

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.linkedin")

# LinkedIn API version for /rest/ endpoints (YYYYMM format)
# Note: LinkedIn regularly sunsets older versions. Check https://learn.microsoft.com/en-us/linkedin/marketing/versioning
# for currently active versions if you encounter 426 errors.
LINKEDIN_API_VERSION = "202503"

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

    async def execute(  # pylint: disable=unused-argument  # Reason: credentials accepted for executor compatibility but access_token is used directly
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        """
        Execute LinkedIn tool by calling LinkedIn API directly.

        Args:
            access_token: OAuth access token (required for LinkedIn API)
            arguments: Tool arguments
            context: Workflow runtime context (required for file resolution when using images)

        Returns:
            Tool execution result
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=f"LinkedIn tool '{self.tool_name}' requires OAuth access token"
            )

        try:
            result = await self._execute_via_linkedin_api(access_token, arguments, context)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LinkedIn tool execution failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"LinkedIn tool execution failed: {str(e)}"
            ) from e

    async def _execute_via_linkedin_api(
        self,
        access_token: str,
        arguments: Dict[str, Any],
        context: Optional["WorkflowRuntimeContext"],
    ) -> Any:
        """
        Execute tool by calling LinkedIn API directly.

        Args:
            access_token: OAuth access token
            arguments: Tool arguments
            context: Workflow runtime context for file resolution
        """
        if self.tool_name == "get_profile":
            return await self._execute_get_profile(access_token)
        if self.tool_name == "create_post":
            return await self._execute_create_post(access_token, arguments, context)
        raise HTTPException(
            status_code=501,
            detail=f"LinkedIn tool '{self.tool_name}' not yet implemented"
        )

    async def _execute_get_profile(self, access_token: str) -> dict:
        """Get authenticated user's LinkedIn profile."""
        url = "https://api.linkedin.com/v2/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="LinkedIn API authentication failed. Token may be expired or invalid."
                )
            response.raise_for_status()
            return response.json()

    async def _execute_create_post(
        self,
        access_token: str,
        arguments: Dict[str, Any],
        context: Optional["WorkflowRuntimeContext"],
    ) -> dict:
        """Create a post on LinkedIn (with optional images)."""
        text = arguments.get("text")
        visibility = arguments.get("visibility", "PUBLIC")
        images = arguments.get("images", [])

        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        if len(images) > 9:
            raise HTTPException(status_code=400, detail="Maximum 9 images allowed per post")

        person_urn = await self._get_person_urn(access_token)

        async with httpx.AsyncClient() as http_client:
            if images:
                image_urns = await self._resolve_and_upload_images(
                    http_client, access_token, person_urn, images, context=context
                )
                return await self._create_post_with_images(
                    http_client,
                    access_token=access_token,
                    person_urn=person_urn,
                    text=text,
                    visibility=visibility,
                    image_urns=image_urns,
                )

            return await self._create_text_only_post(
                http_client, access_token=access_token, person_urn=person_urn, text=text, visibility=visibility
            )

    async def _get_person_urn(self, access_token: str) -> str:
        """Fetch and return the authenticated user's person URN."""
        url = "https://api.linkedin.com/v2/userinfo"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
        }

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="LinkedIn API authentication failed. Token may be expired or invalid."
                )
            response.raise_for_status()
            person_urn = response.json().get("sub")

        if not person_urn:
            raise HTTPException(status_code=500, detail="Failed to get user person URN from LinkedIn")
        return person_urn

    async def _resolve_file_to_bytes(
        self, image_input: Any, context: Optional["WorkflowRuntimeContext"]
    ) -> bytes:
        """Resolve a file input to raw bytes using the unified file resolver."""
        content, _mime_type, _filename = await resolve_file_input(image_input, context)
        return content

    async def _resolve_and_upload_images(
        self,
        http_client: httpx.AsyncClient,
        access_token: str,
        person_urn: str,
        images: List[Any],
        *,
        context: Optional["WorkflowRuntimeContext"],
    ) -> List[str]:
        """Resolve file inputs and upload images to LinkedIn."""
        image_urns: List[str] = []
        for i, image_input in enumerate(images):
            try:
                image_bytes = await self._resolve_file_to_bytes(image_input, context)
                logger.debug("Resolved image %d: %d bytes", i, len(image_bytes))

                image_urn = await self._upload_image(http_client, access_token, person_urn, image_bytes)
                image_urns.append(image_urn)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid image input {i}: {e}") from e
        return image_urns

    async def _create_text_only_post(
        self,
        http_client: httpx.AsyncClient,
        *,
        access_token: str,
        person_urn: str,
        text: str,
        visibility: str,
    ) -> dict:
        """Create a text-only post using legacy UGC Post API."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
        }
        post_data = {
            "author": f"urn:li:person:{person_urn}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": visibility}
        }

        response = await http_client.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers=headers,
            json=post_data,
            timeout=15.0
        )

        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="LinkedIn API authentication failed. Token may be expired or invalid."
            )
        if response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="LinkedIn API permission denied. Ensure 'w_member_social' scope is granted."
            )
        response.raise_for_status()
        return response.json()

    async def _upload_image(
        self,
        http_client: httpx.AsyncClient,
        access_token: str,
        person_urn: str,
        image_bytes: bytes,
    ) -> str:
        """
        Upload an image to LinkedIn and return the image URN.

        Uses the LinkedIn Images API:
        1. Initialize upload to get uploadUrl and image URN
        2. Upload binary image data to the uploadUrl
        3. Return the image URN for use in posts

        Args:
            http_client: HTTP client for requests
            access_token: OAuth access token
            person_urn: User's person URN (from /v2/userinfo sub field)
            image_bytes: Raw image bytes to upload

        Returns:
            Image URN (e.g., "urn:li:image:C4E10AQFoyyAjHPMQuQ")

        Raises:
            HTTPException: If upload fails
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
        }

        # Step 1: Initialize upload
        init_url = "https://api.linkedin.com/rest/images?action=initializeUpload"
        init_payload = {
            "initializeUploadRequest": {
                "owner": f"urn:li:person:{person_urn}"
            }
        }

        init_response = await http_client.post(
            init_url,
            headers=headers,
            json=init_payload,
            timeout=15.0
        )

        if init_response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="LinkedIn API authentication failed. Token may be expired or invalid."
            )
        if init_response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="LinkedIn API permission denied for image upload. Ensure 'w_member_social' scope is granted."
            )
        init_response.raise_for_status()

        init_data = init_response.json()
        upload_url = init_data.get("value", {}).get("uploadUrl")
        image_urn = init_data.get("value", {}).get("image")

        if not upload_url or not image_urn:
            raise HTTPException(
                status_code=500,
                detail="LinkedIn API did not return upload URL or image URN"
            )

        # Step 2: Upload binary image data
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream",
        }

        upload_response = await http_client.put(
            upload_url,
            headers=upload_headers,
            content=image_bytes,
            timeout=60.0  # Longer timeout for large images
        )

        if upload_response.status_code not in (200, 201):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload image to LinkedIn: {upload_response.status_code}"
            )

        logger.debug("Successfully uploaded image to LinkedIn: %s", image_urn)
        return image_urn

    async def _create_post_with_images(
        self,
        http_client: httpx.AsyncClient,
        *,
        access_token: str,
        person_urn: str,
        text: str,
        visibility: str,
        image_urns: List[str],
    ) -> dict:
        """
        Create a LinkedIn post with images using the Posts API.

        Uses:
        - Single image: content.media format
        - Multiple images: content.multiImage format (up to 9 images)

        Args:
            http_client: HTTP client for requests
            access_token: OAuth access token
            person_urn: User's person URN
            text: Post text content
            visibility: "PUBLIC" or "CONNECTIONS"
            image_urns: List of image URNs from _upload_image()

        Returns:
            API response dict with post details

        Raises:
            HTTPException: If post creation fails
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
        }

        post_url = "https://api.linkedin.com/rest/posts"

        # Build post payload
        post_data: Dict[str, Any] = {
            "author": f"urn:li:person:{person_urn}",
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": []
            },
            "lifecycleState": "PUBLISHED",
        }

        # Add media content based on number of images
        if len(image_urns) == 1:
            # Single image format
            post_data["content"] = {
                "media": {
                    "id": image_urns[0]
                }
            }
        else:
            # MultiImage format (2-9 images)
            post_data["content"] = {
                "multiImage": {
                    "images": [{"id": urn} for urn in image_urns]
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

        # LinkedIn Posts API returns 201 with post ID in x-restli-id header (body may be empty)
        if post_response.status_code == 201:
            post_id = post_response.headers.get("x-restli-id")
            # Try to parse JSON body if present, otherwise return header-based response
            try:
                body = post_response.json() if post_response.content else {}
            except Exception:
                body = {}
            return {"id": post_id, "status": "created", **body}

        # For other success codes, try to return JSON body
        try:
            return post_response.json()
        except Exception:
            return {"status": "success", "status_code": post_response.status_code}


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
            description="Create a post on LinkedIn on behalf of the authenticated user. Supports optional image attachments (up to 9 images).",
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
                    },
                    "images": get_file_array_input_property(
                        description="Images to attach to the post (max 9). Select files from workflow outputs or user storage.",
                        min_items=0,
                        max_items=9,
                    )
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
