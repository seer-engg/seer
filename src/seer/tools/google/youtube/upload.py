"""
YouTube upload operations - video uploads.
"""

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.youtube.helpers import (
    VIDEO_SCHEMA,
    YOUTUBE_INTEGRATION_TYPE,
    YOUTUBE_UPLOAD_SCOPE,
    YOUTUBE_UPLOAD_URL,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.youtube.upload")


class YouTubeUploadVideoTool(GoogleAPIClient):
    """Upload a video to YouTube.

    Uses YouTube's resumable upload protocol for reliable uploads.
    Videos are uploaded as private by default and can be made public/unlisted after upload.
    """

    name = "youtube_upload_video"
    description = (
        "Upload a video to YouTube. Supports setting title, description, tags, category, "
        "and privacy status. Videos are uploaded as private by default."
    )
    required_scopes = YOUTUBE_UPLOAD_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    # Override default timeout for large uploads
    default_timeout: float = 300.0  # 5 minutes for upload operations

    def get_output_schema(self) -> Dict[str, Any]:
        return VIDEO_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Video title (required, max 100 characters)",
                    "maxLength": 100,
                },
                "description": {
                    "type": "string",
                    "description": "Video description (max 5000 characters)",
                    "maxLength": 5000,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Video tags for search optimization",
                },
                "category_id": {
                    "type": "string",
                    "description": "YouTube video category ID (default: 22 for 'People & Blogs')",
                    "default": "22",
                },
                "privacy_status": {
                    "type": "string",
                    "description": "Video privacy status",
                    "enum": ["private", "public", "unlisted"],
                    "default": "private",
                },
                "made_for_kids": {
                    "type": "boolean",
                    "description": "Whether the video is made for kids (COPPA compliance)",
                    "default": False,
                },
                "default_language": {
                    "type": "string",
                    "description": "Default language (ISO 639-1 code, e.g., 'en')",
                },
                "embeddable": {
                    "type": "boolean",
                    "description": "Whether the video can be embedded on other websites",
                    "default": True,
                },
                "license": {
                    "type": "string",
                    "description": "Video license type",
                    "enum": ["youtube", "creativeCommon"],
                    "default": "youtube",
                },
                "notify_subscribers": {
                    "type": "boolean",
                    "description": "Whether to notify channel subscribers about the upload",
                    "default": True,
                },
                "video_url": {
                    "type": "string",
                    "description": "URL of the video file to upload (must be publicly accessible)",
                },
                "video_base64": {
                    "type": "string",
                    "description": "Base64-encoded video content (alternative to video_url)",
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME type of the video (e.g., 'video/mp4', 'video/quicktime')",
                    "default": "video/mp4",
                },
            },
            "required": ["title"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        title = arguments.get("title")
        if not title:
            raise ValueError("title is required")

        video_url = arguments.get("video_url")
        video_base64 = arguments.get("video_base64")

        if not video_url and not video_base64:
            raise ValueError("Either video_url or video_base64 is required")

        # Build the video metadata
        snippet: Dict[str, Any] = {
            "title": title[:100],  # YouTube limits to 100 chars
            "categoryId": arguments.get("category_id", "22"),
        }

        if arguments.get("description"):
            snippet["description"] = arguments["description"][:5000]
        if arguments.get("tags"):
            snippet["tags"] = arguments["tags"]
        if arguments.get("default_language"):
            snippet["defaultLanguage"] = arguments["default_language"]

        status: Dict[str, Any] = {
            "privacyStatus": arguments.get("privacy_status", "private"),
            "selfDeclaredMadeForKids": arguments.get("made_for_kids", False),
            "embeddable": arguments.get("embeddable", True),
            "license": arguments.get("license", "youtube"),
        }

        metadata = {
            "snippet": snippet,
            "status": status,
        }

        # Get video content
        content_type = arguments.get("content_type", "video/mp4")

        if video_base64:
            try:
                video_content = base64.b64decode(video_base64)
            except Exception as exc:
                raise ValueError(f"Invalid base64 video content: {exc}") from exc
        elif video_url:
            # Download from URL
            video_content = await self._download_video(video_url)
        else:
            raise ValueError("Either video_url or video_base64 is required")

        logger.info(
            "Uploading YouTube video: title=%s, privacy=%s, size=%d bytes",
            title,
            status["privacyStatus"],
            len(video_content),
        )

        # Step 1: Initialize resumable upload
        upload_url = await self._init_resumable_upload(
            access_token,
            metadata,
            content_type=content_type,
            content_length=len(video_content),
            notify_subscribers=arguments.get("notify_subscribers", True),
        )

        # Step 2: Upload the video content
        result = await self._upload_video_content(
            upload_url,
            video_content,
            content_type,
        )

        return result

    async def _download_video(self, url: str) -> bytes:
        """Download video from URL."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download video from URL: HTTP {exc.response.status_code}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download video from URL: {exc}",
            ) from exc

    async def _init_resumable_upload(
        self,
        access_token: Optional[str],
        metadata: Dict[str, Any],
        *,
        content_type: str,
        content_length: int,
        notify_subscribers: bool,
    ) -> str:
        """Initialize a resumable upload session and return the upload URL."""
        token = self._validate_token(access_token)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": content_type,
            "X-Upload-Content-Length": str(content_length),
        }

        params = {
            "part": "snippet,status",
            "uploadType": "resumable",
            "notifySubscribers": str(notify_subscribers).lower(),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    YOUTUBE_UPLOAD_URL,
                    headers=headers,
                    params=params,
                    json=metadata,
                )

                if resp.status_code != 200:
                    body = resp.text[:500]
                    logger.error(
                        "YouTube resumable upload init failed: status=%s, body=%s",
                        resp.status_code,
                        body,
                    )
                    raise self._handle_api_error(resp)

                # The upload URL is in the Location header
                upload_url = resp.headers.get("Location")
                if not upload_url:
                    raise HTTPException(
                        status_code=500,
                        detail="YouTube did not return upload URL in response",
                    )

                return upload_url

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize YouTube upload: {exc}",
            ) from exc

    async def _upload_video_content(
        self,
        upload_url: str,
        content: bytes,
        content_type: str,
    ) -> Dict[str, Any]:
        """Upload video content to the resumable upload URL."""
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
        }

        try:
            async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                resp = await client.put(
                    upload_url,
                    headers=headers,
                    content=content,
                )

                if resp.status_code not in (200, 201):
                    body = resp.text[:500]
                    logger.error(
                        "YouTube video upload failed: status=%s, body=%s",
                        resp.status_code,
                        body,
                    )
                    raise self._handle_api_error(resp)

                return resp.json()

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"YouTube upload timed out after {self.default_timeout}s",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"YouTube upload failed: {exc}",
            ) from exc
