"""
Google Drive file operations - list, get, download, upload, update.
"""

import base64
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gdrive.helpers import (
    _drive_file_list_schema,
    _drive_file_schema,
    _encode_multipart_related,
)

logger = get_logger("shared.tools.gdrive.files")


class GoogleDriveListFilesTool(GoogleAPIClient):
    """List/search Google Drive files with query support."""

    name = "google_drive_list_files"
    description = "List/search Google Drive files. Supports Drive query 'q' and pagination."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Drive search query",
                    "default": "trashed=false"
                },
                "page_size": {"type": "integer", "default": 100},
                "page_token": {"type": "string"},
                "fields": {
                    "type": "string",
                    "default": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed)"
                },
                "spaces": {"type": "string", "default": "drive"},
                "order_by": {"type": "string"},
                "corpora": {"type": "string", "enum": ["user", "domain", "drive", "allDrives"], "default": "user"},
                "drive_id": {"type": "string"},
                "include_items_from_all_drives": {"type": "boolean", "default": False},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": []
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_list_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        params: Dict[str, Any] = {
            "q": arguments.get("q", "trashed=false"),
            "pageSize": arguments.get("page_size", 100),
            "fields": arguments.get("fields", "nextPageToken,files(id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed)"),
            "spaces": arguments.get("spaces", "drive"),
            "corpora": arguments.get("corpora", "user"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
            "includeItemsFromAllDrives": arguments.get("include_items_from_all_drives", False),
        }

        if arguments.get("page_token"):
            params["pageToken"] = arguments["page_token"]
        if arguments.get("order_by"):
            params["orderBy"] = arguments["order_by"]
        if arguments.get("drive_id"):
            params["driveId"] = arguments["drive_id"]

        logger.info("Listing Drive files q=%r", params.get("q"))

        resp = await self._make_request(
            "GET",
            "https://www.googleapis.com/drive/v3/files",
            access_token,
            params=params
        )
        return resp.json()


class GoogleDriveGetFileMetadataTool(GoogleAPIClient):
    """Get Google Drive file metadata by ID."""

    name = "google_drive_get_file_metadata"
    description = "Get metadata for a Google Drive file by ID."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "fields": {
                    "type": "string",
                    "description": "Fields to retrieve",
                    "default": "id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed,owners"
                },
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        params = {
            "fields": arguments.get("fields", "id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed,owners"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
        }

        resp = await self._make_request(
            "GET",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            access_token,
            params=params
        )
        return resp.json()


class GoogleDriveDownloadFileTool(GoogleAPIClient):
    """Download Google Drive file content."""

    name = "google_drive_download_file"
    description = "Download Google Drive file content (binary data returned as base64)."
    required_scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "mime_type": {
                    "type": "string",
                    "description": "For Google Docs export, specify target MIME type"
                },
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content_base64": {"type": "string", "description": "File content as base64"},
                "size_bytes": {"type": "integer"},
            }
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        mime_type = arguments.get("mime_type")
        supports_all_drives = arguments.get("supports_all_drives", True)

        # Determine if export or download
        if mime_type:
            # Google Docs export
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            params = {"mimeType": mime_type}
        else:
            # Regular download
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": supports_all_drives}

        resp = await self._make_request("GET", url, access_token, params=params)

        content = resp.content
        return {
            "content_base64": base64.b64encode(content).decode("utf-8"),
            "size_bytes": len(content)
        }


class GoogleDriveUploadFileTool(GoogleAPIClient):
    """Upload file to Google Drive."""

    name = "google_drive_upload_file"
    description = "Upload a file to Google Drive using multipart upload."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File name"},
                "content_base64": {"type": "string", "description": "File content as base64"},
                "mime_type": {"type": "string", "description": "File MIME type", "default": "application/octet-stream"},
                "parent_folder_id": {"type": "string", "description": "Parent folder ID"},
                "description": {"type": "string"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["name", "content_base64"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        name = arguments.get("name")
        content_b64 = arguments.get("content_base64")

        if not name or not content_b64:
            raise HTTPException(status_code=400, detail="name and content_base64 are required")

        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")

        mime_type = arguments.get("mime_type", "application/octet-stream")
        metadata = {"name": name, "mimeType": mime_type}

        if arguments.get("parent_folder_id"):
            metadata["parents"] = [arguments["parent_folder_id"]]
        if arguments.get("description"):
            metadata["description"] = arguments["description"]

        # Build multipart body
        multipart = _encode_multipart_related(metadata, content_bytes, mime_type)

        params = {
            "uploadType": "multipart",
            "supportsAllDrives": arguments.get("supports_all_drives", True),
        }

        # Use _make_request but with custom content-type header
        token = self._validate_token(access_token)
        headers = self._build_headers(token)
        headers["Content-Type"] = multipart["content_type"]

        # Make request directly with httpx for custom content
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://www.googleapis.com/upload/drive/v3/files",
                headers=headers,
                params=params,
                content=multipart["body"]
            )
            if resp.is_error:
                raise self._handle_api_error(resp)

        return resp.json()


class GoogleDriveUpdateFileTool(GoogleAPIClient):
    """Update Google Drive file metadata and/or content."""

    name = "google_drive_update_file"
    description = "Update Google Drive file metadata and/or content."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "File ID to update"},
                "name": {"type": "string", "description": "New file name"},
                "description": {"type": "string"},
                "mime_type": {"type": "string"},
                "add_parents": {"type": "array", "items": {"type": "string"}, "description": "Parent folder IDs to add"},
                "remove_parents": {"type": "array", "items": {"type": "string"}, "description": "Parent folder IDs to remove"},
                "content_base64": {"type": "string", "description": "New file content as base64"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    def _build_update_metadata(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata dict from arguments."""
        metadata = {}
        if arguments.get("name"):
            metadata["name"] = arguments["name"]
        if arguments.get("description"):
            metadata["description"] = arguments["description"]
        if arguments.get("mime_type"):
            metadata["mimeType"] = arguments["mime_type"]
        return metadata

    def _build_update_params(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Build query params from arguments."""
        params = {"supportsAllDrives": arguments.get("supports_all_drives", True)}
        if arguments.get("add_parents"):
            params["addParents"] = ",".join(arguments["add_parents"])
        if arguments.get("remove_parents"):
            params["removeParents"] = ",".join(arguments["remove_parents"])
        return params

    async def _update_with_content(
        self, file_id: str, access_token: str, *, metadata: Dict[str, Any],
        params: Dict[str, Any], content_b64: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle multipart file content update."""
        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {str(e)}")

        mime_type = arguments.get("mime_type", "application/octet-stream")
        multipart = _encode_multipart_related(metadata, content_bytes, mime_type)
        params["uploadType"] = "multipart"

        token = self._validate_token(access_token)
        headers = self._build_headers(token)
        headers["Content-Type"] = multipart["content_type"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.patch(
                f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
                headers=headers,
                params=params,
                content=multipart["body"]
            )
            if resp.is_error:
                raise self._handle_api_error(resp)
        return resp.json()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        metadata = self._build_update_metadata(arguments)
        params = self._build_update_params(arguments)

        content_b64 = arguments.get("content_base64")
        if content_b64:
            return await self._update_with_content(
                file_id, access_token, metadata=metadata, params=params,
                content_b64=content_b64, arguments=arguments
            )

        # Metadata-only update
        resp = await self._make_request(
            "PATCH",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            access_token,
            params=params,
            json_body=metadata
        )
        return resp.json()
