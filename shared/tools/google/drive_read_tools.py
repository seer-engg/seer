"""
Google Drive read-only tools: list, get metadata, download, about.get
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import base64
from fastapi import HTTPException

from shared.tools.base import BaseTool
from shared.logger import get_logger
from shared.tools.google._common import (
    _require_access_token,
    _drive_request,
    _drive_file_list_schema,
    _drive_file_schema,
    _drive_about_schema,
    get_google_drive_common_attributes,
    get_file_id_resource_picker,
    get_file_id_parameter_schema,
)

logger = get_logger("shared.tools.google_drive.read")


class GoogleDriveListFilesTool(BaseTool):
    """
    List/search Google Drive files with support for Drive query 'q' and pagination.
    """
    name = "google_drive_list_files"
    description = "List/search Google Drive files. Supports Drive query 'q' and pagination."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Drive search query", "default": "trashed=false"},
                "page_size": {"type": "integer", "default": 100},
                "page_token": {"type": "string"},
                "fields": {"type": "string", "default": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed)"},
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
        _require_access_token(access_token, "Google Drive list files tool")

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {"Authorization": f"Bearer {access_token}"}

        params: Dict[str, Any] = {
            "q": arguments.get("q", "trashed=false"),
            "pageSize": arguments.get("page_size", 100),
            "fields": arguments.get("fields"),
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

        try:
            logger.info("Listing Drive files q=%r", params.get("q"))
            resp = await _drive_request("get", url, headers=headers, params=params, timeout=30.0, prefix="Google Drive files.list")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in files.list: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error listing Drive files: {str(exc)}") from exc


class GoogleDriveGetFileMetadataTool(BaseTool):
    """
    Get Google Drive file metadata by file_id.
    """
    name = "google_drive_get_file_metadata"
    description = "Get Google Drive file metadata by file_id."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

    def __init__(self):
        super().__init__()
        attrs = get_google_drive_common_attributes()
        self.integration_type = attrs["integration_type"]
        self.provider = attrs["provider"]

    def get_resource_pickers(self) -> Dict[str, Any]:
        return get_file_id_resource_picker()

    def get_parameters_schema(self) -> Dict[str, Any]:
        schema = get_file_id_parameter_schema()
        schema["properties"].update({
            "fields": {"type": "string", "default": "id,name,mimeType,parents,modifiedTime,createdTime,size,webViewLink,webContentLink,trashed,owners(displayName,emailAddress),driveId"},
            "supports_all_drives": {"type": "boolean", "default": True},
        })
        schema["required"] = ["file_id"]
        return schema

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive get metadata tool")
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"fields": arguments.get("fields"), "supportsAllDrives": arguments.get("supports_all_drives", True)}

        try:
            logger.info("Getting Drive file metadata file_id=%s", file_id)
            resp = await _drive_request("get", url, headers=headers, params=params, timeout=30.0, prefix="Google Drive files.get metadata")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in files.get metadata: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error getting Drive metadata: {str(exc)}") from exc


class GoogleDriveDownloadFileTool(BaseTool):
    """
    Download a Drive file (returns base64). For Google Docs/Sheets/Slides, provide export_mime_type.
    """
    name = "google_drive_download_file"
    description = "Download a Drive file (returns base64). For Google Docs/Sheets/Slides, provide export_mime_type."
    required_scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    def __init__(self):
        super().__init__()
        attrs = get_google_drive_common_attributes()
        self.integration_type = attrs["integration_type"]
        self.provider = attrs["provider"]

    def get_resource_pickers(self) -> Dict[str, Any]:
        return get_file_id_resource_picker()

    def get_parameters_schema(self) -> Dict[str, Any]:
        schema = get_file_id_parameter_schema()
        schema["properties"].update({
            "export_mime_type": {"type": "string"},
            "acknowledge_abuse": {"type": "boolean", "default": False},
            "supports_all_drives": {"type": "boolean", "default": True},
            "include_metadata": {"type": "boolean", "default": True}
        })
        schema["required"] = ["file_id"]
        return schema

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "exported": {"type": "boolean"},
                "export_mime_type": {"type": ["string", "null"]},
                "metadata": {"anyOf": [_drive_file_schema(), {"type": "null"}]},
                "content_base64": {"type": "string"},
                "content_length": {"type": "integer"},
            },
            "required": ["file_id", "exported", "metadata", "content_base64", "content_length"],
            "additionalProperties": False,
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive download tool")
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        headers = {"Authorization": f"Bearer {access_token}"}
        supports_all_drives = arguments.get("supports_all_drives", True)
        include_metadata = arguments.get("include_metadata", True)

        metadata = None
        if include_metadata:
            meta_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            meta_params = {"fields": "id,name,mimeType,size,modifiedTime,parents,driveId", "supportsAllDrives": supports_all_drives}
            meta_resp = await _drive_request("get", meta_url, headers=headers, params=meta_params, timeout=30.0, prefix="Google Drive metadata fetch (pre-download)")
            metadata = meta_resp.json()

        export_mime_type = arguments.get("export_mime_type")
        is_export = bool(export_mime_type)

        if (not is_export) and metadata and str(metadata.get("mimeType", "")).startswith("application/vnd.google-apps."):
            raise HTTPException(
                status_code=400,
                detail="This file is a Google Workspace document. Provide export_mime_type (e.g., application/pdf) to download via files.export.",
            )

        if is_export:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            params = {"mimeType": export_mime_type}
            logger.info("Exporting Drive file file_id=%s mimeType=%s", file_id, export_mime_type)
        else:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            params = {"alt": "media", "acknowledgeAbuse": arguments.get("acknowledge_abuse", False), "supportsAllDrives": supports_all_drives}
            logger.info("Downloading Drive file (alt=media) file_id=%s", file_id)

        try:
            resp = await _drive_request("get", url, headers=headers, params=params, timeout=60.0, prefix="Google Drive download/export")
            content_b64 = base64.b64encode(resp.content).decode("utf-8")
            return {
                "file_id": file_id,
                "exported": is_export,
                "export_mime_type": export_mime_type if is_export else None,
                "metadata": metadata,
                "content_base64": content_b64,
                "content_length": len(resp.content),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected download/export error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error downloading/exporting Drive file: {str(exc)}") from exc


class GoogleDriveAboutGetTool(BaseTool):
    """
    Get information about the user, the user's Drive, and system capabilities (about.get).
    """
    name = "google_drive_about_get"
    description = "Get information about the user, the user's Drive, and system capabilities (about.get)."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"fields": {"type": "string", "default": "user(displayName,emailAddress),storageQuota,importFormats,exportFormats,maxUploadSize,canCreateDrives"}}}

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_about_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive about.get tool")
        url = "https://www.googleapis.com/drive/v3/about"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"fields": arguments.get("fields")}

        try:
            logger.info("Getting Drive about info")
            resp = await _drive_request("get", url, headers=headers, params=params, timeout=30.0, prefix="Google Drive about.get")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected about.get error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error getting Drive about info: {str(exc)}") from exc
