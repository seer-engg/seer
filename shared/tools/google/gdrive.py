"""
Google Drive Tools

Tools for common Google Drive operations using Google Drive API v3.
- List/search files
- Get file metadata
- Download file content (alt=media) and export Google Docs (files.export)
- Upload files (multipart/media)
- Create folder
- Update file (rename/move, optional content update)
- Delete file
- Share / create permissions
- Get Drive 'about' info
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import base64
import json
import uuid

import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.google_drive")


# -----------------------------
# Helpers
# -----------------------------
def _require_access_token(access_token: Optional[str], tool_name: str) -> None:
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"{tool_name} requires OAuth access token"
        )


def _http_exception_from_response(resp: httpx.Response, prefix: str) -> HTTPException:
    # Keep message bounded
    body_snippet = resp.text[:800] if resp.text else ""
    if resp.status_code == 401:
        return HTTPException(status_code=401, detail=f"{prefix}: authentication failed. Token may be expired/invalid.")
    if resp.status_code == 403:
        return HTTPException(status_code=403, detail=f"{prefix}: permission denied. Ensure access + correct OAuth scopes.")
    return HTTPException(status_code=resp.status_code, detail=f"{prefix}: {body_snippet}")


def _encode_multipart_related(metadata: Dict[str, Any], content_bytes: bytes, content_type: str) -> Dict[str, Any]:
    """
    Build multipart/related payload for Drive uploadType=multipart.
    Returns: { "body": bytes, "content_type": "multipart/related; boundary=..." }
    """
    boundary = f"==============={uuid.uuid4().hex}=="
    meta_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    parts: List[bytes] = []
    parts.append(
        b"--" + boundary.encode("utf-8") + b"\r\n"
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        + meta_json + b"\r\n"
    )
    parts.append(
        b"--" + boundary.encode("utf-8") + b"\r\n"
        + f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        + content_bytes + b"\r\n"
    )
    parts.append(b"--" + boundary.encode("utf-8") + b"--\r\n")

    body = b"".join(parts)
    return {"body": body, "content_type": f"multipart/related; boundary={boundary}"}


# -----------------------------
# Tools
# -----------------------------
class GoogleDriveListFilesTool(BaseTool):
    """
    List/search files using Drive files.list.
    """

    name = "google_drive_list_files"
    description = "List/search Google Drive files. Supports Drive query 'q' and pagination."
    # files.list supports multiple scopes; pick the least-privileged common one for listing.
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Drive search query (e.g., \"name contains 'report' and trashed=false\")",
                    "default": "trashed=false"
                },
                "page_size": {
                    "type": "integer",
                    "description": "Max number of files to return (Drive API pageSize).",
                    "default": 100
                },
                "page_token": {
                    "type": "string",
                    "description": "Token to retrieve next page of results (from nextPageToken)."
                },
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,size,webViewLink,webContentLink,trashed)"
                },
                "spaces": {
                    "type": "string",
                    "description": "Comma-separated spaces to query (e.g. 'drive', 'appDataFolder').",
                    "default": "drive"
                },
                "order_by": {
                    "type": "string",
                    "description": "Comma-separated list of sort keys (Drive API orderBy)."
                },
                "corpora": {
                    "type": "string",
                    "description": "Corpora to query: user, domain, drive, allDrives.",
                    "enum": ["user", "domain", "drive", "allDrives"],
                    "default": "user"
                },
                "drive_id": {
                    "type": "string",
                    "description": "Shared drive ID if corpora=drive."
                },
                "include_items_from_all_drives": {
                    "type": "boolean",
                    "description": "Include both My Drive and shared drive items.",
                    "default": False
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": []
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive list files tool")

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {"Authorization": f"Bearer {access_token}"}

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

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Listing Drive files q={params.get('q')!r}")
                resp = await http_client.get(url, headers=headers, params=params)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive files.list error")
                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive files.list request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in files.list: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing Drive files: {str(e)}")


class GoogleDriveGetFileMetadataTool(BaseTool):
    """
    Get file metadata using Drive files.get (metadata).
    """

    name = "google_drive_get_file_metadata"
    description = "Get Google Drive file metadata by file_id."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "id,name,mimeType,parents,modifiedTime,createdTime,size,webViewLink,webContentLink,trashed,owners(displayName,emailAddress),driveId"
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": ["file_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive get metadata tool")

        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "fields": arguments.get("fields"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Getting Drive file metadata file_id={file_id}")
                resp = await http_client.get(url, headers=headers, params=params)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive files.get metadata error")
                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive files.get request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in files.get metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting Drive metadata: {str(e)}")


class GoogleDriveDownloadFileTool(BaseTool):
    """
    Download file bytes:
    - Non-Google-Docs blobs: files.get?alt=media
    - Google Docs/Sheets/Slides: files.export (requires export_mime_type)
    """

    name = "google_drive_download_file"
    description = "Download a Drive file (returns base64). For Google Docs/Sheets/Slides, provide export_mime_type."
    required_scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "export_mime_type": {
                    "type": "string",
                    "description": "If set, uses files.export to export Google Workspace docs (e.g. application/pdf)."
                },
                "acknowledge_abuse": {
                    "type": "boolean",
                    "description": "Acknowledge risk of downloading malware/abusive content (only for alt=media downloads).",
                    "default": False
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "If true, fetch metadata first and include it in response.",
                    "default": True
                }
            },
            "required": ["file_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive download tool")

        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        headers = {"Authorization": f"Bearer {access_token}"}
        supports_all_drives = arguments.get("supports_all_drives", True)
        include_metadata = arguments.get("include_metadata", True)

        metadata: Optional[Dict[str, Any]] = None
        if include_metadata:
            meta_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            meta_params = {
                "fields": "id,name,mimeType,size,modifiedTime,parents,driveId",
                "supportsAllDrives": supports_all_drives,
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    meta_resp = await http_client.get(meta_url, headers=headers, params=meta_params)
                    if meta_resp.is_error:
                        raise _http_exception_from_response(meta_resp, "Google Drive metadata fetch (pre-download) error")
                    metadata = meta_resp.json()
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Unexpected metadata fetch error: {e}")
                raise HTTPException(status_code=500, detail=f"Error fetching metadata: {str(e)}")

        export_mime_type = arguments.get("export_mime_type")
        is_export = bool(export_mime_type)

        # If not explicitly exporting and this is a Google Workspace doc, ask for export_mime_type.
        if (not is_export) and metadata and str(metadata.get("mimeType", "")).startswith("application/vnd.google-apps."):
            raise HTTPException(
                status_code=400,
                detail=(
                    "This file is a Google Workspace document. Provide export_mime_type "
                    "(e.g., application/pdf) to download via files.export."
                )
            )

        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                if is_export:
                    url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
                    params = {"mimeType": export_mime_type}
                    logger.info(f"Exporting Drive file file_id={file_id} mimeType={export_mime_type}")
                    resp = await http_client.get(url, headers=headers, params=params)
                else:
                    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                    params = {
                        "alt": "media",
                        "acknowledgeAbuse": arguments.get("acknowledge_abuse", False),
                        "supportsAllDrives": supports_all_drives,
                    }
                    logger.info(f"Downloading Drive file (alt=media) file_id={file_id}")
                    resp = await http_client.get(url, headers=headers, params=params)

                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive download/export error")

                content_b64 = base64.b64encode(resp.content).decode("utf-8")
                return {
                    "file_id": file_id,
                    "exported": is_export,
                    "export_mime_type": export_mime_type if is_export else None,
                    "metadata": metadata,
                    "content_base64": content_b64,
                    "content_length": len(resp.content),
                }
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive download/export request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected download/export error: {e}")
            raise HTTPException(status_code=500, detail=f"Error downloading/exporting Drive file: {str(e)}")


class GoogleDriveUploadFileTool(BaseTool):
    """
    Upload file bytes:
    - uploadType=multipart: metadata + bytes (recommended for <= 5MB)
    - uploadType=media: bytes only
    """

    name = "google_drive_upload_file"
    description = "Upload a file to Google Drive (multipart or media). Expects base64 content."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File name (e.g. report.pdf)"},
                "mime_type": {"type": "string", "description": "MIME type of the uploaded content", "default": "application/octet-stream"},
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of parent folder IDs"
                },
                "content_base64": {"type": "string", "description": "Base64-encoded file bytes"},
                "upload_type": {
                    "type": "string",
                    "description": "Upload type: multipart (metadata+media) or media (media only).",
                    "enum": ["multipart", "media"],
                    "default": "multipart"
                },
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "id,name,mimeType,parents,modifiedTime,size,webViewLink"
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": ["name", "content_base64"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive upload tool")

        name = arguments.get("name")
        content_b64 = arguments.get("content_base64")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not content_b64:
            raise HTTPException(status_code=400, detail="content_base64 is required")

        mime_type = arguments.get("mime_type", "application/octet-stream")
        parents = arguments.get("parents")
        upload_type = arguments.get("upload_type", "multipart")
        fields = arguments.get("fields", "id,name,mimeType,parents,modifiedTime,size,webViewLink")

        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="content_base64 is not valid base64")

        headers = {"Authorization": f"Bearer {access_token}"}

        # Upload endpoint
        base_url = "https://www.googleapis.com/upload/drive/v3/files"
        params: Dict[str, Any] = {
            "uploadType": upload_type,
            "fields": fields,
            "supportsAllDrives": arguments.get("supports_all_drives", True),
        }

        metadata: Dict[str, Any] = {"name": name}
        if parents:
            metadata["parents"] = parents
        if mime_type:
            metadata["mimeType"] = mime_type

        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                if upload_type == "media":
                    logger.info(f"Uploading Drive file (media) name={name}")
                    resp = await http_client.post(
                        base_url,
                        headers={**headers, "Content-Type": mime_type},
                        params=params,
                        content=content_bytes,
                    )
                else:
                    logger.info(f"Uploading Drive file (multipart) name={name}")
                    mp = _encode_multipart_related(metadata, content_bytes, mime_type)
                    resp = await http_client.post(
                        base_url,
                        headers={**headers, "Content-Type": mp["content_type"]},
                        params=params,
                        content=mp["body"],
                    )

                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive upload error")

                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive upload request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Error uploading Drive file: {str(e)}")


class GoogleDriveCreateFolderTool(BaseTool):
    """
    Create folder using files.create with folder mimeType.
    """

    name = "google_drive_create_folder"
    description = "Create a folder in Google Drive."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Folder name"},
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of parent folder IDs"
                },
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "id,name,mimeType,parents,modifiedTime,webViewLink"
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": ["name"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive create folder tool")

        name = arguments.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if arguments.get("parents"):
            body["parents"] = arguments["parents"]

        params = {
            "fields": arguments.get("fields", "id,name,mimeType,parents,modifiedTime,webViewLink"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Creating Drive folder name={name}")
                resp = await http_client.post(url, headers=headers, params=params, json=body)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive create folder error")
                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive create folder request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected create folder error: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating Drive folder: {str(e)}")


class GoogleDriveUpdateFileTool(BaseTool):
    """
    Update file metadata/content using files.update.
    - Rename: set 'name'
    - Move: add_parents/remove_parents query params
    - Optional content update: provide content_base64 and mime_type -> uploadType=multipart to /upload endpoint
    """

    name = "google_drive_update_file"
    description = "Update a Drive file (rename/move; optional content update)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
            "add_parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
            "remove_parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "name": {"type": "string", "description": "New file name (rename)"},
                "add_parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Parent IDs to add (move)."
                },
                "remove_parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Parent IDs to remove (move)."
                },
                "content_base64": {"type": "string", "description": "If provided, updates file content with these bytes (base64)."},
                "mime_type": {"type": "string", "description": "MIME type for content update", "default": "application/octet-stream"},
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "id,name,mimeType,parents,modifiedTime,size,webViewLink"
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": ["file_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive update tool")

        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        headers = {"Authorization": f"Bearer {access_token}"}
        fields = arguments.get("fields", "id,name,mimeType,parents,modifiedTime,size,webViewLink")
        supports_all_drives = arguments.get("supports_all_drives", True)

        add_parents = arguments.get("add_parents") or []
        remove_parents = arguments.get("remove_parents") or []
        add_parents_str = ",".join(add_parents) if add_parents else None
        remove_parents_str = ",".join(remove_parents) if remove_parents else None

        # Metadata patch body (only include fields you want to change)
        body: Dict[str, Any] = {}
        if arguments.get("name"):
            body["name"] = arguments["name"]

        # If content update requested, use upload endpoint with uploadType=multipart
        content_b64 = arguments.get("content_base64")
        do_content_update = bool(content_b64)

        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                if do_content_update:
                    try:
                        content_bytes = base64.b64decode(content_b64)
                    except Exception:
                        raise HTTPException(status_code=400, detail="content_base64 is not valid base64")

                    mime_type = arguments.get("mime_type", "application/octet-stream")
                    upload_url = f"https://www.googleapis.com/upload/drive/v3/files/{file_id}"
                    params: Dict[str, Any] = {
                        "uploadType": "multipart",
                        "fields": fields,
                        "supportsAllDrives": supports_all_drives,
                    }
                    if add_parents_str:
                        params["addParents"] = add_parents_str
                    if remove_parents_str:
                        params["removeParents"] = remove_parents_str

                    mp = _encode_multipart_related(body if body else {}, content_bytes, mime_type)
                    logger.info(f"Updating Drive file content file_id={file_id}")
                    resp = await http_client.patch(
                        upload_url,
                        headers={**headers, "Content-Type": mp["content_type"]},
                        params=params,
                        content=mp["body"],
                    )
                else:
                    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                    params = {
                        "fields": fields,
                        "supportsAllDrives": supports_all_drives,
                    }
                    if add_parents_str:
                        params["addParents"] = add_parents_str
                    if remove_parents_str:
                        params["removeParents"] = remove_parents_str

                    logger.info(f"Updating Drive file metadata file_id={file_id}")
                    resp = await http_client.patch(
                        url,
                        headers={**headers, "Content-Type": "application/json"},
                        params=params,
                        json=body,
                    )

                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive files.update error")
                return resp.json()

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive files.update request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected update error: {e}")
            raise HTTPException(status_code=500, detail=f"Error updating Drive file: {str(e)}")


class GoogleDriveDeleteFileTool(BaseTool):
    """
    Delete file using files.delete.
    """

    name = "google_drive_delete_file"
    description = "Permanently delete a Drive file by file_id."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
            },
            "required": ["file_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive delete tool")

        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"supportsAllDrives": arguments.get("supports_all_drives", True)}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Deleting Drive file file_id={file_id}")
                resp = await http_client.delete(url, headers=headers, params=params)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive files.delete error")
                # Drive returns empty JSON object on success
                return resp.json() if resp.text else {}
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive files.delete request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected delete error: {e}")
            raise HTTPException(status_code=500, detail=f"Error deleting Drive file: {str(e)}")


class GoogleDriveCreatePermissionTool(BaseTool):
    """
    Share file by creating a Permission using permissions.create.
    """

    name = "google_drive_create_permission"
    description = "Create a sharing permission for a Drive file (share with user/group/domain/anyone)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "type": {
                    "type": "string",
                    "description": "Permission type.",
                    "enum": ["user", "group", "domain", "anyone"]
                },
                "role": {
                    "type": "string",
                    "description": "Permission role.",
                    "enum": ["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"]
                },
                "email_address": {
                    "type": "string",
                    "description": "Required if type=user or type=group."
                },
                "domain": {
                    "type": "string",
                    "description": "Required if type=domain."
                },
                "allow_file_discovery": {
                    "type": "boolean",
                    "description": "Only for type=anyone. If false, it's 'anyone with the link'.",
                    "default": False
                },
                "send_notification_email": {
                    "type": "boolean",
                    "description": "Whether to send notification email (for user/group).",
                    "default": True
                },
                "email_message": {
                    "type": "string",
                    "description": "Custom email message if sending notification email."
                },
                "transfer_ownership": {
                    "type": "boolean",
                    "description": "Whether to transfer ownership to the new owner (requires role=owner).",
                    "default": False
                },
                "move_to_new_owners_root": {
                    "type": "boolean",
                    "description": "If transferring ownership on My Drive, move to new owner's root and remove prior parents.",
                    "default": False
                },
                "supports_all_drives": {
                    "type": "boolean",
                    "description": "Whether the requesting application supports both My Drives and shared drives.",
                    "default": True
                },
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime,deleted"
                },
            },
            "required": ["file_id", "type", "role"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive permission create tool")

        file_id = arguments.get("file_id")
        p_type = arguments.get("type")
        role = arguments.get("role")

        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")
        if not p_type:
            raise HTTPException(status_code=400, detail="type is required")
        if not role:
            raise HTTPException(status_code=400, detail="role is required")

        # Validate required identity fields
        if p_type in ("user", "group") and not arguments.get("email_address"):
            raise HTTPException(status_code=400, detail="email_address is required for type=user/group")
        if p_type == "domain" and not arguments.get("domain"):
            raise HTTPException(status_code=400, detail="domain is required for type=domain")

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {"type": p_type, "role": role}
        if arguments.get("email_address"):
            body["emailAddress"] = arguments["email_address"]
        if arguments.get("domain"):
            body["domain"] = arguments["domain"]
        if p_type == "anyone":
            body["allowFileDiscovery"] = arguments.get("allow_file_discovery", False)

        params: Dict[str, Any] = {
            "fields": arguments.get("fields", "id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime,deleted"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
            "sendNotificationEmail": arguments.get("send_notification_email", True),
            "transferOwnership": arguments.get("transfer_ownership", False),
            "moveToNewOwnersRoot": arguments.get("move_to_new_owners_root", False),
        }
        if arguments.get("email_message"):
            params["emailMessage"] = arguments["email_message"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Creating permission file_id={file_id} type={p_type} role={role}")
                resp = await http_client.post(url, headers=headers, params=params, json=body)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive permissions.create error")
                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive permissions.create request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected permissions.create error: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating Drive permission: {str(e)}")


class GoogleDriveAboutGetTool(BaseTool):
    """
    Get Drive 'about' information using about.get.
    """

    name = "google_drive_about_get"
    description = "Get information about the user, the user's Drive, and system capabilities (about.get)."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "string",
                    "description": "Partial response fields selector.",
                    "default": "user(displayName,emailAddress),storageQuota,importFormats,exportFormats,maxUploadSize,canCreateDrives"
                }
            },
            "required": []
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive about.get tool")

        url = "https://www.googleapis.com/drive/v3/about"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"fields": arguments.get("fields", "user(displayName,emailAddress),storageQuota,importFormats,exportFormats,maxUploadSize,canCreateDrives")}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info("Getting Drive about info")
                resp = await http_client.get(url, headers=headers, params=params)
                if resp.is_error:
                    raise _http_exception_from_response(resp, "Google Drive about.get error")
                return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Google Drive about.get request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected about.get error: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting Drive about info: {str(e)}")


# -----------------------------
# Register tools
# -----------------------------
register_tool(GoogleDriveListFilesTool())
register_tool(GoogleDriveGetFileMetadataTool())
register_tool(GoogleDriveDownloadFileTool())
register_tool(GoogleDriveUploadFileTool())
register_tool(GoogleDriveCreateFolderTool())
register_tool(GoogleDriveUpdateFileTool())
register_tool(GoogleDriveDeleteFileTool())
register_tool(GoogleDriveCreatePermissionTool())
register_tool(GoogleDriveAboutGetTool())
