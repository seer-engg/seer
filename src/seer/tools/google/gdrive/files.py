"""
Google Drive file operations - list, get, download, upload, update.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.gdrive.base import (
    GoogleDriveFileScopeTool,
    GoogleDriveMetadataScopeTool,
    GoogleDriveReadonlyScopeTool,
)
from seer.tools.google.gdrive.helpers import (
    _drive_file_list_schema,
    _drive_file_schema,
    _encode_multipart_related,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.gdrive.files")


class GoogleDriveListFilesTool(GoogleDriveMetadataScopeTool):
    """List/search Google Drive files with query support."""

    name = "google_drive_list_files"
    description = "List/search Google Drive files. Supports Drive query 'q' and pagination."

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

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
    ) -> Any:
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
            params=params,
        )
        return resp.json()


class GoogleDriveGetFileMetadataTool(GoogleDriveMetadataScopeTool):
    """Get Google Drive file metadata by ID."""

    name = "google_drive_get_file_metadata"
    description = "Get metadata for a Google Drive file by ID."

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

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
    ) -> Any:
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
            params=params,
        )
        return resp.json()


# Google Workspace MIME types that require export instead of download
GOOGLE_WORKSPACE_MIME_TYPES = {
    "application/vnd.google-apps.document": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "application/pdf",
    "application/vnd.google-apps.presentation": "application/pdf",
    "application/vnd.google-apps.drawing": "application/pdf",
}


class GoogleDriveDownloadFileTool(GoogleDriveReadonlyScopeTool):
    """Download Google Drive file content."""

    name = "google_drive_download_file"
    description = (
        "Download Google Drive file content. Returns a file reference for efficient "
        "handling in workflows. Google Workspace files (Docs, Sheets, Slides) are auto-exported to PDF."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file ID"},
                "mime_type": {
                    "type": "string",
                    "description": "For Google Docs export, specify target MIME type (auto-detected if not provided)"
                },
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "object",
                    "description": "File reference for use in other tools",
                    "properties": {
                        "_type": {"type": "string", "const": "workflow_file_ref"},
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "size_bytes": {"type": "integer"},
                    }
                },
                "content_base64": {
                    "type": "string",
                    "description": "File content as base64 (included when not in workflow context)"
                },
                "size_bytes": {"type": "integer"},
                "exported": {"type": "boolean", "description": "True if file was exported (Google Workspace file)"},
                "export_mime_type": {"type": "string", "description": "MIME type used for export (if exported)"},
            }
        }

    # pylint: disable=too-many-locals,too-complex  # File download requires tracking multiple state variables and conditions
    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,  # pylint: disable=unused-argument  # Part of tool interface
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        mime_type = arguments.get("mime_type")
        supports_all_drives = arguments.get("supports_all_drives", True)

        # If no mime_type specified, fetch metadata to detect Google Workspace files
        file_metadata = None
        file_mime_type = None
        if not mime_type:
            metadata_resp = await self._make_request(
                "GET",
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                access_token,
                params={"fields": "name,mimeType", "supportsAllDrives": supports_all_drives},
            )
            file_metadata = metadata_resp.json()
            file_mime_type = file_metadata.get("mimeType")

            # Auto-select export format for Google Workspace files
            if file_mime_type in GOOGLE_WORKSPACE_MIME_TYPES:
                mime_type = GOOGLE_WORKSPACE_MIME_TYPES[file_mime_type]
                logger.info("Auto-exporting Google Workspace file %s (type: %s) to %s", file_id, file_mime_type, mime_type)

        # Determine if export or download
        exported = False
        export_mime_type = None
        if mime_type:
            # Google Docs export
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            params = {"mimeType": mime_type}
            exported = True
            export_mime_type = mime_type
        else:
            # Regular download
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": supports_all_drives}

        resp = await self._make_request("GET", url, access_token, params=params)
        content = resp.content

        # Determine filename
        filename = (file_metadata or {}).get("name", f"file_{file_id}")
        if exported and export_mime_type:
            # Adjust extension for exported files
            filename = _adjust_filename_for_export(filename, export_mime_type)

        # Determine effective mime type
        effective_mime_type = export_mime_type or file_mime_type or "application/octet-stream"

        # If running in workflow context with file system available, store file and return reference
        if context and context.workflow_run_id and context.has_file_system:
            try:
                file_ref = await context.file_system.store_file(
                    user_id=context.user.user_id,
                    run_id=context.workflow_run_id,
                    filename=filename,
                    data=content,
                    mime_type=effective_mime_type,
                )
                logger.info("Stored file %s in workflow file system: %s", filename, file_ref.file_id)

                result = {
                    "file": file_ref.to_dict(),
                    "size_bytes": len(content),
                    "exported": exported,
                }
                if export_mime_type:
                    result["export_mime_type"] = export_mime_type
                return result
            except OSError as e:
                logger.warning("Failed to store file in workflow file system, falling back to base64: %s", e)

        # Fallback: return base64 encoded content (for non-workflow contexts or if storage fails)
        result = {
            "content_base64": base64.b64encode(content).decode("utf-8"),
            "size_bytes": len(content),
            "exported": exported,
        }
        if export_mime_type:
            result["export_mime_type"] = export_mime_type

        return result


def _adjust_filename_for_export(filename: str, mime_type: str) -> str:
    """Adjust filename extension based on export MIME type."""
    mime_to_ext = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "text/html": ".html",
    }
    ext = mime_to_ext.get(mime_type)
    if ext and not filename.lower().endswith(ext):
        # Remove Google Docs extensions and add new one
        base = filename
        for g_ext in [".gdoc", ".gsheet", ".gslides", ".gdraw"]:
            if base.lower().endswith(g_ext):
                base = base[:-len(g_ext)]
                break
        # Remove any existing extension if it's a common doc type
        for common_ext in [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]:
            if base.lower().endswith(common_ext):
                base = base[:-len(common_ext)]
                break
        return base + ext
    return filename


class GoogleDriveUploadFileTool(GoogleDriveFileScopeTool):
    """Upload file to Google Drive."""

    name = "google_drive_upload_file"
    description = (
        "Upload a file to Google Drive using multipart upload. "
        "Accepts either a file reference from another tool or base64-encoded content."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File name"},
                "file": {
                    "type": "object",
                    "description": "File reference from another tool (e.g., google_drive_download_file)"
                },
                "content_base64": {
                    "type": "string",
                    "description": "File content as base64 (use 'file' instead if available)"
                },
                "mime_type": {"type": "string", "description": "File MIME type", "default": "application/octet-stream"},
                "parent_folder_id": {"type": "string", "description": "Parent folder ID"},
                "description": {"type": "string"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["name"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,  # pylint: disable=unused-argument  # Part of tool interface
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        name = arguments.get("name")
        file_ref_data = arguments.get("file")
        content_b64 = arguments.get("content_base64")

        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        # Resolve file content from file reference or base64
        content_bytes, resolved_mime_type = await self._resolve_file_content(
            file_ref_data, content_b64, context
        )

        mime_type = arguments.get("mime_type") or resolved_mime_type or "application/octet-stream"
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

    async def _resolve_file_content(
        self,
        file_ref_data: Optional[Dict[str, Any]],
        content_b64: Optional[str],
        context: Optional["WorkflowRuntimeContext"],
    ) -> tuple[bytes, Optional[str]]:
        """
        Resolve file content from either a file reference or base64 data.

        Returns:
            Tuple of (content_bytes, mime_type)
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports with files module
        from seer.core.files.models import is_file_ref

        # Try file reference first
        if file_ref_data and is_file_ref(file_ref_data):
            if context and context.has_file_system:
                file_ref = context.file_system.parse_file_ref(file_ref_data)
                content_bytes = await context.file_system.get_file_content(file_ref)
                logger.info("Resolved file from workflow file system: %s", file_ref.file_id)
                return content_bytes, file_ref.mime_type
            raise HTTPException(
                status_code=400,
                detail="File reference provided but workflow file system not available"
            )

        # Fall back to base64
        if content_b64:
            try:
                return base64.b64decode(content_b64), None
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}") from e

        raise HTTPException(status_code=400, detail="Either 'file' or 'content_base64' is required")


class GoogleDriveUpdateFileTool(GoogleDriveFileScopeTool):
    """Update Google Drive file metadata and/or content."""

    name = "google_drive_update_file"
    description = "Update Google Drive file metadata and/or content."

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
        self, file_id: str, access_token: Optional[str], *, metadata: Dict[str, Any],
        params: Dict[str, Any], content_b64: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle multipart file content update."""
        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {str(e)}") from e

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

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
    ) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        metadata = self._build_update_metadata(arguments)
        params = self._build_update_params(arguments)

        content_b64 = arguments.get("content_base64")
        if content_b64:
            return await self._update_with_content(
                file_id, access_token, metadata=metadata, params=params,
                content_b64=content_b64, arguments=arguments,
            )

        # Metadata-only update
        resp = await self._make_request(
            "PATCH",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            access_token,
            params=params,
            json_body=metadata,
        )
        return resp.json()
